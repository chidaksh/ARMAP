from abc import abstractmethod
from functools import partial
import argparse
import copy
import gym
import pdb
import os
import re
import random
import sys
from tot.models import gpt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
LIB_DIR_DIR= os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(LIB_DIR_DIR)
sys.path.append(SCRIPT_DIR)
print(SCRIPT_DIR)
import uct
from utils_mcts import call_reward_models, get_proposals, get_samples, set_debugger
from utils_mcts import parse_action_operation, execute_step, rollout_partial_sequence, rollout_partial_sequence_v2, rollout_partial_sequence_v3
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import json
colorama_init()

set_debugger()

class Game24Env(gym.Env):
    def __init__(self, env_ori, idx, reward_func, horizon=10, args=None):
        super(Game24Env, self).__init__()
        self.env = env_ori
        self.reward_func = reward_func
        self.horizon = horizon
        self.reward = None # to be updated during running
        x = env_ori.get_input(idx)  # input
        self.input = x
        self.action_history = []
        self.state_history = []
        self.multi_state_history = [] 
        self.ys_list = ['']
        self.idx = idx
        self.args = args

    def reset(self, state=None):
        if state is None:
            self.action_history = []
            self.state_history = [self.input]
            self.ys_list = ['']
        else:
            state_history, action_history, ys_list = state 
            self.action_history = copy.deepcopy(action_history)
            self.state_history = copy.deepcopy(state_history)
            self.ys_list = copy.deepcopy(ys_list)
            #pdb.set_trace()

    @property
    def state(self):
        state_history = copy.deepcopy(self.state_history)
        action_history = copy.deepcopy(self.action_history)
        ys_list = copy.deepcopy(self.ys_list)
        return state_history, action_history, ys_list 
        

    def transition(self, s, a, is_model_dynamic=False):
        state_env_ori = self.state 
        # reset the env to the transition state
        self.reset(state=s)
        state, done, reward_gt = self.step(a)
        # reset the env to the original state
        if done:
            reward = self.get_reward(state)
        else:
            reward = 0  # no intermediate reward
        self.reset(state=state_env_ori)
        return state, reward, done, reward_gt
  
    def step(self, action):
        self.action_history.append(action)
        self.state_history.append(action)
        new_action = self.ys_list[-1]+action+"\n"
        new_action = new_action.replace("\n\n\n", "\n")
        new_action = new_action.replace("\n\n", "\n")
        self.ys_list.append(new_action)
        if "Answer" in action:
            reward = self.env.test_output(self.idx, action)["r"]
            done = True
        else:
            done = False
            reward = 0.0
        return self.state, done, reward
    
    def get_reward(self, state):
        if self.reward_func=="gt":
            reward = self.env.test_output(self.idx, state[2][-1])["r"]
            print("Debuging with GT rewards: %f."%(reward))
            return reward 
        else:
            state_dict = {"trajectory": state[2][-1], "input": self.input}
            reward = call_reward_models(self.reward_func, state_dict, job_id=self.args.job_id)
        return reward

    
    def equality_operator(self, state1, state2):
        return state1[2][-1]==state2[2][-1]

def build_uct_agent():
    default_policy = GeminiDefaultPolicy(
        env=env,
        horizon=horizon,
        model=args.model
    )
    
    agent = uct.UCT(
        default_policy=default_policy,
    )


class DefaultPolicy:
    def __init__(self, env: gym.Env, horizon: int):
        """
        Args:
            k: number of top k predictions to return
            env: environment
            horizon: horizon of the environment (the maximum number of steps in an episode)
        """
        self.env = env
        self.horizon = horizon

    @abstractmethod
    def rollout_sequence(self, state, horizon: int = None):
        pass

    @abstractmethod
    def get_top_k_tokens(self, state):
        pass

class localAPIPolicy(DefaultPolicy):
    """
    Default policy that uses Google Gemini model.
    """
    def __init__(
            self,
            model,
            env,
            horizon,
            seq_num,
            args
    ):
        self.model = model
        self.env = env
        self.seq_num = seq_num
        self.env.reward = None # to be updated during running
        self.horizon = horizon
        self.args = args
        self.rollout_cache = {}
        self.reward_num = 0 
        self.total_num = 0
        self.max_pred = -100
        self.max_correct_pred = -100
     
    
    def rollout_sequence_bk(self, state):
        args = self.args
        horizon = self.horizon 
        done =False
        state_ori = self.env.state
        self.env.reset(state=state)
        x = self.env.input
        task = self.env.env
        step = len(state[2]) - 1
        input_str = self.env.state[2][-1]
        cache_id = x + "\n" + input_str
        print(cache_id)
        if cache_id not in self.rollout_cache:
            #full_act = rollout_partial_sequence(x, input_str, args.n_generate_sample, temperature=0.0, max_tokens=self.args.max_tokens)
            full_act = rollout_partial_sequence_v2(x, input_str, args.n_generate_sample, temperature=0.0, max_tokens=self.args.max_tokens)
            #full_act = rollout_partial_sequence_v3(x, input_str, args.n_generate_sample, temperature=args.temperature, max_tokens=self.args.max_tokens)
            print(full_act)
            print("length: %d"%(len(full_act)))
            pdb.set_trace()
            act_list = full_act.split("\n")
            for act_id, action in enumerate(act_list):  
                state, reward, done, reward_gt = self.env.transition(state, action)
            print("step: %d, action: %s, reward: %f, predicted reward: %f\n"%(step+act_id, action, reward_gt, reward))
            self.rollout_cache[cache_id] = full_act, state, reward, done 
            self.reward_num += reward_gt
            self.total_num += 1
            if reward_gt==1:
                if reward > self.max_correct_pred:
                    self.max_correct_pred = reward 
            if reward > self.max_pred:
                self.max_pred = reward
            if self.max_correct_pred >=self.max_pred:
                print("Correct Pred: %f, %f\n"%(self.max_correct_pred, self.max_pred)) 
            else:
                print("Wrong Pred: %f, %f\n"%(self.max_correct_pred, self.max_pred)) 
            print("accuracy: %d/%d\n"%(self.reward_num, self.total_num))
            #pdb.set_trace()
        else:
            full_act, state, reward, done = self.rollout_cache[cache_id]
        self.env.reset(state=state_ori)
        # stack reward for estimate
        self.env.reward = reward
        return state


    def rollout_sequence(self, state):
        args = self.args
        horizon = self.horizon 
        done =False
        state_ori = self.env.state
        self.env.reset(state=state)
        x = self.env.input
        task = self.env.env
        step = len(state[2]) - 1
        input_str = self.env.state[2][-1]
        cache_id = x + "\n" + input_str
        print(cache_id)
        max_state, tmp_max_reward = None, -1000
        if cache_id not in self.rollout_cache:
            #full_act = rollout_partial_sequence(x, input_str, args.n_generate_sample, temperature=0.0, max_tokens=self.args.max_tokens)
            full_act = rollout_partial_sequence_v3(x, input_str, args.n_generate_sample, temperature=args.temperature, max_tokens=self.args.max_tokens)
            if isinstance(full_act, list): 
                for tmp_act_id, tmp_act in enumerate(full_act):
                    tmp_act_list = tmp_act.split("\n")
                    state_new, reward, done, reward_gt = self.env.transition(state, tmp_act)
                    print("step: %d, action: %s, reward: %f, predicted reward: %f\n"%(tmp_act_id, tmp_act, reward_gt, reward))
                    self.reward_num += reward_gt
                    self.total_num += 1
                    self.env.multi_state_history.append((state_new, reward_gt, reward))
                    print("accuracy: %d/%d\n"%(self.reward_num, self.total_num))
                    if reward_gt==1:
                        if reward > self.max_correct_pred:
                            self.max_correct_pred = reward 
                    if reward > self.max_pred:
                        self.max_pred = reward
                    if reward > tmp_max_reward:
                        max_state = state_new
                        tmp_max_reward = reward
                    if self.max_correct_pred >=self.max_pred:
                        print("Correct Pred: %f, %f\n"%(self.max_correct_pred, self.max_pred)) 
                    else:
                        print("Wrong Pred: %f, %f\n"%(self.max_correct_pred, self.max_pred)) 
                state = max_state 
            else:
                print(full_act)
                print("length: %d"%(len(full_act)))
                act_list = full_act.split("\n")
                for act_id, action in enumerate(act_list):  
                    state, reward, done, reward_gt = self.env.transition(state, action)
                print("step: %d, action: %s, reward: %f, predicted reward: %f\n"%(step+act_id, action, reward_gt, reward))
                self.rollout_cache[cache_id] = full_act, state, reward, done 
                self.reward_num += reward_gt
                self.total_num += 1
                print("accuracy: %d/%d\n"%(self.reward_num, self.total_num))
        else:
            full_act, state, reward, done = self.rollout_cache[cache_id]
        self.env.reset(state=state_ori)
        # stack reward for estimate
        self.env.reward = reward
        if state is None:
            pdb.set_trace()
        return state


    def get_top_k_tokens(self, state):
        ys = [self.env.ys_list[-1] ]
        x = self.env.input
        task = self.env.env
        args = self.args
        if args.method_generate == 'sample':
            step = len(self.env.state[1])
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, self.model, x, y) for y in ys]
        unique_act_list, unique_score_list = prepare_action_and_scores(new_ys)
        #print(unique_act_list)
        #pdb.set_trace()
        return unique_act_list, unique_score_list

def prepare_action_and_scores(new_ys):
    raw_ys = []
    for ys_list in new_ys:
        raw_ys.extend(ys_list) 
    ys_dict = {} 
    output_ys = []
    unique_act_list = []
    for act in raw_ys:
        try:
            act_key = act.split("(left")[1] 
            #print(act)
            unique_act_list.append(act)
        except:
            print("Invalid format: %s.\n"%(act))
            continue
        if act_key not in ys_dict:
            ys_dict[act_key] = 1
        else:
            ys_dict[act_key] += 1
    sum_fre = sum(list(ys_dict.values()))
    unique_score_list = []
    for act in unique_act_list:
        act_key = act.split("(left")[1]
        tmp_score = ys_dict[act_key] / sum_fre
        unique_score_list.append(tmp_score)
    return unique_act_list, unique_score_list 

def mcts_solve(args, task, idx, to_print=True):
    output_path = os.path.join(args.output_dir, f"{idx}.json") 
    if os.path.isfile(output_path):
        print("%s exists. Skipping."%(output_path))
        return 1
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    # build env 
    env_mcts = Game24Env(task, idx, args.reward_func, args.horizon, args) 
    # build policy model
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    default_policy = localAPIPolicy(model=gpt, env=env_mcts, horizon=args.horizon, seq_num=args.seq_num, args=args)
    agent = uct.UCT(default_policy=default_policy, rollouts=args.rollouts)
    ys = ['']  # current output candidates
    infos = []
    done = False
    for step in range(task.steps):
        act = agent.act(env_mcts, done)
        output_dict = {"trajectories": agent.rolled_out_trajectories, "rewards": agent.rolled_out_rewards}
        with open(output_path, "w") as fh:
            json.dump(output_dict, fh)
        output_multi_path = os.path.join(args.output_dir, f"multi_{idx}.json") 
        with open(output_multi_path, "w") as fh:
            json.dump(env_mcts.multi_state_history, fh)
        #pdb.set_trace()
        break


if __name__=="__main__":
    test_main() 

