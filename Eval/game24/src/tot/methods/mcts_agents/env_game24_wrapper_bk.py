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
from utils_mcts import parse_action_operation,execute_step
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
colorama_init()

set_debugger()

class Game24Env(gym.Env):
    def __init__(self, env_ori, idx, reward_func, horizon=10):
        super(Game24Env, self).__init__()
        self.env = env_ori
        self.reward_func = reward_func
        self.horizon = horizon
        self.reward = None # to be updated during running
        x = env_ori.get_input(idx)  # input
        self.input = x
        self.action_history = []
        self.state_history = [self.input]
        self.ys_list = [['']]
        self.step_num = 4

    def reset(self, state=None):
        if state is None:
            self.action_history = []
            self.state_history = [self.input]
            self.ys_list = [['']]
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
        

    def transition(self, s, a, is_model_dynamic=False, predict_flag=False):
        
        state_env_ori = self.state 
        # reset the env to the transition state
        self.reset(state=s)
        state, done, reward = self.step(a)
        # reset the env to the original state
        self.reset(state=state_env_ori)
        #print(state.history[-1])
        if done or predict_flag:
            if self.reward_func=="gt":
                print("Using GT Reward for Debuging")
            else:
                reward = self.get_reward(state)
        else:
            reward = 0  # no intermediate reward
        return state, reward, done
   
    def step(self, action):
        # only support for now
        cur_ele_list =[float(ele) for ele in self.state_history[-1].split()]
        input1, input2, operator = parse_action_operation(action)
        input1, input2 = float(input1), float(input2)
        if input1 in  cur_ele_list:
            idx1 = cur_ele_list.index(input1)
            cur_ele_list.pop(idx1)
        else:
            print("Invalid Action")
            pdb.set_trace()
        if input2 in  cur_ele_list:
            idx2 = cur_ele_list.index(input2)
            cur_ele_list.pop(idx2)
        else:
            print("Invalid Action")
            pdb.set_trace()
        if operator not in ["+", "-", "*", "/"]:
            print("Invalid Action")
            pdb.set_trace()
        tmp_output = execute_step(input1, input2, operator)
        cur_ele_list.append(float(tmp_output))
        
        for idx, ele in enumerate(cur_ele_list):
            if ele.is_integer():
                cur_ele_list[idx] = int(ele)
        self.update_state(action, cur_ele_list, tmp_output)
        done =  len(cur_ele_list)==1
        reward = 1 if len(cur_ele_list)==1 and cur_ele_list[0].is_integer() and int(cur_ele_list[0]==24) else 0.0
        return self.state, done, reward
        
    def update_state(self, action, ele_list, tmp_output):
        input1, input2, operator = parse_action_operation(action)
        ele_list = sorted(ele_list)
        ele_list_str = [str(ele) for ele in ele_list]
        ele_str = " ".join(ele_list_str).strip()
        output_state_str = f"{input1} {operator} {input2} = {tmp_output} (left: {ele_str})\n"
        action_str = f"{input1} {operator} {input2} = {tmp_output}"
        self.action_history.append(action)
        self.state_history.append(ele_list)
        pdb.set_trace()
    
    def get_reward(self, state):
        if self.reward_func=="gt":
            return state.reward
        reward = call_reward_models(self.reward_func, state)
        return reward

    
    def equality_operator(self, state1, state2):
        return state1.history[-1]==state2.history[-1]

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
        
    
    def rollout_sequence(self, state):
        step = 0
        horizon = self.horizon 
        done =False
        # reset the env to the original state
        self.env.reset(state=state)
        #pdb.set_trace()
        predict_flag = False
        while (not done) and (step < horizon):
            idx = 0
            max_try_time = 5
            while idx < max_try_time :
                idx +=1
                action = self.model(state.history, is_greed=True)
                try:
                    self.env.env.parse_action(action)
                    break
                except:
                    print("Invalid action format:\n%s"%(action))
            
            if step==horizon-1:
                predict_flag = True
            cur_obs = state.history[-1]["content"]
            print("step: %d\n"%(step))
            print(f"{Fore.GREEN}{cur_obs}{Style.RESET_ALL}!")
            state, reward, done = self.env.transition(state, action, predict_flag=predict_flag)
            print("step: %d, action: %s, reward: %f, predicted reward: %f\n"%(step, action, state.reward, reward))
            step +=1
        pdb.set_trace()
        # stack reward for estimate
        self.env.reward = reward
        return state

    
    def rollout_sequence_backup(self, state):
        step = 0
        horizon = self.horizon 
        done =False
        # reset the env to the original state
        self.env.reset(state=state)
        #pdb.set_trace()
        predict_flag = False
        while (not done) and (step < horizon):
            idx = 0
            max_try_time = 5
            while idx < max_try_time :
                idx +=1
                action = self.model(state.history)
                try:
                    self.env.env.parse_action(action)
                    break
                except:
                    print("Invalid action format:\n%s"%(action))
            
            if step==horizon-1:
                predict_flag = True
            cur_obs = state.history[-1]["content"]
            print("step: %d\n"%(step))
            print(f"{Fore.GREEN}{cur_obs}{Style.RESET_ALL}!")
            state, reward, done = self.env.transition(state, action, predict_flag=predict_flag)
            print("step: %d, action: %s, reward: %f, predicted reward: %f\n"%(step, action, state.reward, reward))
            step +=1
        # stack reward for estimate
        self.env.reward = reward

    def get_top_k_tokens(self, state):
        ys = self.env.ys_list[-1]
        x = self.env.input
        task = self.env.env
        args = self.args
        if args.method_generate == 'sample':
            step = len(self.env.state[1])
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, self.model, x, y) for y in ys]
        unique_act_list, unique_score_list = prepare_action_and_scores(new_ys)
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
            print(act)
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
    # build env 
    env_mcts = Game24Env(task, idx, args.reward_func, args.horizon) 
    # build policy model
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    default_policy = localAPIPolicy(model=gpt, env=env_mcts, horizon=args.horizon, seq_num=args.seq_num, args=args)
    agent = uct.UCT(default_policy=default_policy, rollouts=args.rollouts)
    ys = ['']  # current output candidates
    infos = []
    done = False
    for step in range(task.steps):
        act = agent.act(env_mcts, done)  
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        import pdb
        pdb.set_trace()

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}


if __name__=="__main__":
    
    test_main() 

