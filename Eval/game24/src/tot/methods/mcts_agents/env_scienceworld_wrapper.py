from abc import abstractmethod
import argparse
import copy
import gym
import pdb
import os
import re
import random
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
LIB_DIR_DIR= os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(LIB_DIR_DIR)
import uct
from utils_mcts import call_reward_models
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
colorama_init()
from debug_mcts import load_env_debug 
from eval_agent.agents.openllm_lm_agent import OPENLLMLMAgent

class sciWEnv(gym.Env):
    def __init__(self, env_ori, reward_func, horizon=10):
        super(sciWEnv, self).__init__()
        self.env = env_ori
        self.reward_func = reward_func
        self.horizon = horizon
        self.reward = None # to be updated during running
        self.session_index = 0
        #pdb.set_trace()

    def reset(self, state=None):
        self.env.reset()
        if state is None:
            return 0
        instruct_len = len(self.env.state.history)
        # we might need to check the states
        #print("Debuging reset function\n")
        #self.env.state = copy.deepcopy(state)
        for idx, step_info in enumerate(state.history):
            if idx < instruct_len:
                continue
            if step_info["role"]!="assistant":
                continue
            self.env.step(step_info["content"])
        if self.env.state.history[-1]!=state.history[-1] and "adult bee" not in state.history[-1]:
            print("Warning: Inconsistence between states.")
            print("=======================")
            print("Original state: ")
            print("=======================")
            print(state.history[-1]['content'])
            
            print("\n\n=======================")
            print("Reset state: ")
            print("=======================")
            print(self.env.state.history[-1]['content'])
        assert len(self.env.state.history)==len(state.history)
        #assert self.env.state.reward==state.reward
        if self.env.state.reward!=state.reward:
            print("Warning: Inconsistence between rewards.")
            print(self.env.state.reward)
            print(state.reward)

    @property
    def state(self):
        state = copy.deepcopy(self.env.state)
        return state 

    def check_done(self, state, predict_flag=False):
        done = False
        begin = -1
        for i, conv in enumerate(state.history):
            if 'Task Description:' in conv['content']:
                begin = i
                break
        assert begin!=-1 
        observation_history = [ ele["content"] for ele in state.history[begin:] if ele["role"]=="user"]
        observation = observation_history[-1] 
        if predict_flag:
            done = True
            state.finished = True
        elif len(observation_history) >=self.horizon+3:
            done = True
            state.finished = True
            print("Done after max steps: %d\n"%(self.horizon-1))
        else:
            if 'No known action matches that input' in observation:
                done = True
                state.finished = True
                print("rollout ended by %s\n"%observation)
            # prevent repeat and dead loop
            if len(observation_history) > 4 and observation == observation_history[-1] and observation == \
                    observation_history[-2] and observation == observation_history[-3] and observation == \
                    observation_history[-4]:
                observation = 'Same Observation Repeat Four Times'
                print(observation_history[-1])
                print(observation)
                state.finished = True
        #if done:
        #    pdb.set_trace()
        return done, state
        

    def transition(self, s, a, is_model_dynamic=False, predict_flag=False):
        
        state_env_ori = self.state 

        # reset the env to the transition state
        self.reset(state=s)
        observation, state = self.step_without_state_update(a)
        reward = state.reward
        done = state.finished
        # reset the env to the original state
        self.reset(state=state_env_ori)
        #print(state.history[-1])
        if done or predict_flag:
            if self.reward_func=="gt":
                print("Using GT Reward for Debuging")
            else:
                reward = self.get_reward(state)
        else:
            #print("Using GT reward")
            reward = 0  # no intermediate reward
        return state, reward, done
    
    
    def step_without_state_update(self, action):
        """
        record state history and action history for analysis
        """
        observation, state = self.env.step(action)
        done, state = self.check_done(state, predict_flag=False)
        return observation, state
    
    
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


class GeminiDefaultPolicy(DefaultPolicy):
    """
    Default policy that uses Google Gemini model.
    """
    def __init__(
            self,
            model,
            env: gym.Env,
            horizon: int,
    ):
        self.model = model


class localAPIPolicy(DefaultPolicy):
    """
    Default policy that uses Google Gemini model.
    """
    def __init__(
            self,
            model,
            env: gym.Env,
            horizon: int,
            seq_num: int = 3
    ):
        self.model = model
        self.env = env
        self.seq_num = seq_num
        self.env.reward = None # to be updated during running
        self.horizon = horizon
   
    
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
        if not isinstance(self.model, OPENLLMLMAgent):
            # if not support batch decoding
            return self.get_top_k_tokens_naive(state=state)
        else:
            return self.get_top_k_tokens_batch(state)
        
    def get_top_k_tokens_batch(self, state):
        action_list_raw = self.model(state.history, num_seq=self.seq_num, beam_search=True)
        #action_list_raw = self.model(state.history, num_seq=self.seq_num, beam_search=False)
        frq_dict = {}
        key2llm = {}
        action_list = []
        for llm_output in action_list_raw:
            try:
                self.env.env.parse_action(llm_output)
                action_id = llm_output.strip().split("\n")[-1]
                if action_id not in key2llm:
                    frq_dict[llm_output] = 1
                    key2llm[action_id] = llm_output
                    action_list.append(llm_output)
                else:
                    prev_output = key2llm[action_id]
                    frq_dict[prev_output] +=1
            except:
                print("Invalid action format:\n%s"%(llm_output))
        sum_seq = sum(list(frq_dict.values()))
        score_list = [frq_dict[act] * 1.0 / sum_seq  for act in action_list ]
        print(score_list)
        temp = 1.0
        score_list = [ele**temp for ele in score_list]
        score_list = [ele / sum(score_list) for ele in score_list]
        return action_list, score_list
    
    def get_top_k_tokens_naive(self, state):
        #pdb.set_trace()
        action_list = []
        max_try_time = self.seq_num * 2
        idx = 0
        frq_dict = {}
        key2llm = {}
        while len(action_list) < self.seq_num and idx < max_try_time :
            idx +=1
            llm_output = self.model(state.history)
            try:
                self.env.env.parse_action(llm_output)
                action_id = llm_output.strip().split("\n")[-1]
                if action_id not in key2llm:
                    action_list.append(llm_output)
                    frq_dict[llm_output] = 1
                    key2llm[action_id] = llm_output
                else:
                    prev_output = key2llm[action_id]
                    frq_dict[prev_output] +=1
            except:
                print("Invalid action format:\n%s"%(llm_output))
        sum_seq = sum(list(frq_dict.values()))
        score_list = [frq_dict[act] * 1.0 / sum_seq  for act in action_list ]
        print(score_list)
        temp = 1.0
        score_list = [ele**temp for ele in score_list]
        score_list = [ele / sum(score_list) for ele in score_list]
        #print(score_list)
        #print(action_list)
        #if len(score_list)==0:
        #    pdb.set_trace()
        return action_list, score_list
    
    def get_top_k_tokens_backup(self, state):
        action_list = []
        unique_action_list = []
        #for idx in range(self.seq_num):
        max_try_time = 20
        idx = 0
        while len(unique_action_list) < self.seq_num and idx < max_try_time :
            idx +=1
            llm_output = self.model(state.history)
            try:
                self.env.env.parse_action(llm_output)
                action_list.append(llm_output)
            except:
                print("Invalid action format:\n%s"%(llm_output))
            unique_action_list = list(set(action_list))
        frq_dict = {}
        for act in action_list:
            if act in frq_dict:
                frq_dict[act] += 1
            else:
                frq_dict[act] = 1
        score_list = [frq_dict[act] * 1.0 / len(action_list)  for act in unique_action_list ]
        print(score_list)
        temp = 1.0
        score_list = [ele**0.5 for ele in score_list]
        score_list = [ele / sum(score_list) for ele in score_list]
        print(score_list)
        print(unique_action_list)
        return unique_action_list, score_list

def test_main():
    pdb.set_trace()

if __name__=="__main__":
    
    test_main() 

