
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
LIB_DIR_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(LIB_DIR_DIR)

from src.client.agents.new_local_agent import NewLocal
from src.mcts_agents.utils import parse_args, gemini, build_local_model, inference_local_api, save_trajectory_history
from functools import partial
from src.typings import AssignmentConfig, SampleIndex, TaskOutput, TaskClientOutput
from src.configs import ConfigLoader
from src.server.tasks.webshop_docker.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
import src.mcts_agents.uct as uct
from abc import abstractmethod
import argparse
import copy

import gym
import pdb
import re
import random
import requests

from html2image import Html2Image
from PIL import Image
import io
from io import BytesIO
import base64
import json
import imgkit
import numpy as np

from tqdm import tqdm
import sys



def encode_image(image, encoding_format="PNG"):
    buffered = BytesIO()
    image.save(buffered, format=encoding_format)
    buffered.seek(0)
    return buffered


class webshopEnv(gym.Env):
    def __init__(self, env_ori, args):
        super(webshopEnv, self).__init__()
        self.env = env_ori
        self.reward_func = args.rm_api
        self.horizon = args.horizon
        self.reward = None  # to be updated during running
        self.session_index = env_ori.session
        self.llm_api = args.llm_api

    def reset(self, session, state=None):
        self.session_index = session
        self.env.reset(session=session)
        if state is not None:
            try:
                action_list = state[1]
            except:
                pdb.set_trace()
            for act in action_list:
                tmp_out = self.env.step(act)
        st = self.state
        with open("state1.txt", "w") as fh:
            fh.write(state[0][-1])
        with open("st.txt", "w") as fh:
            fh.write(st[0][-1])
        if state is not None and len(action_list) > 0:
            with open("tmp_out.txt", "w") as fh:
                fh.write(tmp_out[0])
            assert state[0][-1] == tmp_out[0]
        if state[0][-1] != st[0][-1]:
            pdb.set_trace()

    @property
    def state(self):
        previous_states = copy.deepcopy(self.env.prev_obs)
        previous_actions = copy.deepcopy(self.env.prev_actions)
        return (previous_states, previous_actions)

    def transition(self, s, a, is_model_dynamic=False):

        action_name, action_arg = parse_action(a)

        state_env_ori = self.state

        # reset the env to the transition state
        self.reset(session=self.session_index, state=s)
        html_observation, reward_gt, done, info = self.step_without_state_update(
            a)
        if "stop" in action_name.lower() or len(s) == self.horizon or (action_arg and "buy now" in action_arg.lower()):
            # either the text finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            #reward_pd = self.get_reward(s)
            # get reward for the update state
            reward_pd = self.get_reward(html_observation)
        else:
            reward_pd = 0  # no intermediate reward
        # reset the env to the original state
        self.reset(session=self.session_index, state=state_env_ori)
        return html_observation, reward_pd, done, reward_gt

    def step_without_state_update(self, action):
        """
        record state history and action history for analysis
        """
        html_observation, reward, done, info = self.env.step(action)
        previous_states = copy.deepcopy(self.env.prev_obs)
        previous_actions = copy.deepcopy(self.env.prev_actions)
        return (previous_states, previous_actions), reward, done, info

    def get_reward(self, state):
        preference_list, factual_list, files = self.convert_history_to_evaluator(
            state)
        # preference_data = open(preference_list).read()
        preference_data = json.dumps(preference_list)
        factual_data = json.dumps(factual_list)
        # factual_data = open(factual_list).read()
        prompt = open(
            "/root/workspace/prompt/fact_rlhf_reward_prompt_wj2.txt").read()
        #print("converted")
        data = dict(
            preference_data=preference_data,
            factual_data=factual_data,
            prompt=prompt,
        )
        headers = {
            "User-Agent": "BLIP-2 HuggingFace Space",
        }
        response = requests.post(
            self.reward_func, data=data, files=files, headers=headers)
        print(response.content)
        decoded_string = response.content.strip()[1:-1]
        reward = float(decoded_string)
        return reward

    def equality_operator(self, state1, state2):
        return state1[0][-1] == state2[0][-1]

    def crop(self, image):
        image_ori = image
        image_data = image.getdata()

        # 获取图片的边界像素值
        bg_color = image_data.getpixel((0, 0))

        # 寻找左上角坐标
        left = 0
        top = 0
        while image_data.getpixel((left, top)) == bg_color:
            left += 1
            if left >= image.width:
                left = 0
                top += 1
            if top >= image.height:
                break

        # 寻找右下角坐标
        right = image.width - 1
        bottom = image.height - 1
        while image_data.getpixel((right, bottom)) == bg_color:
            right -= 1
            if right < 0:
                right = image.width - 1
                bottom -= 1
            if bottom < 0:
                break
        image = image.crop((0, 0, image.width, bottom + 1))
        return image

    def save_screenshot(self, html2img, html_content, html_file_path, image_file_path):
        file_path = html_file_path
        with open(file_path, "w") as html_file:
            html_file.write(html_content)
        output_path = '/root/workspace/file_cache/image/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        options = {'enable-local-file-access': None}
        imgkit.from_file(
            file_path, f'/root/workspace/file_cache/image/{image_file_path}', options=options)
        image = Image.open(
            f'/root/workspace/file_cache/image/{image_file_path}')
        image = self.crop(image)
        image_stream = io.BytesIO()
        image.save(image_stream, format="JPEG")
        image_stream.seek(0)
        imagestr = base64.b64encode(image_stream.read()).decode('utf-8')
        return imagestr

    def convert_history_to_evaluator(self, history):
        previous_states, previous_actions = history
        # previous_actions.append(last_action)
        h2i = Html2Image(output_path='/root/workspace/file_cache/image/',
                         custom_flags=['--no-sandbox', '--disable-gpu'])

        def encode_image(image, encoding_format="PNG"):
            buffered = BytesIO()
            image.save(buffered, format=encoding_format)
            buffered.seek(0)
            return buffered

        preference_list = []
        factual_list = {}
        observation_0 = self.env.convert_html_to_text(
            previous_states[0], simple=True)
        intent = observation_0.split("[SEP]")[2]
        # sample = history
        files = {}
        images = {}
        encoding_format = "PNG"
        preference = None
        print("Converting")
        # env.convert_html_to_text(html_observation, simple=True)
        for sample_index, item in enumerate(previous_actions):
            imagestr = self.save_screenshot(h2i, previous_states[sample_index],
                                            f"/root/workspace/file_cache/html/html_{sample_index}_ori.html",
                                            image_file_path=f"image_{sample_index}_ori.jpg")
            img = imagestr
            try:
                prediction = f'Action:\n{item}'
            except:
                pdb.set_trace()
            # intent = intent_dict[task_id]
            # dir_id = j
            img_name = f"image_task_{sample_index}.png"
            full_image_dir = f"./image_tmp/"
            if not os.path.exists(full_image_dir):
                os.makedirs(full_image_dir)
            full_image = full_image_dir + img_name
            image_data = base64.b64decode(img)
            image_bytes = BytesIO(image_data)
            image = Image.open(image_bytes)
            image.save(full_image)

            images[img_name] = Image.open(full_image)
            factual_list[img_name] = [self.env.convert_html_to_text(
                previous_states[sample_index], simple=True)]

            if not os.path.exists(full_image):
                print("image no exist")
                break

            # dir_id = int(dir_name.rsplit('_', 1)[1])
            image_id = sample_index

            action = f"<image>\nNavigation Intent: {intent}"

            conversations = []
            conversations.append({
                "from": "human",
                "value": action
            })

            conversations.append({
                "from": "gpt",
                "value": "###test gpt###"
            })

            output_1 = {
                "from": "llava",
                "value": prediction
            }

            if not preference:
                preference = dict(
                    image=img_name,
                    conversations=conversations,
                    output_1=[output_1],
                )
            else:
                preference["output_1"].append({
                    "from": "human",
                    "image": img_name,
                    "value": "Current screenshot: <image>. Observation: <obs>."
                })
                preference["output_1"].append(output_1)
        preference["output_1"].append({
            "from": "human",
            "value": "Please evaluate whether your last response achieves the \"Navigation Intent\" or not"
        })
        preference["output_1"].append({
            "from": "llava",
            "value": "Following your definitions, the score of my last response is"
        })
        preference_list.append(preference)
        print("images")
        for k, v in images.items():
            image = encode_image(v, encoding_format=encoding_format)
            files[k] = image
        return preference_list, factual_list, files


def parse_action(action):
    pattern = re.compile(r'(.+)\[(.+)\]')
    m = re.match(pattern, action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()
    return action_name, action_arg


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
            env: gym.Env,
            horizon: int,
            seq_num: int = 5,
            beam_search: bool = True
    ):
        if isinstance(model, NewLocal):
            self.model = partial(inference_local_api, model=model)
        else:
            self.model = partial(model, llm_api=env.llm_api)
        self.env = env
        self.seq_num = seq_num
        self.horizon = horizon
        self.env.reward = None  # to be updated during running
        self.reward_gt_list = []
        self.beam_search = beam_search

    def rollout_sequence(self, state, horizon: int = -1):
        step = 0
        done = False
        if horizon == -1:
            horizon = self.horizon
        # reset the env to the original state
        self.env.reset(session=self.env.session_index, state=state)
        
        obs = self.env.env.convert_html_to_text(state[0][-1], simple=True)
        print(obs)
        step = len(state[1])
        while (not done) and (step < horizon):
            response = self.model(self.env, state, is_greedy=True)
            try:
                action = re.search(
                    r"[Aa]ction: *(?:\n|\\n)* *((search|click)\[.+?])", response
                ).group(1)
                # action = re.search(
                #     r"[Aa]ction: *\n* *((search|click)\[.+?])", response
                # ).group(1)
            except:
                action = response
            state, reward_pd, done, reward_gt = self.env.transition(
                state, action)
            print("step: %d, action: %s, predict: %f, GT: %f\n" %
                  (step, action, reward_pd, reward_gt))
            step += 1
        # stack reward for estimate
        self.env.reward = reward_pd
        self.reward_gt_list.append(reward_gt)
        return state, reward_pd, done, reward_gt

    def get_top_k_tokens(self, state):
        if self.model.func.__name__=="gemini":
            # if not support batch decoding
            return self.get_top_k_tokens_naive(state=state)
        if self.model.func.__name__=="inference_local_api":
            return self.get_top_k_tokens_batch(state)
        else:
            print("Not implemented\n")
            pdb.set_trace()


    def get_top_k_tokens_naive(self, state):
        action_list = []
        max_try_time = self.seq_num * 2 
        idx = 0
        while len(action_list) < self.seq_num and idx < max_try_time:
            idx += 1
            llm_output = self.model(self.env, state)
            try:
                action = re.search(
                    r"[Aa]ction: *(?:\n|\\n)* *((search|click)\[.+?])", llm_output
                ).group(1)
                action_list.append(action)
            except:
                print("Invalid action format:\n%s" % (llm_output))
                action_list.append(llm_output)
        unique_action_list = list(set(action_list))
        frq_dict = {}
        for act in action_list:
            if act in frq_dict:
                frq_dict[act] += 1
            else:
                frq_dict[act] = 1
        score_list = [frq_dict[act] * 1.0 / len(action_list) for act in unique_action_list]
        pdb.set_trace()
        return unique_action_list, score_list
    
    
    def get_top_k_tokens_batch(self, state):
        action_list_raw = self.model(self.env, history=state, num_seq=self.seq_num, beam_search=self.beam_search)
        frq_dict = {}
        key2llm = {}
        action_list = []
        for llm_output in action_list_raw:
            try:
                action = re.search(
                    r"[Aa]ction: *(?:\n|\\n)* *((search|click)\[.+?])", llm_output
                ).group(1)
                action_list.append(action)
            except:
                print("Invalid action format:\n%s" % (llm_output))
                action_list.append(llm_output)
        unique_action_list = list(set(action_list))
        frq_dict = {}
        for act in action_list:
            if act in frq_dict:
                frq_dict[act] += 1
            else:
                frq_dict[act] = 1
        score_list = [frq_dict[act] * 1.0 / len(action_list) for act in unique_action_list]
        print(score_list)
        print(unique_action_list)
        #pdb.set_trace()
        return unique_action_list, score_list


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


class randomPolicy(DefaultPolicy):
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
        self.env = env
        self.env.reward = None  # to be updated during running

    def rollout_sequence(self, state, horizon: int = 8):
        step = 0
        done = False
        # reset the env to the original state
        self.env.reset(session=self.env.session_index, state=state)
        # pdb.set_trace()
        while (not done) and (step < horizon):
            action_list, score_list = self.get_top_k_tokens(state)
            action = action_list[0]
            state, reward, done = self.env.transition(state, action)
            print("step: %d, action: %s, reward: %f\n" % (step, action, reward))
            step += 1
        # stack reward for estimate
        self.env.reward = reward
        # pdb.set_trace()
        return state

    def get_top_k_tokens(self, state):
        state_ori = self.env.state
        self.env.reset(self.env.session_index, state=state)
        ava_act = self.env.env.get_available_actions()
        self.env.reset(self.env.session_index, state=state_ori)
        action_list = get_action_dict(ava_act)
        score_list = [1.0 / len(action_list) for act in action_list]
        return action_list, score_list


def get_action_dict(ava_act):
    "random action dict for debuging"
    print(ava_act)
    action_raw_dict = {}
    if random.random() < 0.2:
        action_list = ["stop[]"]
    if "clickables" in ava_act and len(ava_act["clickables"]) == 1:
        str_ava = str(ava_act)
        # action_raw_dict["{'has_search_bar': True, 'clickables': ['search']}"] = \
        #     [history[-1]["response"]]
        action_raw_dict["{'has_search_bar': True, 'clickables': ['search']}"] = \
            [
                "Thought:\nI think I should use the search bar to look for the products that I need.\n\nAction:\nsearch[long clip-in hair extension]\n\n"]
        action_list = [re.search(r"[Aa]ction: *\n* *((search|click)\[.+?])", action_raw).group(1) for action_raw in
                       action_raw_dict[str_ava]]
    elif "clickables" in ava_act and len(ava_act["clickables"]) > 1:
        # action_list = ["click[%s]"%ele for ele in ava_act["clickables"] if ele not in ["back to search", "next >"]]
        action_list = ["click[%s]" % ele for ele in ava_act["clickables"] if ele not in ["back to search"]]
    elif "clickables" in ava_act and len(ava_act["clickables"]) == 0:
        action_list = ["stop[]"]
    else:
        pdb.set_trace()
    random.shuffle(action_list)
    if "click[buy now]" in action_list:
        action_list[0] = "click[buy now]"
    return action_list


def test_main(args):
    model = build_local_model(args)  if args.with_vllm_api else gemini
    
    if args.debug:
        # load a small set for debugging
        file_path = "/root/webshop/data/items_shuffle_1000.json"
        env = WebAgentTextEnv(observation_mode="html", human_goals=True, file_path=file_path)
    else:
        env = WebAgentTextEnv(observation_mode="html", human_goals=True)
    reward_list = []
    tra_list = []

    sub_folder = os.path.join(args.output_dir, args.split)
    if not os.path.isdir(sub_folder):
        os.makedirs(sub_folder)

    for session_index in tqdm(range(args.start_idx, args.end_idx)):
        env.reset(session=session_index)
        env_mcts = webshopEnv(env_ori=env, args=args)

        timesteps = 1
        done = False
        default_policy = localAPIPolicy(model=model, env=env_mcts, horizon=args.horizon, seq_num =  args.seq_num, beam_search=args.beam_search)
        agent = uct.UCT(default_policy=default_policy, rollouts=args.rollouts)
        output_path = os.path.join(args.output_dir, args.split, "%s.json"%session_index)
        if os.path.isfile(output_path):
            print("File exists, %s"%(output_path))
            continue

        for ts in range(timesteps):
            state_ori = env_mcts.state
            # env.reset(env.session_index, state=history)
            try:
                act = agent.act(env_mcts, done)
                save_trajectory_history(agent.rolled_out_trajectories, agent.rolled_out_rewards, agent.rolled_out_rewards_gt, output_path)
            except:
                print("Fail to generate %s"%(output_path))
            break


if __name__ == "__main__":

    args = parse_args()
    print(args)
    test_main(args)
