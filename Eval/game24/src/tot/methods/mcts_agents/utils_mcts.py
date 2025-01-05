import json
import pdb
import sys
from tot.models import gpt
from tot.prompts.game24_mcts import * 
import os
from PIL import Image
from io import BytesIO
import requests
import time
import random

def encode_image(image, encoding_format="PNG"):
    buffered = BytesIO()
    image.save(buffered, format=encoding_format)
    buffered.seek(0)
    return buffered


def call_reward_models(api, state, job_id):
    return call_api(api, state=state, job_id=job_id)


def call_api(url = 'http://172.30.150.33:15679/api/generate', state=None, num_api=4, job_id=0):
    encoding_format="PNG"
    images = {}
    image_folder = 'src/tot/methods/mcts_agents'
    image_a = 'not_used_%d.png'%job_id
    images['not_used.png'] = Image.open(os.path.join(image_folder, image_a))
    preference_data_list = prepare_chat(state=state)
    preference_data = json.dumps(preference_data_list, indent=2)
    factual_data = os.path.join(image_folder, 'not_used_factual.json')
    prompt = os.path.join(image_folder, 'fact_rlhf_reward_prompt_wj2.txt')
    factual_data = open(factual_data).read()
    prompt = open(prompt).read()
    data = dict(
    preference_data=preference_data,
    factual_data=factual_data,
    prompt=prompt,
    )
    files = {}
    # encoding_format="PNG"
    # for k, v in images.items():
    #     image = encode_image(v, encoding_format=encoding_format)
    #     files[k] = image
    headers = {
        "User-Agent": "BLIP-2 HuggingFace Space",
    }
    try_t, max_try_time = 0, 5
   
    port = url.split(":")[-1].split("/")[0]
    url_list = [ url.replace(port, str(int(port)+idx)) for idx in range(num_api)]  
    #print(url_list)
    while try_t < max_try_time:
        try:
            if try_t == 0:
                tmp_url = url_list[job_id%num_api]
            else:
                #tmp_url = url_list[job_id%num_api]
                tmp_url = random.choice(url_list)
            response = requests.post(tmp_url, data=data, files=files, headers=headers)
            decoded_string = response.content.decode('utf-8')
            score_list = json.loads(decoded_string)
            break
        except:
            print(decoded_string)
            time.sleep(try_t)
            try_t +=1
    #print(state["trajectory"])
    #print(score_list)
    #assert len(score_list)==1
    return score_list[0]


def prepare_chat(state):
    cot_prompt = '''You are a start agent and generate data for Game24. Game24 requires users to use numbers and basic arithmetic operations (+ - * /) to obtain 24. 
    You task is to generate a new input (4 digital number) for Game 24.
    1. each new input number should be in the range of 1 to 13.
    2. People can use numbers and basic arithmetic operations (+ - * /) to obtain 24. At each step, people are only allowed to choose two of the remaining numbers to obtain a new number.
    Here are the few-shot examples.
    3. since there is only four number input and the intermediate steps should only be three.'''
    #input_nums = "4 4 9 3"
    input_nums = state["input"] 
    action = f"{cot_prompt} Input: {input_nums}"
    answer = "Steps: 4 + 4 = 8 (left: 3 8 9)\n9 / 3 = 3 (left: 3 8)\n8 * 3 = 24 (left: 24)\nAnswer: (4 + 4) * (9 / 3)"
    answer_list = state["trajectory"].strip().split("\n")
    #answer_list = [line for line in answer_list if len(line)>0]
    line_num = min(len(answer), 4)
    #remove_part = True
    remove_part = False
    if remove_part:
        answer_list[-1] = answer_list[-1].split("=")[0]
    answer = "Steps:\n"+"\n".join(answer_list[-1*line_num:]).strip()
    #print(answer)
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
            "value": answer
    }

    preference = dict(
        id=0,
        conversations=conversations,
        output_1=[output_1],
        output_2=[output_1],
        preference=1,
        )
            
            
    preference['output_1'].append({
            "from": "human",
            "value": f"Please evaluate whether your last response achieves the goal of Game 24 or not"
        })
    preference['output_1'].append({
        "from": "llava",
        "value": "Following your definitions, the score of my last response is"
    })
            
    preference['output_2'].append({
            "from": "human",
            "value": f"Please evaluate whether your last response achieves the goal of Game 24 or not"
        })
    preference['output_2'].append({
        "from": "llava",
        "value": "Following your definitions, the score of my last response is"
    })
    return [preference]


def parse_action_operation(action_str):
    try:
        operator_str = action_str.split("(left:")[0]
        input_str = operator_str.split("=")[0].strip()
        input_1, operator, input_2 = input_str.split(" ")
    except:
        print(input_str)
        pdb.set_trace()
    return input_1, input_2, operator

def execute_step(input_1, input_2, operator):
    if operator=="+":
        tmp_output = input_1 + input_2
    elif operator=="-":
        tmp_output = input_1 - input_2
    elif operator=="*":
        tmp_output = input_1 * input_2
    return tmp_output


def combinations(space):
    if isinstance(space, spaces.Discrete):
        return range(space.n)
    elif isinstance(space, spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError


def multigpu_breakpoint():
    import torch
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            breakpoint()
        else:
            torch.distributed.barrier()

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def get_proposals(task, model, x, y):
    gpt = model 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    # filter output for llama
    from tot.tasks.game24 import Game24Task
    if isinstance(task, Game24Task):
        proposals = [_ for _ in proposals if len(_)>0 and _[0].isdigit()] 
    return [y + _ + '\n' for _ in proposals]

def set_debugger():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)
    
    
def rollout_partial_sequence(x, y, n_generate_sample, temperature=0.0, max_tokens=None):
    prompt = cot_prompt_rollout_v3.format(input=x) + 'Steps:\n' + y
    if max_tokens is not None:
        raw_output = gpt(prompt, n=n_generate_sample, stop=None, temperature=temperature, max_tokens=max_tokens)
    else:
        raw_output = gpt(prompt, n=n_generate_sample, stop=None, temperature=temperature)
    return raw_output[0]

def rollout_partial_sequence_v2(x, y, n_generate_sample, temperature=0.0, max_tokens=None):
    prompt = cot_prompt_debug_v2.format(input=x) + 'Steps:\n' + y
    if max_tokens is not None:
        raw_output = gpt(prompt, n=n_generate_sample, stop=None, temperature=temperature, max_tokens=max_tokens)
    else:
        raw_output = gpt(prompt, n=n_generate_sample, stop=None, temperature=temperature)
    return raw_output[0]

def rollout_partial_sequence_v3(x, y, n_generate_sample, temperature=0.0, max_tokens=None):
    prompt = cot_prompt_debug_v2.format(input=x) + 'Steps:\n' + y
    if max_tokens is not None:
        raw_output = gpt(prompt, n=n_generate_sample, stop=None, temperature=temperature, max_tokens=max_tokens)
    else:
        raw_output = gpt(prompt, n=n_generate_sample, stop=None, temperature=temperature)
    return raw_output