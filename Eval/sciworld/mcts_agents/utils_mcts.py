import itertools
from gym import spaces
import json
import pdb
import os
import requests
from io import BytesIO
from PIL import Image
import re
import sys
sys.path.append('..')
from eval_agent.utils.datatypes import State
import pdb
from openai import OpenAI


def filter_done_ids(done_task_id, output_path, minimal_sample_num):
    filter_id_list = []
    for task_id in done_task_id:
        full_output_path = os.path.join(output_path, task_id+".json")
        try:
            with open(full_output_path) as fh:
                fdict = json.load(fh)
            if len(fdict["rewards"]) >= minimal_sample_num:
                filter_id_list.append(task_id)
        except:
            print("Fail to load %s\n"%(full_output_path)) 
            filter_id_list.append(task_id)
    return filter_id_list


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


def save_trajectory_history(track_list, reward_list, task, output_path):
    out_dict = {"tracks": [], "rewards": []}
    for state in track_list:
        out_dict["tracks"].append(state.to_dict())
    out_dict["rewards"] = reward_list
    json.dump(out_dict, open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)
    
    
def call_reward_models(url, state):
    response = call_api(url, state=state)
    return response


def encode_image(image, encoding_format="PNG"):
    buffered = BytesIO()
    image.save(buffered, format=encoding_format)
    buffered.seek(0)
    return buffered



def call_api(url = 'http://172.30.150.31:15678/api/generate', state=None ):
    encoding_format="PNG"
    images = {}
    
    if state is None:
        preference_data = './ARMAP/Eval/sciworld/prompt/batch_server_fix_sciworld.json'
        preference_data = open(preference_data).read()
    else:
        preference_data_list = prepare_chat(state=state)
        preference_data = json.dumps(preference_data_list, indent=2)
    factual_data = '''{
    }
    '''
    prompt = '''USER: Observation: {observation0}. Navigation Intent: {Intent}

    ASSISTANT: {response0}

    USER: Observation: {observation1}.

    ASSISTANT: {response1}

    USER: Please evaluate whether your last response achieves the "Navigation Intent" or not.

    ASSISTANT: Following your definitions, the score of my last response is
    '''
    data = dict(
    preference_data=preference_data,
    factual_data=factual_data,
    prompt=prompt,
    )
    files = {}
    encoding_format="PNG"
    for k, v in images.items():
        image = encode_image(v, encoding_format=encoding_format)
        files[k] = image
    headers = {
        "User-Agent": "BLIP-2 HuggingFace Space",
    }
    response = requests.post(url, data=data, files=files, headers=headers)
    decoded_string = response.content.decode('utf-8')
    try:
        score_list = json.loads(decoded_string)
    except:
        print(decoded_string)
    return score_list[0]

def prepare_chat(state):
    info = "You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.\n\nThe available actions are:\nopen OBJ: open a container\nclose OBJ: close a container\nactivate OBJ: activate a device\ndeactivate OBJ: deactivate a device\nconnect OBJ to OBJ: connect electrical components\ndisconnect OBJ: disconnect electrical components\nuse OBJ [on OBJ]: use a device/item\nlook around: describe the current room\nexamine OBJ: describe an object in detail\nlook at OBJ: describe a container's contents\nread OBJ: read a note or book\nmove OBJ to OBJ: move an object to a container\npick up OBJ: move an object to the inventory\npour OBJ into OBJ: pour a liquid into a container\nmix OBJ: chemically mix a container\nteleport to LOC: teleport to a specific room\nfocus on OBJ: signal intent on a task object\nwait: task no action for 10 steps\nwait1: task no action for a step\n"
    sample_conversations =[]
    mx = 0
    test_data = []
    for step in state.history:
        if "role" in step:
            from_id = "gpt" if step["role"]=="assistant" else "human"
            step_out = {"from": from_id, "value": step["content"]}
        else:
            step_out = step
        sample_conversations.append(step_out) 
    begin = -1
    for i, conv in enumerate(sample_conversations):
        if 'Task Description:' in conv['value']:
            begin = i
            break
    if begin == -1:
        pdb.set_trace()

    correct_history = sample_conversations[begin:]
    if len(correct_history) == 1:
        pdb.set_trace()

    action = correct_history[0]['value']
    conversations = []
    conversations.append({
        "from": "human",
        "value": info + action
    })
    conversations.append({
        "from": "gpt",
        "value": "###test gpt###"
    })

    total_len = 0
    output_1 = []
    for i, conv in enumerate(correct_history):
        if i == 0:
            continue
        conv['from'] = 'llava' if i % 2 == 1 else 'human'
        output_1.append(conv)
        total_len += len(conv['value'])
    mx = max(mx, total_len)
    if total_len > 10000:
        print('gg')
        print("max length: %d"%(total_len))
        #pdb.set_trace()
    if output_1[-1]['from'] != 'human':
        print(output_1)
        print('gg')
        pdb.set_trace()

    output_1.append({
        'from': 'llava',
        'value': 'Stop'
    })

    output_1.append({
        'from': 'human',
        'value': 'Please evaluate whether you complete the "Task Description" or not.'
    })
    output_1.append({
        'from': 'llava',
        'value': 'Following your definitions, my task completion score is'
    })

    for idx, step in enumerate(output_1):
        # to be consistent with the processing
        if step["value"].endswith("\n"):
            output_1[idx]["value"] = step["value"][:-1]

    test_data.append(dict(
        id=-1,
        image='not_used.png',
        conversations=conversations,
        output_1=output_1,
        output_2=output_1,
        preference=1,
        score=-1,
    ))
    return test_data
