import os
import json
import requests
from io import BytesIO
from PIL import Image
import re

import json
import os
import sys

info = "You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.\n\nThe available actions are:\nopen OBJ: open a container\nclose OBJ: close a container\nactivate OBJ: activate a device\ndeactivate OBJ: deactivate a device\nconnect OBJ to OBJ: connect electrical components\ndisconnect OBJ: disconnect electrical components\nuse OBJ [on OBJ]: use a device/item\nlook around: describe the current room\nexamine OBJ: describe an object in detail\nlook at OBJ: describe a container's contents\nread OBJ: read a note or book\nmove OBJ to OBJ: move an object to a container\npick up OBJ: move an object to the inventory\npour OBJ into OBJ: pour a liquid into a container\nmix OBJ: chemically mix a container\nteleport to LOC: teleport to a specific room\nfocus on OBJ: signal intent on a task object\nwait: task no action for 10 steps\nwait1: task no action for a step\n"


dir = sys.argv[1]
port = sys.argv[2]

mx = 0
test_data = []
for sub_dir in os.listdir(dir):
    if 'sample' not in  sub_dir:
        continue
    cnt = 0
    for file in os.listdir(os.path.join(dir, sub_dir)):
        if '.json' not in file:
            continue
        cnt += 1
        sample = json.load(open(os.path.join(dir, sub_dir, file)))

        dir_id = int(sub_dir.split('_')[-1])
        image_id = int(file.split('_')[0])
        task_id = int(file.split('_')[1].split('.')[0])

        idx = task_id * 10000 + dir_id * 100 + image_id

        score = sample['meta']['reward']
        begin = -1
        s = ''
        for i, conv in enumerate(sample['conversations']):
            s += conv['value']
            if 'Task Description:' in conv['value']:
                if begin != -1:
                    begin = -1
                    break
                begin = i
        if begin == -1:
            continue


        correct_history = sample['conversations'][begin:]

        if len(correct_history) == 1:
            continue

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
            s += conv['value']
        mx = max(mx, total_len)
        if total_len > 10000:
            continue
        if output_1[-1]['from'] != 'human':
            continue

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




        test_data.append(dict(
            id=idx,
            conversations=conversations,
            output_1=output_1,
            output_2=output_1,
            preference=1,
            score=score,
        ))

def encode_image(image, encoding_format="PNG"):
    buffered = BytesIO()
    image.save(buffered, format=encoding_format)
    buffered.seek(0)
    return buffered


encoding_format="PNG"


url = f'http://0.0.0.0:{port}/api/generate'

images = {}

preference_data = json.dumps(test_data)

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
import json
import numpy as np
score_list = json.loads(decoded_string)


rewards_dict = {}

preference_data = json.loads(preference_data)

save_data = []

for sample, reward in zip(preference_data, score_list):
    id = sample['id']
    fix_id = id%100 + id//10000*10000
    if fix_id not in rewards_dict:
        rewards_dict[fix_id] = []
    rewards_dict[fix_id].append((reward, sample['score']))
    sample['reward'] = reward
    save_data.append(sample)

json.dump(save_data, open(os.path.join(dir, 'save_reward.json'), 'w'), indent=2)


avg = np.mean([np.mean([y[1] for y in x]) for x in rewards_dict.values()])
mx = np.mean([np.max([y[1] for y in x]) for x in rewards_dict.values()])
print(f"port: {port} sample avg : {avg}")

print(f"port: {port} sample max : {mx}")


sel = np.mean([sorted(x,key=lambda y:y[0], reverse=True)[0][1] for x in rewards_dict.values()])

print(f"port: {port} sample sel : {sel}")

