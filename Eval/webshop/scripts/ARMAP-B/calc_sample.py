import json
import os
import numpy as np
import sys
from io import BytesIO
import base64
from PIL import Image
import io
import requests

out_dir = sys.argv[1]
task = sys.argv[2]
port = sys.argv[3]
reward_host = f'http://0.0.0.0:{port}/api/generate'


def encode_image(image, encoding_format="PNG"):
    buffered = BytesIO()
    image.save(buffered, format=encoding_format)
    buffered.seek(0)
    return buffered

def sel(samples):

    images = []
    preference = []
    factual = {}

    def sep(observation):
        observation = observation.split('Instruction: [SEP] ')[1]
        return observation.split('[SEP]', 1)

    for sample in samples:
        pre = {}
        pre['image'] = f"{len(images)}.png"
        inst, obs = sep(sample['history'][0]['observation'])
        factual[f"{len(images)}.png"] = [
            obs
        ]
        images.append(sample['history'][0]['imagestr'])
        

        pre['conversations'] = [
            {
                "from": "human",
                "value": f"Current screenshot: <image>. Observation: <obs>. Navigation Intent: {inst}"
            },
            {
                "from": "gpt",
                "value": "###test gpt###"
            }
        ]

        inst, obs = sep(sample['history'][-1]['observation'])
        factual[f"{len(images)}.png"] = [
            obs
        ]

        pre['output_1'] = [
            {
                "from": "llava",
                "value": sample['history'][0]['response']
            },
            {
                "from": "human",
                "image": f"{len(images)}.png",
                "value": "Current screenshot: <image>. Observation: <obs>."
            },
            {
                "from": "llava",
                "value": sample['history'][-1]['response']
            },
            {
                "from": "human",
                "value": "Please evaluate whether your last response achieves the \"Navigation Intent\" or not"
            },
            {
                "from": "llava",
                "value": "Following your definitions, the score of my last response is"
            }
        ]

        images.append(sample['history'][-1]['imagestr'])
        preference.append(pre)


    prompt = '''USER: Please evaluate whether your last response achieves the "Navigation Intent" or not.\n\nHere's the information you'll have:\nThe current web page screenshot: This is a screenshot of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.\nThe observation, which lists the IDs of all interactable elements on the current web page with their text content if any, in the format [id] [tagType] [text content]. tagType is the type of the element, such as button, link, or textbox. text content is the text content of the element.\n{factual_prompt}{factual_prompt1}\n\nASSISTANT: Following your definitions, the score of my last response is'''


    preference_data = json.dumps(preference)
    factual_data = json.dumps(factual)
    

    data = dict(
        preference_data=preference_data,
        factual_data=factual_data,
        prompt=prompt,
    )
    files = {}


    encoding_format="PNG"
    for k, v in enumerate(images):
        image = encode_image(Image.open(io.BytesIO(base64.b64decode(v))), encoding_format=encoding_format)
        files[f'{k}.png'] = image



    headers = {
        "User-Agent": "BLIP-2 HuggingFace Space",
    }
    response = requests.post(reward_host, data=data, files=files, headers=headers)
    # print(response.content)
    decoded_string = response.content.decode('utf-8')
    predict_rewards = json.loads(decoded_string)

    json.dump(predict_rewards, open(os.path.join(out_dir, f'rewards_{task}_{samples[0]["index"]}.json'), 'w'))

    return samples[predict_rewards.index(max(predict_rewards))]

outputs = {}

for sub_dir in sorted(os.listdir(out_dir)):
    dir = os.path.join(out_dir, sub_dir, task)
    if not os.path.isdir(dir):
        continue
    if 'sample' not in dir:
        continue
    if not os.path.exists(os.path.join(dir, 'runs.jsonl')):
        continue

    total = 0

    lines = open(os.path.join(dir, 'runs.jsonl')).readlines()

    for line in lines:
        try:
            sample = json.loads(line)
            index = sample['index']
            # print(sample['output']['result']['reward'])
            # import pdb; pdb.set_trace()

            total += 1

            if index not in outputs:
                outputs[index] = []

            sample['output']['result']['index'] = index
            outputs[index].append(sample['output']['result'])
        except Exception as e:
            print('json fail!')
            continue
    print(sub_dir, f"total = {total}")


print(f"sample avg : {np.mean([np.mean([y['reward'] for y in x]) for x in outputs.values()])}")

print(f"sample max : {np.mean([np.max([y['reward'] for y in x]) for x in outputs.values()])}")

print(f"sample sel : {np.mean([sel(x)['reward'] for x in outputs.values()])}")

