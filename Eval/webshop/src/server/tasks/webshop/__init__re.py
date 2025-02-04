from os.path import dirname, realpath
import os

import re
import sys
import time
from typing import Dict, List, Any
import requests

import json

import imgkit

sys.path.append(dirname(realpath(__file__)))

from html2image import Html2Image
from PIL import Image
import io
from io import BytesIO
import base64
from htmlwebshot import WebShot


from src.server.task import Task, Session
from src.typings import SampleStatus, TaskOutput

from .web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from .web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv

prompt: str = """
You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions, \
you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
search[keywords]
click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:

Thought:
I think ...

Action:
click[something]
"""


class WebShop(Task):
    def __init__(self, **configs):
        super().__init__(**configs)
        self.ranging = (configs.pop("start", 0), configs.pop("end", 500))
        self.env = WebAgentTextEnv(observation_mode="html", human_goals=True)
        # gemini_text = GeminiText()
    def get_indices(self) -> List[Any]:
        return list(range(*self.ranging))

    def save_as_html(self, html_content, file_path):
        with open(file_path, "w") as file:
            file.write(html_content)

    def crop(self, image):
        image_ori = image
        image_data = image.getdata()
        bg_color = image_data.getpixel((0, 0))
        left = 0
        top = 0
        while image_data.getpixel((left, top)) == bg_color:
            left += 1
            if left >= image.width:
                left = 0
                top += 1
            if top >= image.height:
                break
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
        imgkit.from_file(file_path, f'/root/workspace/file_cache/image/{image_file_path}', options=options)
        image = Image.open(f'/root/workspace/file_cache/image/{image_file_path}')
        image = self.crop(image)

        image_stream = io.BytesIO()
        image.save(image_stream, format="JPEG")

        image_stream.seek(0)
        imagestr = base64.b64encode(image_stream.read()).decode('utf-8')
        return imagestr
    def convert_history_to_evaluator(self,history,index):
        def encode_image(image, encoding_format="PNG"):
            buffered = BytesIO()
            image.save(buffered, format=encoding_format)
            buffered.seek(0)
            return buffered
        preference_list = []
        factual_list = {}
        intent = history[0]["observation"].split("[SEP]")[2]
        sample = history
        files = {}
        images = {}
        encoding_format = "PNG"
        preference = None

        for sample_index, item in enumerate(sample):
            if 'observation' in item and 'response' in item:
                img = item["imagestr"]
                prediction = item['response']
                img_name = f"image_task_{index}_{sample_index}.png"
                full_image_dir = f"./image_tmp/"
                if not os.path.exists(full_image_dir):
                    os.makedirs(full_image_dir)
                full_image = full_image_dir + img_name
                image_data = base64.b64decode(img)
                image_bytes = BytesIO(image_data)
                image = Image.open(image_bytes)
                image.save(full_image)
                
                images[img_name] = Image.open(full_image)
                factual_list[img_name] = [item['observation']]


                if not os.path.exists(full_image):
                    print("image no exist")
                    break

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
                    preference['output_1'].append({
                        "from": "human",
                        "image": img_name,
                        "value":"Current screenshot: <image>. Observation: <obs>."
                    })
                    preference['output_1'].append(output_1)
        preference['output_1'].append({
            "from": "human",
            "value": "Please evaluate whether your last response achieves the \"Navigation Intent\" or not"
        })
        preference['output_1'].append({
            "from": "llava",
            "value": "Following your definitions, the score of my last response is"
        })
        preference_list.append(preference)
        for k, v in images.items():
            image = encode_image(v, encoding_format=encoding_format)
            files[k] = image
        return preference_list,factual_list,files
    async def start_sample(self, index: int, session: Session) -> TaskOutput:

        meta_data = {
            "action_history": ["None"],
            "memory": []
        }
        max_num_attempts = 10
        for trail_idx in range(max_num_attempts):
            history = []
            env = self.env
            env.reset(index)

            session.inject({"role": "user", "content": prompt})
            session.inject({"role": "agent", "content": "Ok."})

            # one shot

            session.inject({'role': 'user',
                            'content': 'Observation:\n"WebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Search"\n\nAvailable Actions:\n{"has_search_bar": true, "clickables": ["..."]}'})
            session.inject({'role': 'agent',
                            'content': 'Thought:\nI think I should use the search bar to look for the product I need.\n\nAction:\nsearch[l\'eau d\'issey 6.76 fl oz bottle price < 100.00]'})
            session.inject({'role': 'user',
                            'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] B000MJZOPK [SEP] L\'eau d\'Issey by Issey Miyake for Women 3.3 oz Eau de Toilette Spray [SEP] $49.98 [SEP] B0012S249E [SEP] L\'eau D\'issey By Issey Miyake For Women. Shower Cream 6.7-Ounces [SEP] $31.36 [SEP] B01H8PGKZS [SEP] L\'eau D\'Issey FOR MEN by Issey Miyake - 6.7 oz EDT Spray [SEP] $67.97 [SEP] B00G3C8FHE [SEP] L\'Eau d\'Issey pour Homme - Eau de Toilette 4.2 fl oz [SEP] $51.25 [SEP] B000R94HRG [SEP] Issey Miyake L\'Eau D\'Issey Pour Homme Eau De Toilette Natural Spray [SEP] $44.99 [SEP] B000C214CO [SEP] Issey Miyake L\'eau D\'issey Eau de Toilette Spray for Men, 4.2 Fl Oz [SEP] $53.99 [SEP] B0018SBRDC [SEP] Issey Miyake L\'eau d\'Issey for Women EDT, White, 0.84 Fl Oz [SEP] $27.04 [SEP] B000XEAZ9Y [SEP] L\'eau De Issey By Issey Miyake For Men. Eau De Toilette Spray 6.7 Fl Oz [SEP] $67.08 [SEP] B079HZR2RX [SEP] L\'eau d\'Issey Pure by Issey Miyake for Women 3.0 oz Nectar de Parfum Spray [SEP] $71.49"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "..."]}'})
            session.inject({'role': 'agent',
                            'content': 'Thought:\nI think I should click on the product I need, which is B000VOHH8I.\n\nAction:\nclick[B000VOHH8I]'})
            session.inject({'role': 'user',
                            'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "..."]}'})
            session.inject({'role': 'agent',
                            'content': 'Thought:\nI think I should click on the \'6.76 fl oz (pack of 1)\' option to select the size I need.\n\nAction:\nclick[6.76 fl oz (pack of 1)]'})
            session.inject({'role': 'user',
                            'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "..."]}'})
            session.inject({'role': 'agent',
                            'content': 'Thought:\nI think I should click on the \'Buy Now\' button to purchase the product.\n\nAction:\nclick[Buy Now]'})
            if len(meta_data["memory"])!=0:
                session.inject({'role': 'user',
                                'content': 'REFLECTIONS FROM PREVIOUS ATTEMPTS'+"\n".join(meta_data["memory"])})
                session.inject({"role": "agent", "content": "Ok."})
            html_observation = env.observation
            
            observation = env.convert_html_to_text(html_observation, simple=True)
            url = env.state['url']
            imagestr = "ff"
            h2i = Html2Image(output_path='/root/workspace/file_cache/image/',
                             custom_flags=['--no-sandbox', '--disable-gpu'])
            imagestr = self.save_screenshot(h2i, html_observation, f"/root/workspace/file_cache/html/html_{index}_ori.html",
                                            image_file_path=f"image_{index}_ori.jpg")
            
            
            reward = 0
            finish_reason = SampleStatus.COMPLETED

            for j in range(10):
                available_actions = env.get_available_actions()
                session.inject(
                    {
                        "role": "user",
                        "content": f"Observation:\n{observation}\n\n"
                                   f"Available Actions:\n{available_actions}"
                                   f"$ImageStr${imagestr}",
                    }
                )
                response = await session.action()
                if response.status == "AGENT_CONTEXT_LIMIT":
                    finish_reason = SampleStatus.AGENT_CONTEXT_LIMIT
                    break
                response = response.content
                try:
                    action = re.search(
                        r"[Aa]ction: *\n* *((search|click)\[.+?])", response
                    ).group(1)
                except:
                    finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
                    action = None

                # if action:
                history.append(
                    {
                        "imagestr": imagestr,
                        "html_observation": html_observation,
                        "observation": observation,
                        "available_actions": available_actions,
                        "response": response,
                        "action": action,
                    }
                )
                if not action:
                    reward = 0
                    break
                html_observation, reward, done, info = env.step(action)
                
                observation = env.convert_html_to_text(html_observation, simple=True)
                url = env.state['url']
                # print(url)
                imagestr = self.save_screenshot(h2i, html_observation,
                                                f"/root/workspace/file_cache/html/html_{index}_{j}.html",
                                                image_file_path=f"image_{index}_{j}.jpg")
                                                
                history[-1]["reward"] = reward
                history[-1]["done"] = done
                if done:
                    break
            else:
                finish_reason = SampleStatus.TASK_LIMIT_REACHED

            headers = {
                "User-Agent": "BLIP-2 HuggingFace Space",
            }
            
            preference_data,factual_data,files = self.convert_history_to_evaluator(history = history,index = index)
            preference_data = json.dumps(preference_data)
            preference_data = preference_data.encode('utf-8')
            factual_data = json.dumps(factual_data)
            factual_data = factual_data.encode('utf-8')
            data = dict(
                preference_data=preference_data,
                factual_data=factual_data,
                prompt="prompt".encode('utf-8'),
            )
            response = requests.post('http://172.17.0.1:15678/api/generate', data=data, files=files, headers=headers)
            oracle_score = response.content
            print(oracle_score)
            score_source = "gt"
            score = oracle_score
            decoded_string = score.decode('utf-8').strip()
            score_list = json.loads(decoded_string)
            score = score_list[0]
            status = "PASSED" if score >= 0.9 else "FAILED"
            print(f"[Trail {trail_idx}] GT eval: {score} | {status}")
            if score == 1:
                break
            if trail_idx == (max_num_attempts - 1):
                break
            session.inject(
                {
                    "role": "user",
                    "content": "$reflection$"
                }
            )
            reflection = await session.action()
            meta_data["memory"].append(reflection.content)
            session.clear()
        return TaskOutput(
            status=finish_reason,
            result={
                "reward": reward,
                "history": history,
            },
        )

    def calculate_overall(self, results: List[TaskOutput]) -> Dict:
        def factory(key):
            def f(output):
                output = [x for x in output if x]
                if key == "history":
                    return (
                        sum([len(x[key]) for x in output]) / len(output)
                        if len(output) > 0
                        else 0
                    )
                return (
                    sum([x[key] for x in output]) / len(output)
                    if len(output) > 0
                    else 0
                )

            return f

        results = [x.result for x in results if x]

        return {
            "reward": factory("reward")(results),
        }
