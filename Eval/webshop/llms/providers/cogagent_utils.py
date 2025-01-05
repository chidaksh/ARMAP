"""Tools to generate from Cogagent prompts."""

import random
import time
from typing import Any

from google.api_core.exceptions import InvalidArgument
from vertexai.preview.generative_models import (
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
)
from utils.util import config
import os

"""
Use genai to call models for simplicity
"""
args = config()

from prompts.template_cogagent import en_template_task
import random
class CogagentModel:
    def __init__(self) -> None:
        self.template_cog = en_template_task
    def generate_cogagent_prompt(self, task, grounding=True):
        template = self.template_cog[random.randrange(0, len(self.template_cog))]
        prompt = template.replace('<TASK>', f'"{task}"')
        if grounding:
            prompt += '(with grounding)'
        return prompt
    
    def decode_cogagent(self, size, ques, response: str, extra):
        print(response)
        # import pdb;pdb.set_trace()
        try:

            template = "Let's think step-by-step. " + response + \
                " In summary, the next action I will perform is ```<ACTION>```"

            operation = response.rsplit('Grounded Operation:', 1)[1]
            box = operation.rsplit('[[', 1)[1].split(']]',1)[0]
            # x0, y0, x1, y1 = eval(box)
            x0, y0, x1, y1 = [int(num) for num in box.split(',')]
            iw, ih = size
            x0 *= iw / 1000
            y0 *= ih / 1000
            x1 *= iw / 1000
            y1 *= ih / 1000

            template = template.replace(box, f"{x0},{y0},{x1},{y1}")
            size0 = (x1 - x0) * (y1 - y0)
            if not extra:
                assert 0, "no box info!"

            box_id = -1
            max_overlap = 0
            for i in extra:
                box = extra[i]
                xm, ym, w, h = box
                x2 = xm - w / 2
                y2 = ym - h / 2
                x3 = xm + w / 2
                y3 = ym + h / 2
                size1 = w * h

                x4 = max(x0, x2)
                y4 = max(y0, y2)
                x5 = min(x1, x3)
                y5 = min(y1, y3)

                if x4 <= x5 and y4 <= y5:
                    size2 = (x5 - x4) * (y5 - y4)
                else:
                    size2 = 0

                s0 = size2 / size0
                s1 = size2 / size1

                if s0 > 0 and s1 > 0:
                    sc = 1 / (1/s0 + 1/s1)
                    if sc > max_overlap:
                        max_overlap = sc
                        box_id = i

            if 'CLICK' in response:
                return template.replace('<ACTION>', f'click [{box_id}]')
            elif 'TYPE' in response:
                text = response.rsplit('TYPE', 1)[1].split(' ', 1)[1].rsplit(' at', 1)[0]
                return template.replace('<ACTION>', f'type [{box_id}] [{text}]')
            elif 'SELECT' in response: # no select in viswebarena
                return template.replace('<ACTION>', f'click [{box_id}]')
            else:
                assert 0, "Undefined operation!"
        except Exception as e:
            print(e)
            return "Decode ERROR!"

    def call_API(self, prompt, max_length, top_p, temperature, image):
        def encode_image(image, encoding_format="PNG"):
            from io import BytesIO
            buffered = BytesIO()
            image.save(buffered, format=encoding_format)
            buffered.seek(0)
            return buffered

        url = 'http://localhost:5555/api/generate'
        data = {
            "prompts": prompt,
            "max_length": max_length,
            "top_p": top_p,
            "temperature": temperature
        }
        paired_image_text = [(image, "1")]
        files = {}
        for idx, (img, _) in enumerate(paired_image_text):
            # image = encode_image(image, encoding_format=image.format)
            img = encode_image(img)
            files[f"image{idx}"] = img
        headers = {
            "User-Agent": "Cogagent",
        }
        import requests
        response = requests.post(url, data=data, files=files, headers=headers).json()[0]
        return response


    def generate_reward(self, prompt, generation_config):
        # prompt_cog = prompt[0]
        # image = prompt[1]
        
        prompt_cog = ""
        for x in prompt:
            if isinstance(x, str):
                prompt_cog += x + '\n'
            else:
                image = x

        response = self.call_API(
            prompt_cog,
            generation_config["max_output_tokens"],
            generation_config["top_p"],
            generation_config["temperature"],
            image,
        )
        print(response)
        return response

    def generate_agent(self, prompt, generation_config):
        # import pdb; pdb.set_trace()

        extra = generation_config['extra']
        '''
        prompt_cog = ""
        example_cnt = 0
        for x in prompt:
            if isinstance(x, str):
                if x.split('\n')[0].strip() == "Observation":
                    example_cnt += 1
                if x.strip() == "Now make prediction given the observation":
                    # prompt_cog += "[split]"
                    example_cnt = 0
                if example_cnt <= 1:
                    prompt_cog += x + '\n'
            else:
                image = x
        items = prompt_cog.rsplit(':OBSERVATION:', 1)[1].rsplit('URL:', 1)[0]
        prompt_cog = prompt_cog.replace(items, items[0:1000])
        # prompt_cog.replace("http://ec2-3-13-232-171.us-east-2.compute.amazonaws.com:9980", "CLASSIFIEDS")
        # prompt_cog.replace("http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770", "SHOPPING")
        # prompt_cog.replace("http://ec2-3-13-232-171.us-east-2.compute.amazonaws.com:9999", "REDDIT")
        # prompt_cog.replace("http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888", "WIKIPEDIA")
        '''

        prompt_ques = ""
        begin = False
        for x in prompt:
            if isinstance(x, str):
                if begin:
                    prompt_ques += x + '\n'
                if x.strip() == "Now make prediction given the observation":
                    begin = True
            else:
                image = x
        size = image.size
        # import pdb; pdb.set_trace()
        '''
        items = prompt_ques.rsplit(':OBSERVATION:', 1)[1].rsplit('URL:', 1)[0]
        if len(items) > 2500:
            print("!!!!!! THE PROMPT IS TOO LONG !!!!!!")
            prompt_ques = prompt_ques.replace(items, items[0:2500])

        prompt_cog = open('prompts/prompt_cogagent.py').read()
        prompt_cog = prompt_cog.replace('__INSERT_QUESTION_HERE__', prompt_ques)
        # prompt_cog = prompt_cog.strip() + " Let's think step-by-step. "
        import pdb; pdb.set_trace()
        '''

        objective = prompt_ques.split('OBJECTIVE: ')[1].split('\n')[0]

        prompt_cog = self.generate_cogagent_prompt(objective)

        response = self.call_API(
            prompt_cog,
            generation_config["max_output_tokens"],
            generation_config["top_p"],
            generation_config["temperature"],
            image,
        )
        response = self.decode_cogagent(size, prompt_ques, response, extra)
        print(response)
        return response


    def generate_content(self, prompt, generation_config):
        class Response:
            def __init__(self, text) -> None:
                self.text = text
        try:
            if generation_config['reward']:
                response = self.generate_reward(prompt, generation_config)
            else:
                response = self.generate_agent(prompt, generation_config)
            response = Response(response)
            import pdb;pdb.set_trace()
            return response
        except Exception as e:
            print(e)

model = CogagentModel()

def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 1,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (InvalidArgument,),
):
    #max_retries: int = 10,
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:

                print(e)

                # Increment retries
                num_retries += 1
                print("Number retries: %d/%d\n"%(num_retries, max_retries))

                # Check if max retries has been reached
                if num_retries > max_retries:
                    print("Unexpected errors from APIs, type 1\n")
                    print("Number retries: %d/%d\n"%(num_retries, max_retries))
                    if kwargs['candidate_count']==1:
                        return ""
                    return ["" for idx in range(kwargs['candidate_count'])]
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:

                print(e)
                # Increment retries
                num_retries += 1
                print("Unexpected errors from APIs, type 2\n")
                print("Number retries: %d/%d\n"%(num_retries, max_retries))
                import pdb
                pdb.set_trace()
                # Check if max retries has been reached
                if num_retries > max_retries:
                    if kwargs['candidate_count']==1:
                        return ""
                    return ["" for idx in range(kwargs['candidate_count'])]
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                # Sleep for the delay
                time.sleep(delay)
    return wrapper


@retry_with_exponential_backoff
def generate_from_cogagent_completion(
    prompt: list[str | Image],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    candidate_count = 1,
    extra = None,
    reward = False,
) -> str:
    del engine
    safety_config = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    #import pdb
    #pdb.set_trace()

    # sample multiple paths candidate_count times since gemini only supports 1 sequence a time
    if candidate_count==1:
        response = model.generate_content(
            prompt,
            generation_config=dict(
                candidate_count=candidate_count,
                max_output_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                extra=extra,
                reward=reward,
            ),
            #safety_settings=safety_config,
        )
        try:
            answer = response.text
        except:
            answer = ""
        return answer
    else:
        answer_list = []
        for smp_idx in range(candidate_count):
            response = model.generate_content(
                prompt,
                generation_config=dict(
                    candidate_count=1,
                    max_output_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    extra=extra,
                    reward=reward,
                ),
            )
            try:
                answer = response.text
            except:
                answer = ""
            answer_list.append(answer)
        #import pdb;pdb.set_trace()
        # print(answer_list)
        return answer_list


@retry_with_exponential_backoff
# debug only
def fake_generate_from_gemini_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer
