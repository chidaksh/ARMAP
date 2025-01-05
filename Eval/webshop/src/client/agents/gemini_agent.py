# import anthropic
import os
from copy import deepcopy

from ..agent import AgentClient
from typing import Any, Tuple
import random
import time
from typing import Any,List,Union
import PIL
import io
import base64

from google.api_core.exceptions import InvalidArgument
from vertexai.preview.generative_models import (
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
)
# from utils.util import config

import  google.generativeai as genai

class Gemini(AgentClient):
    def __init__(self, api_args=None, *args, **config):
        super().__init__(*args, **config)
        if not api_args:
            api_args = {}
        self.HUMAN_PROMPT = "\n\nHuman:"
        self.AI_PROMPT = "\n\nAssistant:"
        api_args = deepcopy(api_args)
        genai.configure(api_key=self.key)
        
        api_args["model"] = api_args.pop("name", None)
        self.model = genai.GenerativeModel("gemini-pro-vision")
        if not self.key:
            raise ValueError("Gamini API KEY is required, please assign api_args.key or set OPENAI_API_KEY "
                             "environment variable.")
        self.api_args = api_args
        # if not self.api_args.get("stop_sequences"):
        #     self.api_args["stop_sequences"] = [anthropic.HUMAN_PROMPT]
    def retry_with_exponential_backoff(  # type: ignore
        func,
        initial_delay: float = 1,
        exponential_base: float = 1,
        jitter: bool = True,
        max_retries: int = 10,
        errors: Tuple[Any] = (InvalidArgument,),
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
                    #import pdb
                    #pdb.set_trace()
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
    def generate_from_gemini_completion(
        self,
        prompt: list,
        temperature: float,
        max_tokens: int,
        top_p: float,
        candidate_count = 1,
        top_k = None
    ) -> str:
        
        # import pdb
        # pdb.set_trace()
        
        safety_config = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

        # sample multiple paths candidate_count times since gemini only supports 1 sequence a time
        if top_k is None:
            generation_config=dict(
                candidate_count=candidate_count,
                max_output_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature)
        else:
            generation_config=dict(
                candidate_count=candidate_count,
                max_output_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature)
        if candidate_count==1:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            try:
                answer = response.text
            except:
                answer = ""
            return answer
        else:
            answer_list = []
            for smp_idx in range(candidate_count):
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                try:
                    answer = response.text
                except:
                    answer = ""
                answer_list.append(answer)
            #import pdb;pdb.set_trace()
            # print(answer_list)
            return answer_list

    def inference(self, history: List[dict]) -> str:
        prompt = ""
        image_input = ""
        # prompt = []
        # print(history)
        history_ori = history
        if "reflection" in history[-1]["content"]:
            with open(
                    'agents/reflection_few_shot_examples.txt',
                    'r', encoding='utf-8') as file:
                file_content = file.read()

            prompt = file_content
            history = history_ori[:-1]

        for message in history:
            if message["role"] == "user":
                # if "image" in message.keys() and message["image"]!="none":
                index = message["content"].find("$ImageStr$")
                if index != -1:
                    imagestr = message["content"][index + len("$ImageStr$"):]
                    image_bytes = base64.b64decode(imagestr)
                    image_input = PIL.Image.open(io.BytesIO(image_bytes))
                    # image_input = image_input.resize((40,40))
                #     # print(message["image"])
                prompt += self.HUMAN_PROMPT + message["content"][:index]
                # prompt.append(self.HUMAN_PROMPT + message["content"])
            else:
                prompt += self.AI_PROMPT + message["content"]
                    # prompt.append(self.AI_PROMPT + message["content"])
        if "reflection" in history_ori[-1]["content"]:
            prompt += "\n\nSTATUS: FAIL\n\nNext plan: "
        else:
            prompt += self.AI_PROMPT
        # prompt.append(self.AI_PROMPT)
        # c = anthropic.Client(api_key=self.key)
        # print("---------------------------------------------------------------------")
        resp = self.generate_from_gemini_completion(prompt=[prompt,image_input],temperature=1,max_tokens=15360,top_p=1,candidate_count = 1,top_k = 40)
        # print("**************************************************************************")
        # resp = c.completions.create(prompt=prompt, **self.api_args)
        return str(resp)


class GeminiText(AgentClient):
    def __init__(self, api_args=None, *args, **config):
        super().__init__(*args, **config)
        if not api_args:
            api_args = {}
        self.HUMAN_PROMPT = "\n\nHuman:"
        self.AI_PROMPT = "\n\nAssistant:"
        api_args = deepcopy(api_args)
        genai.configure(api_key=self.key)

        api_args["model"] = api_args.pop("name", None)
        self.model = genai.GenerativeModel("gemini-pro")
        if not self.key:
            raise ValueError("Gamini API KEY is required, please assign api_args.key or set OPENAI_API_KEY "
                             "environment variable.")
        self.api_args = api_args
        # if not self.api_args.get("stop_sequences"):
        #     self.api_args["stop_sequences"] = [anthropic.HUMAN_PROMPT]

    def retry_with_exponential_backoff(  # type: ignore
            func,
            initial_delay: float = 1,
            exponential_base: float = 1,
            jitter: bool = True,
            max_retries: int = 10,
            errors: Tuple[Any] = (InvalidArgument,),
    ):
        # max_retries: int = 10,
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
                    print("Number retries: %d/%d\n" % (num_retries, max_retries))

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        print("Unexpected errors from APIs, type 1\n")
                        print("Number retries: %d/%d\n" % (num_retries, max_retries))
                        if kwargs['candidate_count'] == 1:
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
                    print("Number retries: %d/%d\n" % (num_retries, max_retries))
                    # import pdb
                    # pdb.set_trace()
                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        if kwargs['candidate_count'] == 1:
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
    def generate_from_gemini_completion(
            self,
            prompt: list,
            temperature: float,
            max_tokens: int,
            top_p: float,
            candidate_count=1,
            top_k=None
    ) -> str:

        # import pdb
        # pdb.set_trace()

        safety_config = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

        # sample multiple paths candidate_count times since gemini only supports 1 sequence a time
        if top_k is None:
            generation_config = dict(
                candidate_count=candidate_count,
                max_output_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature)
        else:
            generation_config = dict(
                candidate_count=candidate_count,
                max_output_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature)
        if candidate_count == 1:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            try:
                answer = response.text
            except:
                answer = ""
            return answer
        else:
            answer_list = []
            for smp_idx in range(candidate_count):
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                try:
                    answer = response.text
                except:
                    answer = ""
                answer_list.append(answer)
            # import pdb;pdb.set_trace()
            # print(answer_list)
            return answer_list

    def generate_reflection(self, history: List[dict]) -> str:
        #need to fill in the prompt
        with open('agents/reflection_few_shot_examples.txt', 'r', encoding='utf-8') as file:
            file_content = file.read()

        prompt = file_content
        # image_input = ""
        # prompt = []
        # print(history)
        intent = history[0]["observation"].split("[SEP]")[2]
        for index, message in enumerate(history):
            obs = message['observation']
            if index ==0:
                prompt += self.HUMAN_PROMPT + f"Navigation Intent: {intent}. Observation: {obs}."
            else:
                prompt += self.HUMAN_PROMPT + f"Observation: {obs}."
            response = message['response']
            prompt += self.AI_PROMPT + response
                # prompt.append(self.AI_PROMPT + message["content"])
        prompt += "\n\nSTATUS: FAIL\n\nNext plan: "
        # prompt.append(self.AI_PROMPT)
        # c = anthropic.Client(api_key=self.key)
        # print("---------------------------------------------------------------------")
        resp = self.generate_from_gemini_completion(prompt=[prompt], temperature=1, max_tokens=15360,
                                                    top_p=1, candidate_count=1, top_k=40)
        # print("**************************************************************************")
        # resp = c.completions.create(prompt=prompt, **self.api_args)
        return str(resp)
"""Tools to generate from Gemini prompts."""



"""
Use genai to call models for simplicity
"""
