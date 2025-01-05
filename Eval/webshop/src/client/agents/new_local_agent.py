prompt_llama = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{USER_INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''


import anthropic
import os
from copy import deepcopy
from ..agent import AgentClient
import time
from typing import Any,List,Union
import PIL
import io
import base64

import asyncio
import logging
import os
import random
import time
from typing import Any
from typing import Optional
import re
import aiolimiter
from tqdm.asyncio import tqdm_asyncio
from openai import OpenAI

"""
load_dotenv()
client = Client(credentials=Credentials.from_env())
for model in client.model.list(limit=100).results:
    print(model.model_dump(include=["name", "id"]))
import pdb
pdb.set_trace()
"""

class NewLocal(AgentClient):
    def __init__(self, api_args=None, *args, **config):
        super().__init__(*args, **config)
        if not api_args:
            api_args = {}
        self.HUMAN_PROMPT = "\n\nHuman: "
        self.AI_PROMPT = "\n\nAssistant: "
        api_args = deepcopy(api_args)
    
        api_args["model"] = api_args.pop("name", None)
        self.api_args = api_args
        self.model_id = config["name"]
        self.temperature = config['temperature']

        model_id = self.model_id
        print(f"model_id : {model_id}")
        self.model_id = model_id
        port = 17777
        if self.model_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
            port = 17777
        elif self.model_id == 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4':
            port = 17778
        elif self.model_id == 'mistralai/Mistral-7B-Instruct-v0.3':
            port = 17779
        elif self.model_id == 'microsoft/Phi-3.5-mini-instruct':
            port = 17780
        self.client = OpenAI(base_url=f'http://localhost:{port}/v1', api_key='token-abc123')

    
    def generate_from_local_completion(self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        stop_token: Optional[str] = None
    ) -> str:

        chat_completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
        response = chat_completion.choices[0].message.content
        return response

    def inference(self, history: List[dict]) -> str:
        prompt = ""
        image_input = ""
        for message in history:
            if message["role"] == "user":
                
                index = message["content"].find("$ImageStr$")
                if index != -1:
                    imagestr = message["content"][index + len("$ImageStr$"):]
                    image_bytes = base64.b64decode(imagestr)
                    image_input = PIL.Image.open(io.BytesIO(image_bytes))
                prompt += self.HUMAN_PROMPT + message["content"][:index]
                
            else:
                prompt += self.AI_PROMPT + message["content"]
                
        prompt += self.AI_PROMPT

        for t in range(5):
            try:
                resp = self.generate_from_local_completion(prompt=prompt, temperature=self.temperature,max_new_tokens=150,top_p=1)
                print(resp)
                break
            except Exception as e:
                print(e)
        return str(resp)
