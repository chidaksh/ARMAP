
prompt_llama = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{USER_INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''


from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

import logging
import backoff

from typing import Optional
from .base import LMAgent

from openai import OpenAI

logger = logging.getLogger("agent_frame")


def pre_process(prompt):
    import pdb; pdb.set_trace()
    processed_input = ''
    for sent in prompt:
        # processed_input += sent['role']
        if sent['content'] == 'OK': continue
        if 'Task Description:' in sent['content']:
            processed_input += '\n'
            processed_input += 'Here is the task you need to do, please only generate the thought and action without any other words:'
            processed_input += '\n'
        processed_input += sent['content']
        processed_input += '\n'
    return processed_input

def post_process(generation):
    generation = generation.strip()
    generation_sent = generation.split('\n')
    post_generation = ''
    for sent in generation_sent:
        post_generation += sent + '\n'
        if 'Action:' in sent:
            break
    return post_generation

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def count_token(prompt):
    res = 0
    for msg in prompt:
        res += len(enc.encode(msg['content']))
    return res

import copy
def add_history(prompt, memory):
    prompt = copy.deepcopy(prompt)
    base = count_token(prompt)
    memory_token = [count_token(x) for x in memory]
    for i in range(len(memory)):
        if base + sum(memory_token[i:]) > 6000:
            continue
        p_a = prompt[:16]
        p_c = prompt[16:]
        p_c[-1]['content'] += '\nPlease output an action different from "Failed history"'
        p_b = [copy.deepcopy(prompt[16])]
        p_b[0]['content'] = '\nYou will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task.\n' + p_b[0]['content'].split('Please only output one Thought and one Action.')[0]
        p_b += [{'role': 'assistant', 'content': 'ok'}]
        for mem in memory[i:]:
            mem = copy.deepcopy(mem)
            mem[0]['content'] = 'Failed history:\n'
            p_b += mem
            p_b += [{'role': 'assistant', 'content': 'Stopped'}]
        prompt = p_a + p_b + p_c
        # import pdb; pdb.set_trace()
        return prompt
    return prompt


def add_history2(prompt, memory):
    prompt = copy.deepcopy(prompt)
    base = count_token(prompt)
    memory_token = [count_token(x) for x in memory]
    for i in range(len(memory)):
        if base + sum(memory_token[i:]) > 6000:
            continue
        p_a = prompt[:16]
        p_c = prompt[16:]
        p_c[-1]['content'] += '\nPlease output an action different from "Failed history"'
        p_b = '\nYou will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task.\n' + p_c[0]['content'].split('Please only output one Thought and one Action.')[0]
        for mem in memory[i:]:
            mem = copy.deepcopy(mem)
            p_b += '\nFailed history:'
            for msg in mem[1:]:
                p_b +=  '\n' + msg['content']
        p_c[0]['content'] = p_b + '\n' +  p_c[0]['content']
        prompt = p_a + p_c
        # import pdb; pdb.set_trace()
        return prompt
    return prompt


reflection_prompt = open('agents/reflection_few_shot_examples.txt').read()

def add_reflection(prompt):
    res = reflection_prompt
    for msg in prompt[16:]:
        res += '\n' + msg['content']
        if 'Please only output one Thought and one Action.' in res:
            res = res.split('Please only output one Thought and one Action.')[0]
    # import pdb; pdb.set_trace()

    return [{'role': 'user', 'content': res}]

import random
import os
class LocalAgent(LMAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        self.model_id = config['model_name']
        self.temperature = config['temperature']
        self.top_p = config['top_p']
        self.top_k = config['top_k']
        self.max_new_tokens = config['max_new_tokens']
        self.min_new_tokens = config['min_new_tokens']
        self.data_gen = config['data_gen']


        model_id = self.model_id
        print(f"model_id : {model_id}")
        self.model_id = model_id
        port = 17777
        if self.model_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
            port = 17777
        elif self.model_id == 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4':
            port = 17781

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
            messages=prompt,
            max_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
        response = chat_completion.choices[0].message.content
        return response

    def __call__(self, prompt, debug=False, is_re=False) -> str:

        if is_re:
            prompt = add_reflection(prompt)

        # import pdb; pdb.set_trace()


        for t in range(5):
            try:
                resp = self.generate_from_local_completion(prompt=prompt, temperature=self.temperature,max_new_tokens=150,top_p=1)
                print(resp)
                break
            except Exception as e:
                print(e)
                if 'maximum' in e.message:
                    return 'Action: Stop'
                import pdb;pdb.set_trace()
        return str(resp)