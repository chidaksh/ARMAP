
prompt_llama = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{USER_INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''

import logging
from openai import OpenAI
import pdb
import openai

from .base import LMAgent
import time
logger = logging.getLogger("agent_frame")

def pre_process(prompt):
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

class OPENLLMLMAgent(LMAgent):
    
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        self.model_id = config['model_name']
        self.temperature = config['temperature']
        self.top_p = config['top_p']
        self.max_new_tokens = config['max_new_tokens']
        self.data_gen = config['data_gen']
        self.url = config["model_url_add"]
        self.api_key = config["api_key"] 

    def __call__(self, prompt, debug=False, is_greed=False, num_seq=1, beam_search=False) -> str:
        client = OpenAI(base_url=self.url, api_key=self.api_key)
        if self.data_gen:
            processed_input = prompt
        else:
            processed_input = pre_process(prompt)
        # yields batch of results that are produced asynchronously and in parallel
        if self.model_id == 'llama-3.1-70b':
            model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
            # processed_input = prompt_llama.format(USER_INPUT=processed_input)
        elif self.model_id == 'llama-3.1-8b':
            # model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            # processed_input = prompt_llama.format(USER_INPUT=processed_input)
        elif self.model_id == 'mixtral-7b':
            model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        elif self.model_id == "phi_mini":
            model_id = "microsoft/Phi-3.5-mini-instruct"
        else:
            model_id = self.model_id
            pdb.set_trace()
        temperature = self.temperature if not is_greed else 0
        temperature = temperature if not beam_search else 0
        
        cur_iter, max_iter = 0, 5
        while cur_iter < max_iter:
            # Prepend the prompt with the system message
            messages = [{
                "role": "user",
                "content": processed_input
            }]
            #try:
            if True:
                if num_seq>1 and beam_search:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        max_tokens=self.max_new_tokens,
                        stop=self.stop_words,
                        top_p=self.top_p,
                        temperature=temperature,
                        n = num_seq, 
                        extra_body={
                            "use_beam_search": True,
                            "best_of": num_seq * 2
                            }
                    )
                    post_list = [post_process(gen.message.content) for gen in response.choices]
                    return post_list
                elif num_seq > 1:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        max_tokens=self.max_new_tokens,
                        stop=self.stop_words,
                        top_p=self.top_p,
                        temperature=temperature,
                        n = num_seq, 
                    )
                    post_list = [post_process(gen.message.content) for gen in response.choices]
                    return post_list
                else: 
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        max_tokens=self.max_new_tokens,
                        stop=self.stop_words,
                        top_p=self.top_p,
                        temperature=temperature,
                    )
                    generation = response.choices[0].message.content
                    if self.data_gen:
                        post_generation = generation
                    else:
                        post_generation = post_process(generation)
                    return post_generation # response
            #except:
            else:
                time.sleep(cur_iter)
                cur_iter +=1
                print("Fail at iteration %d/%d\n"%(cur_iter, max_iter))
                print(generation)
                pdb.set_trace()
        return "Fail at iteration %d/%d\n"%(cur_iter, max_iter)
        