import os
import openai
import backoff 
import pdb
completion_tokens = prompt_tokens = 0
"""
api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)
"""

def completions_with_backoff_debug(**kwargs):
    from openai import OpenAI
    client = OpenAI(
        #base_url="http://9.33.169.168:7777/v1",
        base_url="http://9.33.169.168:7778/v1",
        api_key="token-abc123",
    )
    MD_ID="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    #MD_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
    #import pdb
    #pdb.set_trace()
    #MD_ID="NousResearch/Meta-Llama-4-8B-Instruct"
    kwargs["model"] = MD_ID
    #kwargs["stop"] = "\n"
    output =  client.chat.completions.create(**kwargs)
    print(output)
    return output

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-5", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff_debug(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        #outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        #completion_tokens += res["usage"]["completion_tokens"]
        completion_tokens += res.usage.completion_tokens
        #prompt_tokens += res["usage"]["prompt_tokens"]
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
