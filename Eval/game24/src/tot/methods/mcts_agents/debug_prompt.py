import json
from tot.prompts.game24_mcts import * 
from tot.models_new import gpt
import pdb
prompt_llama = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant to complete steps for the Game24.<|eot_id|><|start_header_id|>user<|end_header_id|>

{USER_INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''

def debug_prompt():
    x = '3 4 4 13'
    y = '13 + 3 = 16 (left: 4 4 16)\n4 + 4 = 8 (left: 8 16)\n8 + 16 = 24 (left: 24)\n'
    prompt = cot_prompt.format(input=x) + 'Steps:\n' + y + "Answer: "
    #prompt = prompt_llama.format(USER_INPUT=prompt)
    raw_output = gpt(prompt, n=1, stop=None)
    proposals = raw_output[0].split('\n')
    print(prompt)
    #print(proposals)
    print(raw_output[0])
    pdb.set_trace()

def debug_prompt2():
    x = '3 4 4 13'
    #x = '4 5 6 10'
    #y = '4 + 5 = 9 (left: 6 9 10)\n' 
    #y = '4 + 5 = 9 (left: 6 9 10)\n6 + 9 = 15 (left: 10 15)\n' 
    #x = '3 4 4 13'
    #y = '13 + 3 = 16 (left: 4 4 16)\n4 + 4 = 8 (left: 8 16)\n8 + 16 = 24 (left: 24)\n'
    #y = '13 + 3 = 16 (left: 4 4 16)\n4 + 4 = 8 (left: 8 16)\n'
    y = '13 + 3 = 16 (left: 4 4 16)\n'
    #prompt = prompt_llama.format(USER_INPUT=prompt)
    prompt = cot_prompt_rollout_v3.format(input=x) + 'Steps:\n' + y 
    #prompt = cot_prompt_rollout.format(input=x) + 'Steps:\n' + y + "Answer: "
    #prompt = prompt_llama.format(USER_INPUT=prompt)
    raw_output = gpt(prompt, n=1, stop=None, temperature=0.0)
    proposals = raw_output[0].split('\n')
    print(prompt)
    #print(proposals)
    print(raw_output[0])
    pdb.set_trace()


if __name__=="__main__":
   #debug_prompt()
   debug_prompt2() 