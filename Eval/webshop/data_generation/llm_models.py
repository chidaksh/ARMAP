import argparse
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR= os.path.dirname(SCRIPT_DIR)
sys.path.append(LIB_DIR)
from utils import goal_prompt_attributes, goal_prompt_attributes_v2, product_prompt_attributes
#comment this if using step 1 for data generation
#from llms import call_llm, lm_config
import pdb
import random

def map_attributes(attirbute, example_info):
    if attirbute=="attributes":
        return example_info["Attributes"]
    elif attirbute=="instruction_attributes":
        if isinstance(example_info["Attributes"], list):
            smp_num = random.randint(1,2)
            smp_num = min(len(example_info["Attributes"]), smp_num)
            ins_attr_list = random.sample(example_info["Attributes"], smp_num)
            return ins_attr_list
        else:
            import pdb
            pdb.set_trace()
    elif attirbute=="instruction_options":
        options = example_info["options"]
        if isinstance(options, dict):
            option_values = list(options.values())
            smp_num = random.randint(0,2)
            smp_num = min(len(option_values), smp_num)
            instruction_options = random.sample(option_values, smp_num)
            return instruction_options
    elif attirbute=="options":
        options = example_info["options"]
        options = ["%s: %s"%(key, val) for key, val in options.items()]
        return options
    else:
        raise NotImplemented 

class llm_prompt_model():
    def __init__(self, args):
        self.llm_config = lm_config.construct_llm_config(args)
        self.args = args
        self.prompt_ori = open(args.stage2_prompt_path).read().strip()
        print(self.prompt_ori)
     
    def generate_synthetic_goal(self, example_info):
        prompt, missing_dict = self.construct_prompt(example_info)
        response = call_llm(self.llm_config, prompt)
        return response, missing_dict
        
    def construct_prompt(self, example_info):
        prompt_str = self.prompt_ori
        missing_dict = {}
        product_str = ""
        for attr in product_prompt_attributes:
            tmp_str = "%s: %s\n"%(attr, str(example_info[attr]))
            product_str +=tmp_str 
        for attr in goal_prompt_attributes_v2:
            if attr in example_info and "option" not in attr:
                tmp_str = "%s: %s\n"%(attr, str(example_info[attr]))
                product_str +=tmp_str
            else:
                tmp_attr = map_attributes(attr, example_info)
                tmp_str = "%s: %s\n"%(attr, str(tmp_attr))
                product_str +=tmp_str
                missing_dict[attr] = tmp_attr 
        prompt_str = self.prompt_ori.replace("__PRODUCT_INFO__", product_str)
        print(product_str)
        print(missing_dict)
        return prompt_str, missing_dict
