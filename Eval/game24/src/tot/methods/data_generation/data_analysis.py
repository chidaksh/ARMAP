import json
import pdb
import re
import os
        
def check_exist(smp, output_list):
    exist_flag = False 
    exp = smp["exp"]
    numbers_exp = re.findall(r'\d+', exp)
    numbers_exp = sorted(numbers_exp)
    num_exp_str = [str(ele) for ele in numbers_exp]
    num_str = "_".join(num_exp_str).strip()
    if num_str in output_list:
        exist_flag = True
    return exist_flag

def check_unique_input():
    #result_fn = "output/data_generation/stage2_v3_100k/samples.json"
    result_fn = "output/data_generation/stage2_v3_2_100k/samples.json"
    with open(result_fn, "r") as fh:
        res_list = json.load(fh)
    cnt = 0
    exp_list = []
    for smp_dict in res_list:
        exp = smp_dict["exp"]
        numbers_exp = re.findall(r'\d+', exp)
        #print(numbers_exp)
        numbers_exp =  sorted(numbers_exp)
        """
        if not numbers_exp in exp_list:
            exp_list.append(numbers_exp)
        if numbers_exp == exp_list[0]:
            print(smp_dict["raw_data"])
            print(exp)
        """
        if not exp in exp_list:
            exp_list.append(exp)
    valid_num = len(exp_list)
    total = len(res_list)
    print("Valid: %d/%d\n"%(valid_num, total))
    pdb.set_trace()

def check_negative_list():
    neg_dir = "output/data_generation/stage3_v3_100k" 
    fn_list = os.listdir(neg_dir)
    fn_list = sorted(fn_list)
    output_list = []
    for idx, fn in enumerate(fn_list):
        full_fn = os.path.join(neg_dir, fn)
        with open(full_fn, "r") as fh:
            fdict = json.load(fh)
        exp = fdict["positive"]["exp"]
        numbers_exp = re.findall(r'\d+', exp)
        numbers_exp = sorted(numbers_exp)
        num_exp_str = [str(ele) for ele in numbers_exp]
        num_str = "_".join(num_exp_str).strip()
        output_list.append(num_str)
    pdb.set_trace()
    neg_dir = "output/data_generation/stage3_v3_2_100k" 
    fn_list = os.listdir(neg_dir)
    fn_list = sorted(fn_list)
    #output_list = []
    for idx, fn in enumerate(fn_list):
        full_fn = os.path.join(neg_dir, fn)
        with open(full_fn, "r") as fh:
            fdict = json.load(fh)
        exp = fdict["positive"]["exp"]
        numbers_exp = re.findall(r'\d+', exp)
        numbers_exp = sorted(numbers_exp)
        num_exp_str = [str(ele) for ele in numbers_exp]
        num_str = "_".join(num_exp_str).strip()
        output_list.append(num_str)
    pdb.set_trace()


def get_input_dict(ori_res_dir):
    fn_list = os.listdir(ori_res_dir)
    fn_list = sorted(fn_list)
    output_list = []
    for idx, fn in enumerate(fn_list):
        full_fn = os.path.join(ori_res_dir, fn)
        with open(full_fn, "r") as fh:
            fdict = json.load(fh)
        exp = fdict["positive"]["exp"]
        numbers_exp = re.findall(r'\d+', exp)
        numbers_exp = sorted(numbers_exp)
        num_exp_str = [str(ele) for ele in numbers_exp]
        num_str = "_".join(num_exp_str).strip()
        output_list.append(num_str)
    output_list = list(set(output_list))
    return output_list

if __name__=="__main__":
    #check_unique_input()
    check_negative_list()