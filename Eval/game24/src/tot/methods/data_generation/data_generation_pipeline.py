import argparse
import ast
import json
import os
import re
from tot.prompts.game24_data import *
from tot.models import gpt
from tot.argparser import parse_args
import pdb
from data_analysis import get_input_dict, check_exist


def stage1_v1(args):
    """
    generate input and proposal list
    """
    for idx in range(args.start_idx, args.end_idx, args.batch_size):
        out_fn = os.path.join(args.output_dir, "%d.json" % (idx))
        if os.path.isfile(out_fn):
            continue
        if idx < args.start_idx:
            continue
        if os.path.isfile(out_fn):
            print("%s exists. Skipping." % (out_fn))
            continue
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        prompt = cot_prompt
        raw_output = gpt(prompt, model=args.backend, n=args.batch_size, stop=None,
                         temperature=args.temperature)
        print(raw_output[0])
        with open(out_fn, "w") as fh:
            json.dump(raw_output, fh)
        print(idx)


def stage2_v1(args):
    fn_list = os.listdir(args.output_dir)
    full_list = []
    for idx, fn in enumerate(fn_list):
        full_fn = os.path.join(args.output_dir, fn)
        with open(full_fn, "r") as fh:
            f_list = json.load(fh)
            full_list.extend(f_list)
    output_list = []
    for idx, smp in enumerate(full_list):
        line_list = smp.split("\n")
        if len(line_list) < 4:
            continue
        start_prefix = "Thus, we have "
        number_prefix = "The four numbers are"
        if start_prefix not in line_list[3]:
            continue
        if not number_prefix in line_list[3]:
            continue
        exp = line_list[3].replace(start_prefix, "")
        exp = exp.split("=")[0].strip()
        number_str = line_list[3].split(number_prefix)[
            1].replace(".", "").strip()
        numbers_exp = re.findall(r'\d+', exp)
        numbers_parse = re.findall(r'\d+', number_str)
        if sorted(numbers_exp) != sorted(numbers_parse):
            continue
        chain_exp = exp_to_chain_of_thought(exp, line_list, numbers_exp)
        if len(chain_exp)==0:
            continue
        out_dict = {"exp": exp, "thoughts": chain_exp, "raw_data": smp} 
        output_list.append(out_dict)
    print("Original: %d, Refined: %d\n"%(len(full_list), len(output_list)))
    if not os.path.isdir(args.stage2_output_dir):
        os.makedirs(args.stage2_output_dir)
    out_fn = os.path.join(args.stage2_output_dir, "samples.json")
    with open(out_fn, "w") as fh:
        json.dump(output_list, fh)

def exp_to_chain_of_thought(exp, line_list, numbers_exp):
    """
    Change example to chain of thoughts
    """
    thought_list = []
    for idx in range(2, -1, -1):
        line = line_list[idx] 
        line = line[:-1] if line.endswith(".") else line
        print(line)
        try:
            output, input = line.split("can be obtained by ")
        except:
            try:
                output, input = line.split("can be obtain by ")
            except:
                return []
        tmp_exp = input.strip() + " = " + output.strip()
        input_ele = re.findall(r'\d+', input)
        output_ele = re.findall(r'\d+', output)
        for ele in input_ele:
            try:
                ele_idx = numbers_exp.index(ele)
                numbers_exp.pop(ele_idx)
            except:
                return []
        numbers_exp.extend(output_ele)
        numbers_exp = sorted(numbers_exp, key=int) 
        ele_str = " ".join(numbers_exp).strip()
        tmp_full_exp = "%s (left: %s)"%(tmp_exp, ele_str)
        #print(tmp_full_exp)
        thought_list.append(tmp_full_exp)
    return thought_list

def stage3_v1(args):
    if not os.path.isdir(args.stage3_output_dir):
        os.makedirs(args.stage3_output_dir)
    out_fn = os.path.join(args.stage2_output_dir, "samples.json")
    with open(out_fn, "r") as fh:
        smp_list = json.load(fh)
    
    output_list = get_input_dict(args.stage3_previous_output_dir)
    for idx, smp in enumerate(smp_list):
        full_out_path = os.path.join(args.stage3_output_dir, "%d.json"%(idx))
        if os.path.isfile(full_out_path):
            continue
        if idx < args.start_idx:
            continue
        exist_flag = check_exist(smp, output_list)
        if exist_flag:
            print("%s exist. Skipping"%smp["exp"])
            continue
        
        input_ele = re.findall(r'\d+', smp['exp'])
        input_str = " ".join(input_ele).strip()
        steps = "\n".join(smp["thoughts"]) + "\n"
        sample_input = "Input: " + input_str + "\nCorrect Answer:\nSteps:\n" + steps + smp["exp"] + " = 24"
        prompt = cot_negative_prompt.format(NewInput=sample_input)
        raw_output = gpt(prompt, model=args.backend, n=args.batch_size, stop=None,
                         temperature=args.temperature)
        smp_out = {"positive": smp, "negative": raw_output, "idx": idx}
        with open(full_out_path, "w") as fh: 
            json.dump(smp_out, fh)
        print(raw_output[0])



if __name__ == "__main__":
    args = parse_args()
    if args.stage == "stage1":
        stage1_v1(args)
    elif args.stage == "stage2":
        stage2_v1(args)
    elif args.stage == "stage3":
        stage3_v1(args)
