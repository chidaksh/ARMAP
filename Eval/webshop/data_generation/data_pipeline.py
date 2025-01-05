import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
LIB_DIR= os.path.dirname(SCRIPT_DIR)
sys.path.append(LIB_DIR)
from utils import config, build_prompt_v1, set_debugger
from data_modules import step_1_sample_data, step_2_generate_goal, get_anno_statistics, sample_goal_data, step_3_pack_data, step_1_sample_data_v2
from data_modules import step_4_construct_triplet, step_4_construct_triplet_v2
import json
import pdb

set_debugger()

def run_data_generation(args):
    # step 1: sample products from the database
    if args.step_id == 1:
        step_1_sample_data(args)
        #step_1_sample_data_v2(args)
    # step 2: generate intent samples based on attributes
    elif args.step_id == 2:
        step_2_generate_goal(args)
    # step 3: pack data into webshop formats
    elif args.step_id == 3:
        step_3_pack_data(args)
    elif args.step_id == 4:
        #step_4_construct_triplet(args)
        step_4_construct_triplet_v2(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = config()
    #build_prompt_v1()
    run_data_generation(args)
    #sample_goal_data(args)
    #sample_goal_data(args)
    # get_anno_statistics()
