import pdb
import random
import os
import json
from llm_models import llm_prompt_model
from utils import load_products


def step_1_sample_data_v2(args):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    all_products, product_item_dict, product_prices, attribute_to_asins = load_products(args, num_products=None, human_goals=True)
    product_ins = [ item  for item in all_products if 'instructions'  in item ]
    with open(args.test_goal_path, "r") as fh:
        test_goal_list = json.load(fh)
    test_id_list = [ item["asin"] for item in test_goal_list ]
    random.shuffle(all_products)
    # sample product list
    sampled_list = []
    for idx, product_info in enumerate(all_products):
        if product_info["asin"] not in test_id_list:
            sampled_list.append(product_info)
        if len(sampled_list) >= args.sample_product_num:
            break
    # sample some training instruction samples for data generation
    train_product_ins = [item for item in product_ins if item["asin"] not in test_id_list] 
    train_product_ins = train_product_ins[:100]
    
    out_fn = "step1_sample_%d.json" % (args.sample_product_num)
    out_path = os.path.join(args.output_dir, out_fn)
    with open(out_path, "w") as fh:
        json.dump({"sampled_products": sampled_list, "train_instructions": train_product_ins}, fh)
    pdb.set_trace()

def step_1_sample_data(args):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    out_fn = "step1_sample_%d.json" % (args.sample_product_num)
    out_path = os.path.join(args.output_dir, out_fn)

    from src.server.tasks.webshop_docker.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    pdb.set_trace()
    env = WebAgentTextEnv(observation_mode="html", human_goals=True)
    test_goal_num = 500
    test_id_list = []
    for idx, goal in enumerate(env.server.goals):
        if idx >= test_goal_num:
            break
        test_id_list.append(goal["asin"])
    random.shuffle(env.server.all_products)
    sampled_list = []
    new_price_dict = []
    for idx, product_info in enumerate(env.server.all_products):
        if product_info["asin"] not in test_id_list:
            product_info["price_for goal"] = env.server.product_prices[product_info["asin"]] 
            #product_info["price_upper"] = env.server.product_prices[product_info["asin"]] 
            sampled_list.append(product_info)
        if len(sampled_list) > args.sample_product_num:
            break
    with open(out_path, "w") as fh:
        json.dump({"products": sampled_list, "prices": new_price_dict}, fh)
    print("Finish sampling %d samples" % (args.sample_product_num))


def step_2_generate_goal(args):
    if not os.path.isdir(args.stage2_output_dir):
        os.makedirs(args.stage2_output_dir)
    out_fn = "step1_sample_%d.json" % (args.sample_product_num)
    out_path = os.path.join(args.output_dir, out_fn)
    with open(out_path, "r") as fh:
        data_dict = json.load(fh)
        smp_product_list = data_dict["sampled_products"]
        smp_instruct_list = data_dict["train_instructions"]
    goal_generator = llm_prompt_model(args)
    #for idx, smp_product in enumerate(smp_product_list[:1000]):
    for idx, smp_product in enumerate(smp_product_list):
        cache_path = os.path.join(args.stage2_output_dir, smp_product["asin"]+".json")
        if os.path.isfile(cache_path):
            print("Product %s has been generated.\n"%(cache_path))
            continue
        try:
            response, missing_dict = goal_generator.generate_synthetic_goal(smp_product)
        except:
            print("Fail to generate %s\n"%(cache_path))
            continue
        output_dict = {"instruction": response, "ins_selection": missing_dict}
        with open(cache_path, "w") as fh:
            json.dump(output_dict, fh)

def step_3_pack_data(args):
    # load the original data
    BASE_DIR = args.original_data_dir
    HUMAN_ATTR_PATH = os.path.join(BASE_DIR, 'items_human_ins.json')
    DEFAULT_FILE_PATH = os.path.join(BASE_DIR, 'items_shuffle.json')
    
    with open(HUMAN_ATTR_PATH) as f:
        human_attributes = json.load(f)
    attr_keys = list(human_attributes.keys())
    attr_vals = list(human_attributes.values())
    fn_list = os.listdir(args.stage2_output_dir)
    syn_attr_dict = {}
    for fn in fn_list:
        full_fn = os.path.join(args.stage2_output_dir, fn)
        with open(full_fn, "r") as fh:
            response_info = json.load(fh) 
        asin = fn.replace(".json", "")
        tmp_dict = {"asin": asin, "instruction": response_info["instruction"]}
        tmp_dict.update(response_info["ins_selection"])
        syn_attr_dict[asin] = [tmp_dict]
    with open(args.stage3_output_path, "w") as fh:
        json.dump(syn_attr_dict, fh)

def get_anno_statistics():
    from src.server.tasks.webshop_docker.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    env = WebAgentTextEnv(observation_mode="html", human_goals=True)
    for idx, goal in enumerate(env.server.goals):
        if "goal_options" not in goal:
            continue
        print(goal["instruction_text"])
        print(goal["goal_options"])
        print(goal["attributes"])
        print(goal["price_upper"])
        tar_item = env.server.product_item_dict[goal["asin"]]
        pdb.set_trace()


def sample_goal_data(args):
    from src.server.tasks.webshop_docker.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    env = WebAgentTextEnv(observation_mode="html", human_goals=True)
    test_num = 500
    smp_train_num = 100
    goal_id_list = [goal['asin']
                    for idx, goal in enumerate(env.server.goals) if idx < test_num]
    test_list = [goal for idx, goal in enumerate(
        env.server.goals) if idx < test_num]
    test_goal_path = os.path.join(args.output_dir, "test_goal.json")
    train_goal_path = os.path.join(args.output_dir, "train_sample_goal.json")
    with open(test_goal_path, "w") as fh:
        json.dump(test_list, fh)
    print("Finish sampling test goals\n")
    random.shuffle(env.server.goals)
    smp_goal_list = []
    for idx, goal in enumerate(env.server.goals):
        if goal["asin"] in goal_id_list:
            continue
        smp_goal_list.append(goal)
        if len(smp_goal_list) >= smp_train_num:
            break
    with open(train_goal_path, "w") as fh:
        json.dump(smp_goal_list, fh)
    print("Finish sampling training goals\n")
    
def step_4_construct_triplet(args):
    track_info_dict1 = {}
    with open(args.stage4_input_track_path1, "r") as fh:
        track_info_list1 = list(fh)
    for idx, track_info_str in enumerate(track_info_list1):
        track_info1 = json.loads(track_info_str)
        track_info_dict1[track_info1["index"]] = track_info1 
    
    track_info_dict2 = {}
    with open(args.stage4_input_track_path2, "r") as fh:
        track_info_list2 = list(fh)
    for idx, track_info_str in enumerate(track_info_list2):
        track_info2 = json.loads(track_info_str)
        track_info_dict2[track_info2["index"]] = track_info2
    
    output_dict = {}
    out_num1, out_num2 = 0, 0
    for track_key, track_info1 in track_info_dict1.items():
        #print(track_info1["output"]["result"]["history"][-2]["action"])
        #print(track_info1["output"]["result"]["history"][-1]["action"])
        if track_key not in track_info_dict2:
            print("Skipping key: %d"%(track_key))
            continue
        
        print("Analyzing key: %d"%(track_key))
        track_info2 = track_info_dict2[track_key]
        track_info1_result = track_info1["output"]["result"]
        track_info2_result = track_info2["output"]["result"]
        if track_info1_result["reward"]>track_info2_result["reward"]:
            output_dict[track_key] = {"positive": track_info1, "negative": track_info2}
            out_num1 +=1
        elif track_info1_result["reward"]<track_info2_result["reward"]:
            output_dict[track_key] = {"positive": track_info2, "negative": track_info1}
            out_num2 +=1
        
    print("Constructing %d samples\n"%len(output_dict))
    print("%d samples is better for%s\n"%(out_num1, args.stage4_input_track_path1))
    print("%d samples is better for%s\n"%(out_num2, args.stage4_input_track_path2))
    with open(args.stage4_output_path, "w") as fh:
        json.dump(output_dict, fh)


def step_4_construct_triplet_v2(args):
    track_info_dict1 = {}
    count = 0
    #ignore_list = [2784]
    igore_list = []
    with open(args.stage4_input_track_path1, "r") as fh:
        for line in fh:
            track_info1 = json.loads(line)
            track_info_dict1[track_info1["index"]] = track_info1
            print("track path1:%d"%count)
            count +=1
            # skip bad annotation.
            #if count in ignore_list and "llama370B_v1_10k_fix" in args.stage4_input_track_path1:
            #if count in ignore_list and "llama370B_v1_10k_fix" in args.stage4_input_track_path1:
            #    break
    
    track_info_dict2 = {}
    count = 0
    with open(args.stage4_input_track_path2, "r") as fh:
        for line in fh:
            track_info2 = json.loads(line)
            if track_info2["index"] in track_info_dict1:
                track_info_dict2[track_info2["index"]] = track_info2
            print("track path2:%d"%count)
            count +=1
    #import pdb
    #pdb.set_trace()
    
    output_dict = {}
    out_num1, out_num2 = 0, 0
    for track_key, track_info1 in track_info_dict1.items():
        if track_key not in track_info_dict2:
            print("Skipping key: %d"%(track_key))
            continue
        
        print("Analyzing key: %d"%(track_key))
        track_info2 = track_info_dict2[track_key]
        track_info1_result = track_info1["output"]["result"]
        track_info2_result = track_info2["output"]["result"]
        if track_info1_result["reward"]>track_info2_result["reward"]:
            output_dict[track_key] = {"positive": track_info1, "negative": track_info2}
            out_num1 +=1
        elif track_info1_result["reward"]<track_info2_result["reward"]:
            output_dict[track_key] = {"positive": track_info2, "negative": track_info1}
            out_num2 +=1
        
    print("Constructing %d samples\n"%len(output_dict))
    print("%d samples is better for%s\n"%(out_num1, args.stage4_input_track_path1))
    print("%d samples is better for%s\n"%(out_num2, args.stage4_input_track_path2))
    dir_path = os.path.dirname(args.stage4_output_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(args.stage4_output_path, "w") as fh:
        json.dump(output_dict, fh)
