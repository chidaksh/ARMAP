import json
import os
import pdb
from utils_mcts import call_api 
from tot.tasks import get_task

def parse_mcts_trajectories_v2():
    result_dir = "output/llm380b4bit_rm07_0505_max_tokens_1000"
    #result_dir = "output/llm380b4bit_rm07_0520_max_tokens_1000"
    #result_dir = "output/llm380b4bit_rm07_2005_max_tokens_1000"
    #result_dir = "output/llm3_1_8b_rm07_2005_max_tokens_1000"
    #result_dir = "output/llm3_1_8b_rm07_1010_max_tokens_1000"
    #result_dir = "output/llm380b4bit_rm07_2005_max_tokens_1000"
    #result_dir = "output/mixtral_rm07_2005_max_tokens_1000"
    result_dir = "output/phi35_rm07_2005_max_tokens_1000"
    output_id = "check4500_mcts_2005_v2.json"
    fn_list = os.listdir(result_dir)
    fn_list = sorted(fn_list)
    task = get_task("game24")
    avg_perf = []
    max_perf = []
    mcts_perf = []
    raw_pred = {}
    for idx, fn in enumerate(fn_list):
        full_fn = os.path.join(result_dir, fn)
        with open(full_fn, "r") as fh:
            res_dict = json.load(fh)
        try:
            task_id = int(fn.replace(".json", ""))
        except:
            continue
        reward_gts = []
        full_track_fn = os.path.join(result_dir, "multi_"+fn)
        full_track_list = []
        with open(full_track_fn, "r") as fh:
            track_list = json.load(fh)
            full_track_list += track_list 
        for tid, track in enumerate(res_dict["trajectories"]):
            action = track[2][-1].strip()
            reward = task.test_output(task_id, action)["r"]
            reward_gts.append(reward)
        reward_preds = res_dict["rewards"]
        avg_r = sum(reward_gts) / len(reward_gts)
        max_r = max(reward_gts)
        max_pred = max(reward_preds)
        max_idx = reward_preds.index(max_pred)
        mcts_r = reward_gts[max_idx]
        assert len(reward_preds)==len(reward_gts)
        avg_perf.append(avg_r)
        max_perf.append(max_r)
        mcts_perf.append(mcts_r)
        raw_pred[task_id] = [reward_preds, reward_gts]
        
        r_avg = sum(avg_perf) / len(avg_perf)
        r_mcts = sum(mcts_perf) / len(mcts_perf)
        r_max = sum(max_perf) / len(max_perf)
    print("Average: %f\n"%(r_avg))
    print("MCTS: %f\n"%(r_mcts))
    print("MAX: %f\n"%(r_max))
    print("result number: %d\n"%(len(max_perf)))
    full_res_fn = os.path.join(result_dir, output_id)
    with open(full_res_fn, "w") as fh:
        json.dump(raw_pred, fh)


def parse_mcts_trajectories_v3():
    #result_dir = "output/llm380b4bit_rm07_0505_max_tokens_1000"
    #result_dir = "output/llm380b4bit_rm07_0520_max_tokens_1000"
    #result_dir = "output/llm380b4bit_rm07_1010_max_tokens_1000"
    #result_dir = "output/llm380b4bit_rm07_0505_max_tokens_1000"
    #result_dir = "output/llm380b4bit_rm07_2005_max_tokens_1000"
    #result_dir = "output/llm3_1_8b_rm07_2005_max_tokens_1000"
    #result_dir = "output/llm3_1_8b_rm07_1010_max_tokens_1000"
    #result_dir = "output/phi35_rm07_2005_max_tokens_1000"
    #result_dir = "output/mixtral_rm05_2005_max_tokens_1000"
    #result_dir = "output/phi35_rm05_2005_max_tokens_1000"
    #result_dir = "output/phi35_rm05_0520_max_tokens_1000"
    #output_id = "check4500_mcts_0520_rm05_v3.json"
    result_dir = "output/mixtral_rm05_0520_max_tokens_1000"
    output_id = "check4500_mcts_0520_rm05_v3.json"
    fn_list = os.listdir(result_dir)
    fn_list = sorted(fn_list)
    task = get_task("game24")
    avg_perf = []
    max_perf = []
    mcts_perf = []
    raw_pred = {}
    top_k = 100
    for idx, fn in enumerate(fn_list):
        full_fn = os.path.join(result_dir, fn)
        with open(full_fn, "r") as fh:
            res_dict = json.load(fh)
        try:
            task_id = int(fn.replace(".json", ""))
        except:
            continue
        reward_gts = []
        full_track_fn = os.path.join(result_dir, "multi_"+fn)
        full_track_list = []
        with open(full_track_fn, "r") as fh:
            track_list = json.load(fh)
            full_track_list += track_list 
        reward_preds = [ele[2] for ele in full_track_list ]
        reward_gts = [ele[1] for ele in full_track_list ]
        reward_gts = reward_gts[:top_k]
        reward_preds = reward_preds[:top_k]
        avg_r = sum(reward_gts) / len(reward_gts)
        max_r = max(reward_gts)
        max_pred = max(reward_preds)
        max_idx = reward_preds.index(max_pred)
        mcts_r = reward_gts[max_idx]
        assert len(reward_preds)==len(reward_gts)
        avg_perf.append(avg_r)
        max_perf.append(max_r)
        mcts_perf.append(mcts_r)
        raw_pred[task_id] = [reward_preds, reward_gts]
        
        r_avg = sum(avg_perf) / len(avg_perf)
        r_mcts = sum(mcts_perf) / len(mcts_perf)
        r_max = sum(max_perf) / len(max_perf)
       
        #if max(reward_preds) != max(res_dict['rewards']) and mcts_r ==1:
        #    pdb.set_trace()
    print("Average: %f\n"%(r_avg))
    print("MCTS: %f\n"%(r_mcts))
    print("MAX: %f\n"%(r_max))

    print("result number: %d\n"%(len(max_perf)))
    print(result_dir)
    full_res_fn = os.path.join(result_dir, output_id)
    with open(full_res_fn, "w") as fh:
        json.dump(raw_pred, fh)


def parse_cot_result_v2():
    result_fn = "Meta-Llama-3.1-70B-Instruct-AWQ-INT4_0.7_naive_cot_sample_100_start900_end1000.json"
    result_dir = "logs/game24/hugging-quants" 
    full_result_fn = os.path.join(result_dir, result_fn) 
    fh = open(full_result_fn, "r")
    res_list = json.load(fh)
    task = get_task("game24")
    avg_perf = []
    max_perf = []
    mcts_perf = []
    output_id = "v2_check4500_cot_new.json"
    full_result_fn = os.path.join(result_dir, output_id) 
    fh = open(full_result_fn, "r")
    output_dict = json.load(fh)
    raw_pred = {}
    avg_perf = []
    max_perf = []
    mcts_perf = []
    raw_pred = {}
    
    top_k = 10
    for idx, res_info in enumerate(res_list):
        
        reward_gts = [info["r"] for info in res_info["infos"]]
        task_id = int(res_info["idx"])
        reward_preds = output_dict[str(task_id)][0]
        reward_gts2 = output_dict[str(task_id)][1]
        assert reward_gts==reward_gts2

        reward_gts = reward_gts[:top_k]
        reward_preds = reward_preds[:top_k]
        
        avg_r = sum(reward_gts) / len(reward_gts)
        max_r = max(reward_gts)
        max_pred = max(reward_preds)
        max_idx = reward_preds.index(max_pred)
        mcts_r = reward_gts[max_idx]
        assert len(reward_preds)==len(reward_gts)
        avg_perf.append(avg_r)
        max_perf.append(max_r)
        mcts_perf.append(mcts_r)
        raw_pred[task_id] = [reward_preds, reward_gts]
        
        r_avg = sum(avg_perf) / len(avg_perf)
        r_mcts = sum(mcts_perf) / len(mcts_perf)
        r_max = sum(max_perf) / len(max_perf)
    print("Average: %f\n"%(r_avg))
    print("SEL: %f\n"%(r_mcts))
    print("MAX: %f\n"%(r_max))



if __name__ == "__main__":
    #parse_cot_trajectories()
    #parse_mcts_trajectories_v2()
    parse_mcts_trajectories_v3()
    #parse_cot_result_v2()
