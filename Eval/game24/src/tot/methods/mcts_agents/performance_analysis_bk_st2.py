import json
import os
import pdb
from utils_mcts import call_api 
from tot.tasks import get_task

def analyze_performance():
    result_dir = "output/llm380b4bit_GT_Debug"
    fn_list = os.listdir(result_dir)
    best_reward_list = []
    for fn in fn_list:
        full_fn = os.path.join(result_dir, fn)
        with open(full_fn) as fh:
            fdict = json.load(fh)
        max_re = max(fdict["rewards"])
        best_reward_list.append(max_re)
        if max_re !=1 or len(fdict["rewards"])==1:
            continue
        visualize_example(fdict)
    best_reward = sum(best_reward_list) / len(best_reward_list)
    print("Evaluating number: %d\n"%(len(best_reward_list)))
    print("Best rewards: %f\n"%(best_reward))
    pdb.set_trace()

def visualize_example(fdict):
    print(fdict['trajectories'][-1][2][-1])
    pdb.set_trace()
    
    
def rescore_mcts_trajectories():
    result_dir = "output/llm380b4bit_GT_Debug"
    #result_dir = "output/llm380b4bit_gt_debug_v2"
    #result_dir = "output/llm380b4bit_gt_debug_100"
    fn_list = os.listdir(result_dir)
    best_reward_list = []
    avg_perf = []
    max_perf = []
    mcts_perf = []
    #url = 'http://172.30.150.31:15679/api/generate'
    #url = 'http://172.30.150.32:15679/api/generate'
    #url = 'http://172.30.150.32:15678/api/generate'
    #url = 'http://172.30.150.32:15681/api/generate'
    #url = 'http://172.30.150.32:15682/api/generate'
    url = 'http://9.33.169.150:15681/api/generate'
    output_id = "mcts_4500iter.json"
    raw_pred ={}
    for fn in fn_list:
        full_fn = os.path.join(result_dir, fn)
        with open(full_fn) as fh:
            fdict = json.load(fh)
        max_re = max(fdict["rewards"])
        #if max_re !=1 or len(fdict["rewards"])==1:
        #    continue
        fid = int(fn.replace(".json", ""))
        if fid < 900:
            continue
        task = get_task("game24")
        reward_preds = []
        for track_id, traject_info in enumerate(fdict["trajectories"]): 
            state = {"trajectory": traject_info[2][-1], "input": task.get_input(fid)}
            reward_pred = call_api(url = url, state= state, num_api=1)
            reward_preds.append(reward_pred)
        rewards_gt = fdict["rewards"] 
        avg_r = sum(rewards_gt) / len(rewards_gt)
        max_r = max(rewards_gt)
        max_pred = max(reward_preds)
        max_idx = reward_preds.index(max_pred)
        mcts_r = rewards_gt[max_idx]
        assert len(reward_preds)==len(rewards_gt)
        avg_perf.append(avg_r)
        max_perf.append(max_r)
        mcts_perf.append(mcts_r)
        
        raw_pred[fid] = [reward_preds, rewards_gt]
        
        r_avg = sum(avg_perf) / len(avg_perf)
        r_mcts = sum(mcts_perf) / len(mcts_perf)
        r_max = sum(max_perf) / len(max_perf)
        print("Average: %f\n"%(r_avg))
        print("MCTS: %f\n"%(r_mcts))
        print("MAX: %f\n"%(r_max))
        print(fid)
        print(rewards_gt)
        print(reward_preds)
        pdb.set_trace()
        """
        if mcts_r < avg_r:
            print(rewards_gt)
            print(reward_preds)
            pdb.set_trace()
        """
    full_res_fn = os.path.join(result_dir, output_id)
    with open(full_res_fn, "w") as fh:
        json.dump(raw_pred, fh)

def rescore_cot_trajectories():
    result_fn = "Meta-Llama-3.1-70B-Instruct-AWQ-INT4_0.7_naive_cot_sample_100_start900_end1000.json"
    #result_dir = "logs/game24"
    result_dir = "logs/game24/hugging-quants" 
    full_result_fn = os.path.join(result_dir, result_fn) 
    fh = open(full_result_fn, "r")
    res_list = json.load(fh)
    task = get_task("game24")
    avg_perf = []
    max_perf = []
    mcts_perf = []
    #url = 'http://172.30.150.32:15681/api/generate'
    url = 'http://9.33.169.150:15678/api/generate'
    output_id = "v2_check4500_cot.json"
    raw_pred = {}
    for idx, res_info in enumerate(res_list):
        reward_preds = []
        reward_gts = [info["r"] for info in res_info["infos"]]
        task_id = res_info["idx"]
        for idx2, pred in enumerate(res_info["ys"]):
            state = {"trajectory": pred, "input": task.get_input(res_info["idx"])}
            reward_pred = call_api(url = url, state= state, num_api=1)
            reward_preds.append(reward_pred)
            #pdb.set_trace()
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
        print("Number: %d"%(len(avg_perf)))
    full_res_fn = os.path.join(result_dir, output_id)
    with open(full_res_fn, "w") as fh:
        json.dump(raw_pred, fh)

def parse_mcts_trajectories():
    #result_dir = "output/llm380b4bit_rm09_debug_100_max_tokens_150"
    #result_dir = "output/llm380b4bit_rm05_debug_100"
    result_dir = "output/llm380b4bit_rm07_debug_100"
    output_id = "v2_check4500_mcts.json"
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
        #pdb.set_trace()
        if mcts_r < avg_r:
            print(reward_gts)
            print(reward_preds)
            print("Task id: %d\n"%(task_id))
            print(max_idx)
            print(task.get_input(task_id))
            print(res_dict["trajectories"][max_idx][2][-1])
            #pdb.set_trace()
    print("result number: %d\n"%(len(max_perf)))
    full_res_fn = os.path.join(result_dir, output_id)
    with open(full_res_fn, "w") as fh:
        json.dump(raw_pred, fh)

if __name__ == "__main__":
    #analyze_performance()
    #rescore_mcts_trajectories()
    #rescore_cot_trajectories()
    parse_mcts_trajectories()
