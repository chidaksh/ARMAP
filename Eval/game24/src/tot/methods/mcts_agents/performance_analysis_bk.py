import json
import os
import pdb


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

def analyze_performance():

if __name__ == "__main__":
    analyze_performance()
