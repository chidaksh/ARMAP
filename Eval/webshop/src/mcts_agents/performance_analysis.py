import json
import os
import pdb


def parse_performance_mcts():
	result_dir = "PATH_TO_RESULT_DIR"
	fn_list = os.listdir(result_dir)
	mcts_perf, avg_perf, max_perf = [], [], []
	for fn in fn_list:
		full_fn = os.path.join(result_dir, fn)
		with open(full_fn, "r") as fh:
			fdict = json.load(fh)
		reward_pds = fdict["rewards_pred"]
		reward_gts = fdict["rewards_gt"]
		max_val = max(reward_pds)
		max_idx_list = [idx for idx, ele in enumerate(reward_pds) if ele==max_val]
		max_gt_list = [reward_gts[idx] for idx in max_idx_list]
		mcts_rewards = sum(max_gt_list) / len(max_gt_list)
		mcts_perf.append(mcts_rewards)

		mean_reward = sum(reward_gts) / len(reward_gts)
		max_perf.append(max(reward_gts))
		avg_perf.append(mean_reward)
		print(reward_gts)
		if max(reward_gts)==0:
			print(fn)

	mean_p = sum(avg_perf)/len(avg_perf)
	mcts_p = sum(mcts_perf)/len(mcts_perf)
	max_p = sum(max_perf)/len(max_perf)
	print("Testing sample number: %d\n"%(len(fn_list)))
	print("Average MEAN performance: %f\n"%(mean_p))
	print("Average MCTS performance: %f\n"%(mcts_p))
	print("Average MAX performance: %f\n"%(max_p))
	pdb.set_trace()

if __name__=="__main__":
    parse_performance_mcts()