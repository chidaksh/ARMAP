import argparse
import pdb

def parse_args():
    # task runing
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, default='game24', choices=['game24', 'text', 'crosswords', 'game24MCTS'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    args.add_argument('--llm_api', type=str)
    # data generation
    args.add_argument('--stage', type=str,
                      choices=['stage1', 'stage2', 'stage3'], default='stage2')
    args.add_argument('--start_idx', type=int, default=0)
    args.add_argument('--end_idx', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=10)
    args.add_argument('--stage2_output_dir', type=str,
                      default="output/data_generation/stage2_v1")
    args.add_argument('--stage3_output_dir', type=str,
                      default="output/data_generation/stage3_v1")
    args.add_argument('--max_tokens', type=int, default=100)
    # param for MCTS
    #args.add_argument('--reward_func', type=str, default="prompt")
    args.add_argument('--reward_func', type=str, default="gt")
    args.add_argument('--horizon', type=int, default=10)
    args.add_argument('--seq_num', type=int, default=10)
    args.add_argument('--rollouts', type=int, default=10)
    args.add_argument('--output_dir', type=str, default="output/llm380b4bit_gt_debug")
    args.add_argument('--mcts_run', action='store_true')
    args.add_argument('--rm_api', type=str)
    args.add_argument('--job_id', type=int, default=0)
    args.add_argument('--reward', type=str, default='gt')
    args.add_argument('--re_iterations', type=int, default=5)

    args = args.parse_args()
    return args

def parse_args_bk():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo'], default='gpt-4')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords', 'game24MCTS'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    args.add_argument('--mcts_run', action='store_true')
    # param for MCTS
    #args.add_argument('--reward_func', type=str, default="prompt")
    args.add_argument('--reward_func', type=str, default="gt")
    args.add_argument('--horizon', type=int, default=10)
    args.add_argument('--seq_num', type=int, default=10)
    args.add_argument('--rollouts', type=int, default=10)
    args.add_argument('--output_dir', type=str, default="output/llm380b4bit_GT_Debug")

    args = args.parse_args()
    return args

