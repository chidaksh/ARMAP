import argparse
import pdb

def parse_args():
    # task runing
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, default='game24', choices=['game24', 'text', 'crosswords'])
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
    args.add_argument('--output_dir', type=str,
                      default="output/data_generation/stage1_v1")
    args.add_argument('--start_idx', type=int, default=0)
    args.add_argument('--end_idx', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=10)
    args.add_argument('--stage2_output_dir', type=str,
                      default="output/data_generation/stage2_v1")
    args.add_argument('--stage3_output_dir', type=str,
                      default="output/data_generation/stage3_v1")
    args.add_argument('--max_tokens', type=int, default=100)
    args.add_argument('--stage3_previous_output_dir', type=str,
                      default="output/data_generation/stage3_v1")
    args = args.parse_args()
    return args
