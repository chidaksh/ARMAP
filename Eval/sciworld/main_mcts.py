import os
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore

import sys

sys.path.append('./ARMAP/Eval/sciworld')

import sciworld.tasks as tasks
import sciworld.agents as agents
import sciworld.envs as envs
from sciworld.utils.datatypes import State

import pdb
import mcts_agents.uct as uct
from mcts_agents.env_scienceworld_wrapper import sciWEnv, build_uct_agent, localAPIPolicy
from mcts_agents.utils_mcts import save_trajectory_history, filter_done_ids
logger = logging.getLogger("agent_frame")



def interactive_loop(  # llm generation
        task: tasks.Task,
        agent: agents.LMAgent,
        env_config: Dict[str, Any],
        output_path: str,
        args: argparse.Namespace
) -> State:
    logger.info(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)  # call task files
    # reset the environment and set the prompt
    observation, state = env.reset()

    # mcts settings
    env_mcts = sciWEnv(env_ori=env, reward_func=args.reward_func, horizon=args.horizon)
    default_policy = localAPIPolicy(model=agent, env=env_mcts, seq_num=args.seq_num, horizon=args.horizon) 
    agent_mcts = uct.UCT(default_policy=default_policy, rollouts=args.rollouts)
    
    init_msg = observation

    logger.info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}")

    cur_step = 1

    observation_history = []  # to prevent same observation too many times; dead loop
    while not state.finished:
        logger.info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n")
        cur_step += 1
        try:
            llm_output = agent_mcts.act(env_mcts, done=False)
            save_trajectory_history(agent_mcts.rolled_out_trajectories, agent_mcts.rolled_out_rewards, task, output_path)
            break
            logger.info(
                f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n"
            )
        #else:
        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = "exceeding maximum input length"
            break
        # environment step
        observation, state = env.step(llm_output)
        observation_history.append(observation)
        # color the state in blue
        if not state.finished:
            # color the observation in blue
            logger.info(
                f"\n{Fore.BLUE}{observation}{Fore.RESET}\n"
            )

        if 'No known action matches that input' in observation:
            state.finished = True
        # prevent repeat and dead loop
        if len(observation_history) > 4 and observation == observation_history[-1] and observation == \
                observation_history[-2] and observation == observation_history[-3] and observation == \
                observation_history[-4]:
            observation = 'Same Observation Repeat Four Times'
            logger.info(
                f"\n{Fore.GREEN}{observation}{Fore.RESET}\n"
            )
            state.finished = True

        if state.finished:
            break

    if state.reward is not None:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}"
        )
    else:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}"
        )
    return state


def main(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)

    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name

    if args.model_url_add is not None:
        agent_config['config']['model_url_add'] = args.model_url_add
        print(agent_config["config"]["model_url_add"])

    if args.task == 'sample-test' or args.task == 'data-fake' or args.task == 'data-gen':
        output_path = args.output_path
    else:
        output_path = os.path.join("outputs", agent_config['config']['model_name'].replace('/', '_'),
                                   args.exp_config + args.exp_name)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]  

    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    if env_config['env_class'] == 'WebShopEnv':
        from webshop.web_agent_site.envs import WebAgentTextEnv
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)
    elif env_config['env_class'] == 'SciWorldEnv':
        from scienceworld import ScienceWorldEnv
        from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']),
                                            envStepLimit=200)
    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)

    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(  # different llms
        agent_config["config"]
    )

    state_list = []

    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            full_output_path = os.path.join(output_path, file)
            done_task_id.append(file.split('.')[0])
        if args.minimal_sample_num > 0:
            done_task_id = filter_done_ids(done_task_id, output_path, args.minimal_sample_num)
        logger.info(f"Existing output file found. {len(done_task_id)} tasks done.")

    if len(done_task_id) == n_tasks:
        logger.info("All tasks done. Exiting.")
        return

    # run the loop for all tasks
    logging.info(f"Running interactive loop for {n_tasks} tasks.")
    n_todo_tasks = n_tasks - len(done_task_id)  # only run the remaining tasks

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(all_tasks):

            # skip done tasks
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue
            if i < args.start_id:
                print("Skipping %d, %s since the start id is %d."%(i, str(task.task_id), args.start_id))
                continue

            full_output_path = os.path.join(output_path, "%s.json"%(str(task.task_id)))
            if os.path.isfile(full_output_path):
                if args.minimal_sample_num==0:
                    print("%s exist. Skipping!"%full_output_path)
                    continue
                else:
                    with open(full_output_path) as fh:
                        fdict = json.load(fh)
                        if len(fdict["rewards"])>=args.minimal_sample_num:
                            print("%s exist. Skipping!"%full_output_path)
                            continue
            state = interactive_loop(
                task, agent, env_config, output_path, args
            )

            state_list.append(state)
            
            pbar.update(1)
        pbar.close()

    logger.warning("All tasks done.")
    logger.warning(f"Output saved to {output_path}")

    # calculate metrics
    reward_list = []
    success_list = []
    for state in state_list:
        if state.reward is not None:
            reward_list.append(state.reward)
        success_list.append(state.success)

    if len(reward_list) != 0:
        logger.warning(f"Average reward: {sum(reward_list) / len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list) / len(success_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")

    # for data generation
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="The task of data generation, including test set generation and train",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="The file path to save data",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiemnt.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="webshop",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="fastchat",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to run in interactive mode for demo purpose.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=10,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--seq_num",
        type=int,
        default=5,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--reward_func",
        type=str,
        default="gt",
        help="reward func GT vs api address"
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="id to start generation",
    )
    parser.add_argument(
        "--minimal_sample_num",
        type=int,
        default=0,
        help="rerun the sampling if failing to get so many samples",
    )
    parser.add_argument(
        "--model_url_add",
        type=str,
        required=False,
        help="to design which address to use",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)
