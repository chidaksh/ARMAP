import os
import json
import logging
import pathlib
import requests
import argparse
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore
import io
from io import BytesIO
import base64
import json
import sys

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State


logger = logging.getLogger("agent_frame")

info = "You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.\n\nThe available actions are:\nopen OBJ: open a container\nclose OBJ: close a container\nactivate OBJ: activate a device\ndeactivate OBJ: deactivate a device\nconnect OBJ to OBJ: connect electrical components\ndisconnect OBJ: disconnect electrical components\nuse OBJ [on OBJ]: use a device/item\nlook around: describe the current room\nexamine OBJ: describe an object in detail\nlook at OBJ: describe a container's contents\nread OBJ: read a note or book\nmove OBJ to OBJ: move an object to a container\npick up OBJ: move an object to the inventory\npour OBJ into OBJ: pour a liquid into a container\nmix OBJ: chemically mix a container\nteleport to LOC: teleport to a specific room\nfocus on OBJ: signal intent on a task object\nwait: task no action for 10 steps\nwait1: task no action for a step\n"

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

import copy
def convert_history_to_evaluator(history):
    history = copy.deepcopy(history)
    def encode_image(image, encoding_format="PNG"):
        buffered = BytesIO()
        image.save(buffered, format=encoding_format)
        buffered.seek(0)
        return buffered

    '''
    {
  "image": "4.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nNavigation Intent: How much RAM (in GB) does the item with blue LED lights on this page have?"
    },
    {
      "from": "gpt",
      "value": "###test gpt###"
    }
  ],
  "output_1": {
    "from": "llava",
    "value": " Let's think step-by-step. The objective is to find out the amount of RAM (in GB) of the item with blue LED lights on this page. None of the information on this page seems relevant to the objective. Thus, I cannot find the answer to the question. In summary, the next action I will perform is ```stop []```"
  }
},

    '''
    preference_list = []
    factual_list = {}
    
    files = {}
    images = {}
    encoding_format = "PNG"
    preference = None
    img_name = "/nobackup/users/zfchen/cdl/LLaVA-RLHF/data/data_wj/eval/not_used.png"
    factual_list[img_name] = [" "]
    images[img_name] = Image.open(img_name)
    begin = -1
    for i, conv in enumerate(history):
        if 'Task Description:' in conv['content']:
            begin = i

    correct_history = history[begin:]

    # if len(correct_history) == 1:
    #     print(idx)
    #     continue

    action = 'Task Description:' + correct_history[0]['content'].split('Task Description:')[-1].split('Please only output one Thought and one Action.')[0]
    # import pdb; pdb.set_trace()
    conversations = []
    s = ''
    conversations.append({
        "from": "human",
        "value": info + action
    })
    s += info + action
    conversations.append({
        "from": "gpt",
        "value": "###test gpt###"
    })
    total_len = 0

    output_1 = []
    for i, conv in enumerate(correct_history):
        if i == 0:
            continue
        conv['from'] = 'llava' if i % 2 == 1 else 'human'
        conv['value'] = conv['content']
        output_1.append(conv)
        total_len += len(conv['content'])
        s += conv['content']
    # mx = max(mx, total_len)
    if total_len > 6000:
        if len(enc.encode(s)) >  5000:
            return None, None, None
    # if output_1[-1]['from'] != 'human':
    #     print(output_1)
    #     print(idx)
    #     print('gg')

    output_1.append({
        'from': 'llava',
        'value': 'Stop'
    })

    output_1.append({
        'from': 'human',
        'value': 'Please evaluate whether you complete the "Task Description" or not.'
    })
    output_1.append({
        'from': 'llava',
        'value': 'Following your definitions, my task completion score is'
    })
    # import pdb;pdb.set_trace()

    
    preference = dict(
        image=img_name,
        conversations=conversations,
        output_1=output_1,
    )
    preference_list.append(preference)
    for k, v in images.items():
        image = encode_image(v, encoding_format=encoding_format)
        files[k] = image
    return preference_list, factual_list, files

reflection_prompt = open('agents/reflection_few_shot_examples.txt').read()

re_cp = []

first_reward = []
retry_reward = []
import numpy as np

def interactive_loop( # llm generation
    task: tasks.Task,
    agent: agents.LMAgent,
    env_config: Dict[str, Any],
    args
) -> State:
    logger.info(f"Loading environment: {env_config['env_class']}")
    meta_data = {
        "action_history": ["None"],
        "memory": []
    }
    max_num_attempts = args.re_iters
    first_state = None
    for trail_idx in range(max_num_attempts):
        logger.info(f"\n{Fore.RED}trail_idx{trail_idx}{Fore.RESET}")
        env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config) # call task files
        # reset the environment and set the prompt
        
        observation, state = env.reset()
    
        init_msg = observation
    
        logger.info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}")
    
        cur_step = 1
        
        observation_history = [] # to prevent same observation too many times; dead loop
        if trail_idx > 0:
            # state.history[0]['content'] = reflection_prompt + '\n' + state.history[0]['content']
            state.history[-1]['content'] = 'REFLECTIONS FROM PREVIOUS ATTEMPTS\n'+"\n".join(meta_data["memory"]) + "\n" + state.history[-1]['content']
            # import pdb; pdb.set_trace()
        while not state.finished:
            logger.info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n")
            cur_step += 1
            # agent act
            try:
                # import pdb; pdb.set_trace()
                llm_output: str = agent(state.history)
                # color the action in green
                # logger.info(f"\nLM Agent Action:\n\033[92m{action.value}\033[0m")
                logger.info(
                    f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n"
                )
            except Exception as e:
                logger.info(f"Agent failed with error: {e}")
                state.success = False
                state.finished = True
                state.terminate_reason = "exceeding maximum input length"
                break
            # environment step
            observation, state = env.step(llm_output)
            observation_history.append(observation)
            # import pdb; pdb.set_trace()
            # color the state in blue
            if not state.finished:
                # color the observation in blue
                logger.info(
                    f"\n{Fore.BLUE}{observation}{Fore.RESET}\n"
                )
    
            if 'No known action matches that input' in observation:
                state.finished = True
            # prevent repeat and dead loop
            if len(observation_history) > 4 and observation == observation_history[-1] and observation == observation_history[-2] and observation == observation_history[-3] and observation == observation_history[-4]:
                observation = 'Same Observation Repeat Four Times'
                logger.info(
                    f"\n{Fore.GREEN}{observation}{Fore.RESET}\n"
                )
                state.finished = True
    
            if state.finished:
                break
        if not first_state:
            first_state  = copy.deepcopy(state)


        if args.rm == "gt":
            reward = state.reward
        else:
            preference_data, factual_data, files = convert_history_to_evaluator(history=state.history)
            if not preference_data:
                state = first_state
                break
            preference_data = json.dumps(preference_data)
            preference_data = preference_data.encode('utf-8')
            factual_data = json.dumps(factual_data)
            factual_data = factual_data.encode('utf-8')
            data = dict(
                preference_data=preference_data,
                factual_data=factual_data,
                prompt="prompt".encode('utf-8'),
            )
            headers = {
                "User-Agent": "BLIP-2 HuggingFace Space",
            }
            response = requests.post(args.rm, data=data, files=files, headers=headers)
            decoded_string = response.content.decode('utf-8')
            
            score_list = json.loads(decoded_string)
            reward = score_list[0]
        if trail_idx > 0:
            print(first_state.reward)
            print(state.reward)
            # import pdb; pdb.set_trace()
        if trail_idx == 0:
            re_cp.append((state.reward, reward))
        if reward > args.threshold:
            if trail_idx > 0:
                first_reward.append(first_state.reward)
                retry_reward.append(state.reward)
                print("_______________________________")
                print(f"np.mean(first_reward) : {np.mean(first_reward)}")
                print(f"np.mean(retry_reward) : {np.mean(retry_reward)}")
                print("_______________________________")
            break
        if trail_idx == (max_num_attempts - 1):
            # if trail_idx > 0:
            #     first_reward.append(first_state.reward)
            #     retry_reward.append(state.reward)
            #     print("_______________________________")
            #     print(f"np.mean(first_reward) : {np.mean(first_reward)}")
            #     print(f"np.mean(retry_reward) : {np.mean(retry_reward)}")
            #     print("_______________________________")
            state = first_state
            break
        # global total_retry
        # total_retry += 1
        # print(f"total_retry : {total_retry}")
        # meta_data["memory"].append(state.history[16:])

        state.history[-1]['content'] += "\n\nSTATUS: FAIL\n\nNew plan: "
        # import pdb; pdb.set_trace()

        llm_output: str = agent(state.history,is_re = True)
        meta_data["memory"].append(llm_output)
        # meta_data["memory"].append(llm_output.split('\n')[0])
        # import pdb;pdb.set_trace()
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

    if args.task == 'sample-test' or args.task == 'data-fake' or args.task == 'data-gen':
        output_path = args.output_path
        # if not os.path.isdir(output_path):
        #     os.makedirs(output_path)
    else:
        output_path = os.path.join("outputs", agent_config['config']['model_name'].replace('/', '_'), args.exp_config+args.exp_name)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"] # exp_config eval_agent/configs/task/sciworld.json
    
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    if env_config['env_class'] == 'SciWorldEnv':
        from scienceworld import ScienceWorldEnv
        from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']), envStepLimit=200)
        # env_config['env'] = ScienceWorldEnv("", serverPath="/ML-A800/home/yifan/code/AgentPipeline/envs/scienceworld/scienceworld.jar", envStepLimit=200)

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)
    
    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])( # different llms
        agent_config["config"]
    )

    state_list = []

    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(file.split('.')[0])
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
            # Only test 10 tasks in debug mode
            if args.debug and i == 5:
                break

            # skip done tasks
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue

            state = interactive_loop(
                task, agent, env_config, args
            )

            state_list.append(state)
            json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

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
        logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")


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
        default="SciWorld",
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
        "--rm",
        type=str,
        default="gt",
        help="reward model",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold",
    )

    parser.add_argument(
        "--re_iters",
        type=int,
        default=5,
        help="re_iters",
    )
    
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)
    print(len(first_reward))

    import json
    json.dump(re_cp, open('re_cp.json', 'w'), indent=2)
