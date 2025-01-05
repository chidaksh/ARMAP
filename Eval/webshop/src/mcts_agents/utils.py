import argparse
from gym import spaces
import itertools
import json
import requests
from src.client.agents.new_local_agent import NewLocal
import pdb
import os

prompt: str = """
You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions, \
you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
search[keywords]
click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:

Thought:
I think ...

Action:
click[something]
"""


def combinations(space):
    if isinstance(space, spaces.Discrete):
        return range(space.n)
    elif isinstance(space, spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError

def multigpu_breakpoint():
    import torch
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            breakpoint()
        else:
            torch.distributed.barrier()


def gemini(env, history, llm_api):
    state_ori = env.state
    env.reset(env.session_index, state=history)
    history_input = []
    history_input.append({"role": "user", "content": prompt})
    history_input.append({"role": "agent", "content": "Ok."})

    # one shot

    history_input.append({'role': 'user',
                          'content': 'Observation:\n"WebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Search"\n\nAvailable Actions:\n{"has_search_bar": true, "clickables": ["..."]}'})
    history_input.append({'role': 'agent',
                          'content': 'Thought:\nI think I should use the search bar to look for the product I need.\n\nAction:\nsearch[l\'eau d\'issey 6.76 fl oz bottle price < 100.00]'})
    history_input.append({'role': 'user',
                          'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] B000MJZOPK [SEP] L\'eau d\'Issey by Issey Miyake for Women 3.3 oz Eau de Toilette Spray [SEP] $49.98 [SEP] B0012S249E [SEP] L\'eau D\'issey By Issey Miyake For Women. Shower Cream 6.7-Ounces [SEP] $31.36 [SEP] B01H8PGKZS [SEP] L\'eau D\'Issey FOR MEN by Issey Miyake - 6.7 oz EDT Spray [SEP] $67.97 [SEP] B00G3C8FHE [SEP] L\'Eau d\'Issey pour Homme - Eau de Toilette 4.2 fl oz [SEP] $51.25 [SEP] B000R94HRG [SEP] Issey Miyake L\'Eau D\'Issey Pour Homme Eau De Toilette Natural Spray [SEP] $44.99 [SEP] B000C214CO [SEP] Issey Miyake L\'eau D\'issey Eau de Toilette Spray for Men, 4.2 Fl Oz [SEP] $53.99 [SEP] B0018SBRDC [SEP] Issey Miyake L\'eau d\'Issey for Women EDT, White, 0.84 Fl Oz [SEP] $27.04 [SEP] B000XEAZ9Y [SEP] L\'eau De Issey By Issey Miyake For Men. Eau De Toilette Spray 6.7 Fl Oz [SEP] $67.08 [SEP] B079HZR2RX [SEP] L\'eau d\'Issey Pure by Issey Miyake for Women 3.0 oz Nectar de Parfum Spray [SEP] $71.49"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "..."]}'})
    history_input.append({'role': 'agent',
                          'content': 'Thought:\nI think I should click on the product I need, which is B000VOHH8I.\n\nAction:\nclick[B000VOHH8I]'})
    history_input.append({'role': 'user',
                          'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "..."]}'})
    history_input.append({'role': 'agent',
                          'content': 'Thought:\nI think I should click on the \'6.76 fl oz (pack of 1)\' option to select the size I need.\n\nAction:\nclick[6.76 fl oz (pack of 1)]'})
    history_input.append({'role': 'user',
                          'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "..."]}'})
    history_input.append({'role': 'agent',
                          'content': 'Thought:\nI think I should click on the \'Buy Now\' button to purchase the product.\n\nAction:\nclick[Buy Now]'})
    previous_states, previous_actions = history
    if len(previous_states) != len(previous_actions)+1:
        previous_states_copy = previous_states[:-2]
    else:
        previous_states_copy = previous_states
    for idx, html_obs in enumerate(previous_states_copy[:-1]):
        obs = env.env.convert_html_to_text(html_obs, simple=True)

        history_input.append({'role': 'user',
                              'content': f'Observation:\n"{obs}"'})
        history_input.append({'role': 'agent',
                              'content': f'Action:\n{previous_actions[idx]}'})
    available_actions = env.env.get_available_actions()
    html_observation = env.env.observation
    observation = env.env.convert_html_to_text(html_observation, simple=True)
    history_input.append(
        {
            "role": "user",
            "content": f"Observation:\n{observation}\n\n"
                       f"Available Actions:\n{available_actions}",
        }
    )
    print(history_input[-1])
    history_input = json.dumps(history_input)
    data = dict(
        history=history_input,
    )
    headers = {
        "User-Agent": "BLIP-2 HuggingFace Space",
    }
    images = {}
    files = {}
    response = requests.post(llm_api, data=data, headers=headers)
    decoded_string = response.content.decode('utf-8')
    env.reset(env.session_index, state=state_ori)
    """
    aa = json.loads(history_input)
    print(decoded_string)
    pdb.set_trace()
    """
    return decoded_string



def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="configs/assignments/default.yaml"
    )
    parser.add_argument(
        "--auto-retry", "-r", action="store_true", dest="retry"
    )
    parser.add_argument(
        "--llm_api", type=str, default=""
    )
    parser.add_argument(
        "--rm_api", type=str, default=None
    )
    parser.add_argument(
        "--horizon", type=int, default=10
    )
    parser.add_argument(
        "--seq_num", type=int, default=10
    )
    parser.add_argument(
        "--debug", action="store_true"
    )
    parser.add_argument(
        "--rollouts", type=int, default=10
    )
    parser.add_argument(
        "--with_vllm_api", action="store_true"
    )
    parser.add_argument(
        "--temperature", type=float, default=1
    )
    parser.add_argument(
        "--model_id", type=str, default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    )
    parser.add_argument(
        "--api_token", type=str, default="token-abc123"
    )
    parser.add_argument(
        "--split", type=str, default="dev"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/mcts_debug"
    )
    parser.add_argument(
        "--beam_search", action="store_true"
    )
    args = parser.parse_args()
    assert args.split in ["dev", "test"]
    if args.split=="dev":
        args.start_idx, args.end_idx = 200, 280 
    else:
        args.start_idx, args.end_idx = 0, 200
    if args.debug:
        args.start_idx, args.end_idx = 0, 10
    print(args)
    return args

def inference_local_api(env, history, model, num_seq=1, beam_search=False, is_greedy=False):
    state_ori = env.state
    env.reset(env.session_index, state=history)
    history_input = []
    history_input.append({"role": "user", "content": prompt})
    history_input.append({"role": "agent", "content": "Ok."})

    # one shot

    history_input.append({'role': 'user',
                          'content': 'Observation:\n"WebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Search"\n\nAvailable Actions:\n{"has_search_bar": true, "clickables": ["..."]}'})
    history_input.append({'role': 'agent',
                          'content': 'Thought:\nI think I should use the search bar to look for the product I need.\n\nAction:\nsearch[l\'eau d\'issey 6.76 fl oz bottle price < 100.00]'})
    history_input.append({'role': 'user',
                          'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] B000MJZOPK [SEP] L\'eau d\'Issey by Issey Miyake for Women 3.3 oz Eau de Toilette Spray [SEP] $49.98 [SEP] B0012S249E [SEP] L\'eau D\'issey By Issey Miyake For Women. Shower Cream 6.7-Ounces [SEP] $31.36 [SEP] B01H8PGKZS [SEP] L\'eau D\'Issey FOR MEN by Issey Miyake - 6.7 oz EDT Spray [SEP] $67.97 [SEP] B00G3C8FHE [SEP] L\'Eau d\'Issey pour Homme - Eau de Toilette 4.2 fl oz [SEP] $51.25 [SEP] B000R94HRG [SEP] Issey Miyake L\'Eau D\'Issey Pour Homme Eau De Toilette Natural Spray [SEP] $44.99 [SEP] B000C214CO [SEP] Issey Miyake L\'eau D\'issey Eau de Toilette Spray for Men, 4.2 Fl Oz [SEP] $53.99 [SEP] B0018SBRDC [SEP] Issey Miyake L\'eau d\'Issey for Women EDT, White, 0.84 Fl Oz [SEP] $27.04 [SEP] B000XEAZ9Y [SEP] L\'eau De Issey By Issey Miyake For Men. Eau De Toilette Spray 6.7 Fl Oz [SEP] $67.08 [SEP] B079HZR2RX [SEP] L\'eau d\'Issey Pure by Issey Miyake for Women 3.0 oz Nectar de Parfum Spray [SEP] $71.49"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "..."]}'})
    history_input.append({'role': 'agent',
                          'content': 'Thought:\nI think I should click on the product I need, which is B000VOHH8I.\n\nAction:\nclick[B000VOHH8I]'})
    history_input.append({'role': 'user',
                          'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "..."]}'})
    history_input.append({'role': 'agent',
                          'content': 'Thought:\nI think I should click on the \'6.76 fl oz (pack of 1)\' option to select the size I need.\n\nAction:\nclick[6.76 fl oz (pack of 1)]'})
    history_input.append({'role': 'user',
                          'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "..."]}'})
    history_input.append({'role': 'agent',
                          'content': 'Thought:\nI think I should click on the \'Buy Now\' button to purchase the product.\n\nAction:\nclick[Buy Now]'})
    previous_states, previous_actions = history
    if len(previous_states) != len(previous_actions)+1:
        previous_states_copy = previous_states[:-2]
    else:
        previous_states_copy = previous_states
    for idx, html_obs in enumerate(previous_states_copy[:-1]):
        obs = env.env.convert_html_to_text(html_obs, simple=True)

        history_input.append({'role': 'user',
                              'content': f'Observation:\n"{obs}"'})
        history_input.append({'role': 'agent',
                              'content': f'Action:\n{previous_actions[idx]}'})
    available_actions = env.env.get_available_actions()
    html_observation = env.env.observation
    observation = env.env.convert_html_to_text(html_observation, simple=True)
    history_input.append(
        {
            "role": "user",
            "content": f"Observation:\n{observation}\n\n"
                       f"Available Actions:\n{available_actions}",
        }
    )
    if is_greedy:
        assert num_seq==1
    response_output = model.inference(history=history_input, num_seq=num_seq, is_greedy=is_greedy, beam_search=beam_search)
    env.reset(env.session_index, state=state_ori)
    #print(history_input[-1]['content'])
    print(response_output)
    return response_output

def build_local_model(args):
    config = {"name": args.model_id, "temperature": args.temperature, "llm_api": args.llm_api, "api_token": args.api_token}
    model = NewLocal(**config)
    return model


def save_trajectory_history(track_list, reward_list, reward_list_gt, output_path):
    out_dict = {"tracks": [], "rewards": []}
    for state in track_list:
        out_dict["tracks"].append(track_list)
    out_dict["rewards_pred"] = reward_list
    out_dict["rewards_gt"] = reward_list_gt
    json.dump(out_dict, open(os.path.join(output_path), 'w'), indent=4)