import json
import pandas as pd
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from persona_pos_data import get_positive_trajectories
from persona_neg_data import get_negative_trajectories

def load_task_queries(file_path: str) -> List[str]:
    df = pd.read_csv(file_path)
    df = df[['Task Query', 'Could be approach', 'How will it help in answering the task?']].rename(columns={'Task Query': 'task', 'Could be approach': 'approach', 'How will it help in answering the task?': 'reasoning'})
    df_ = df[:1]
    return df_.to_dict('records')

def load_config(config_path: str = "./plan_config.json") -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    task_queries_file = "TS_Sim_Intern_Golden_Tasks_llm-generated-tasks.csv"
    task_queries = load_task_queries(task_queries_file)
    config_data = load_config()
    pos_results = get_positive_trajectories(config_data, task_queries)
    all_results = get_negative_trajectories(config_data, pos_results)

    processed_conversations = []
    for result_item in all_results:
        task_query = result_item["task_query"]
        domains = result_item["plan_domains"]
        negative_plans = result_item["negative_plans"]
        flaws = result_item['flaws']
        for persona_level, plans in result_item["correct_plans"].items():
            if plans:
                plan_text = plans
                conversation = {
                    "persona_id": persona_level,
                    "domain": domains[persona_level],
                    "pos_conversation": {
                        "human": task_query,
                        "agent": plan_text
                    },
                    "flaws": flaws,
                    "neg_conversation": {
                        "human": task_query,
                        "agent": negative_plans[persona_level]
                    }
                }
                processed_conversations.append(conversation)

    output_file = config_data.get("output_file_name", "generated_pos_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_conversations, f, indent=4)