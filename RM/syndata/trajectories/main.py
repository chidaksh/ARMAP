import json
import pandas as pd
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from prompts import PERSONAS
from persona_pos_data import generate_single_positive
from persona_neg_data import generate_flaw, generate_single_negative

def load_task_queries(file_path: str) -> List[str]:
    df = pd.read_csv(file_path)
    df = df[['Task Query', 'Could be approach', 'How will it help in answering the task?']].rename(columns={'Task Query': 'task', 'Could be approach': 'approach', 'How will it help in answering the task?': 'reasoning'})
    # df_ = df[:1]
    return df.to_dict('records')

def load_config(config_path: str = "syndata/trajectories/plan_config.json") -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def main():
    task_queries_file = "syndata/trajectories/TS_Sim_Intern_Golden_Tasks_llm-generated-tasks.csv"
    task_queries = load_task_queries(task_queries_file)
    config = load_config()
    all_results = {}
    tasks_to_process = []
    
    persona_info_dict = {}
    for persona_id in ["Level-0", "Level-1", "Level-2"]:
        persona_dict = {}
        persona_dict['level'] = persona_id
        persona_dict['description'] = PERSONAS[persona_id]
        persona_info_dict[persona_id] = persona_dict
        
    for i, task_data in enumerate(task_queries):
        company_id = f"company_{i+1}"
        true_persona_id = f"Level-{i % 3}"
        all_results[company_id] = {
            "company_id": company_id,
            "true_persona_id": true_persona_id,
            "conversations": []
        }
        for persona_id in ["Level-0", "Level-1", "Level-2"]:
            tasks_to_process.append(
                ("GENERATE_POSITIVE", task_data, company_id, persona_info_dict[persona_id], config, None, None)
            )

    with ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
        future_to_task = {
            executor.submit(worker_function, task): task for task in tasks_to_process
        }
        with tqdm(total=len(task_queries) * 9, desc="Processing Tasks") as pbar:
            while future_to_task:
                done_future = next(as_completed(future_to_task))
                original_task = future_to_task.pop(done_future)
                try:
                    result = done_future.result()
                    if result:
                        task_type, company_id, persona_info, data = result
                        
                        if task_type == "POSITIVE_RESULT":
                            all_results[company_id].setdefault("temp_pos", {})[persona_info['level']] = data
                            new_flaw_task = ("GENERATE_FLAW", original_task[1], company_id, persona_info, config, data, None)
                            future = executor.submit(worker_function, new_flaw_task)
                            future_to_task[future] = new_flaw_task
                        
                        elif task_type == "FLAW_RESULT":
                            all_results[company_id].setdefault("temp_flaw", {})[persona_info['level']] = data
                            pos_convo = all_results[company_id]["temp_pos"][persona_info['level']]
                            new_neg_task = ("GENERATE_NEGATIVE", original_task[1], company_id, persona_info, config, pos_convo, data)
                            future = executor.submit(worker_function, new_neg_task)
                            future_to_task[future] = new_neg_task
                            
                        elif task_type == "NEGATIVE_RESULT":
                            pos_convo = all_results[company_id]["temp_pos"][persona_info['level']]
                            flaws = all_results[company_id]["temp_flaw"][persona_info['level']]
                            final_convo_entry = {
                                "persona_id": persona_info['level'],
                                "pos_conversation": pos_convo,
                                "injected_flaws": flaws,
                                "neg_conversation": data
                            }
                            all_results[company_id]["conversations"].append(final_convo_entry)
                except Exception as e:
                    print(f"A task failed with an exception: {e}")
                pbar.update(1)

    for res in all_results.values():
        if "temp_pos" in res:
            del res["temp_pos"]
        if "temp_flaw" in res:
            del res['temp_flaw']
        # res["conversations"].sort(key=lambda x: x["persona_id"])

    final_output = [res for res in all_results.values() if len(res["conversations"]) == 3]
    with open(config["output_file_name"], 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\nTrajectory generation complete. {len(final_output)} successful results saved to {config['output_file_name']}")

def worker_function(task):
    task_type, task_data, company_id, persona_info, config, positive_convo, flaw = task
    
    if task_type == "GENERATE_POSITIVE":
        result = generate_single_positive(task_data, persona_info, config)
        if result:
            return "POSITIVE_RESULT", company_id, persona_info, result
    
    elif task_type == "GENERATE_FLAW":
        result = generate_flaw(task_data, persona_info, positive_convo, config)
        if result:
            return "FLAW_RESULT", company_id, persona_info, result
    
    elif task_type == "GENERATE_NEGATIVE":
        result = generate_single_negative(task_data, persona_info, positive_convo, flaw, config)
        if result:
            return "NEGATIVE_RESULT", company_id, persona_info, result

    return None

if __name__ == "__main__":
    main()