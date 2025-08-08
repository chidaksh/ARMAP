import json
import random
import argparse
from typing import List, Dict

def reformat_data_for_reward_modeling(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        all_company_data = json.load(f)

    all_preference_pairs = []
    all_personas = {"Level-0", "Level-1", "Level-2"}
    label_to_id = {"Level-0": 0, "Level-1": 1, "Level-2": 2}

    for company_entry in all_company_data:
        company_id = company_entry["company_id"]
        true_persona_id = company_entry["true_persona_id"]

        conversation_map = {
            convo["persona_id"]: {
                "pos": convo["pos_conversation"],
                "neg": convo["neg_conversation"]
            }
            for convo in company_entry["conversations"]
        }
        for persona_id in all_personas:
            if persona_id in conversation_map:
                pair = {
                    "company_id": company_id,
                    "true_persona_id": true_persona_id,
                    "chosen": conversation_map[persona_id]["pos"],
                    "rejected": conversation_map[persona_id]["neg"]
                }
                all_preference_pairs.append(pair)

        other_personas = all_personas - {true_persona_id}
        pos_true = conversation_map[true_persona_id]["pos"]
        
        for other_persona in other_personas:
            if other_persona in conversation_map:
                pos_other = conversation_map[other_persona]["pos"]
                pair = {
                    "company_id": company_id,
                    "true_persona_id": true_persona_id,
                    "chosen": pos_true,
                    "rejected": pos_other
                }
                all_preference_pairs.append(pair)

    random.shuffle(all_preference_pairs)
    
    final_output_data = []
    num_samples = len(all_preference_pairs)
    num_preference_1 = num_samples // 2

    for i, pair in enumerate(all_preference_pairs):
        human_query = pair["chosen"][0]["value"]
        chosen_agent_response = [pair["chosen"][1]]
        rejected_agent_response = [pair["rejected"][1]]
        if i < num_preference_1:
            output_1 = chosen_agent_response
            output_2 = rejected_agent_response
            preference = 1
        else:
            output_1 = rejected_agent_response
            output_2 = chosen_agent_response
            preference = 2
    
        if num_samples % 2 != 0 and i == num_samples - 1:
            preference = random.choice([1, 2])

        formatted_sample = {
            "id": i,
            "company_id": int(pair['company_id'].split('_')[1]),
            "persona_id": label_to_id[pair["true_persona_id"]],
            "image": "not_used.png",
            "conversations": [
                {"from": "human", "value": human_query},
                {"from": "gpt", "value": "###test gpt###"}
            ],
            "output_1": output_1,
            "output_2": output_2,
            "preference": preference
        }
        final_output_data.append(formatted_sample)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_data, f, indent=2)

    print(f"Successfully reformatted data. {len(final_output_data)} preference samples saved to '{output_path}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reformat generated data for Reward Model training.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input generated_pos_data.json file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output reformatted JSON file.")
    args = parser.parse_args()
    
    reformat_data_for_reward_modeling(args.input_file, args.output_file)