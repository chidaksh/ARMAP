import json
import google.generativeai as genai
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from prompts import FLAW_GENERATION_PROMPT_TEMPLATE, NEG_PROMPT_TEMPLATE, PERSONAS

def worker_generate_negative_plan(
    api_key: str,
    model_name: str,
    generation_config_dict: Dict,
    task_query: str,
    correct_plan: str,
    persona_level: str,
    persona_info: Dict[str, str]
) -> Dict[str, Any]:
    genai.configure(api_key=api_key)
    generation_config_obj = genai.GenerationConfig(**generation_config_dict)
    model = genai.GenerativeModel(model_name, generation_config=generation_config_obj)

    flaw_prompt = FLAW_GENERATION_PROMPT_TEMPLATE.format(
        task_query=task_query,
        correct_plan=correct_plan,
        persona_level=persona_level,
        persona_description=persona_info["description"]
    )
    max_retries = 3
    flaws = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(flaw_prompt)
            if response.parts:
                flaws = response.text.strip()
                break
        except Exception as e:
            print(f"Exception during LLM query for {persona_level}: {e}")

    if flaws is None:
        print(f"Failed to generate flaw instructions for {persona_level} after {max_retries} attempts.")
        return {}

    neg_plan_prompt = NEG_PROMPT_TEMPLATE.format(
        task_query=task_query,
        correct_plan=correct_plan,
        persona_level=persona_level,
        persona_description=persona_info["description"],
        flaws=flaws
    )
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(neg_plan_prompt)
            if response.parts:
                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[len("```json"):].strip()
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-len("```")].strip()
                try:
                    data = json.loads(raw_text)
                    if isinstance(data, dict) and "plan" in data:
                        data['flaws'] = flaws
                        return data
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON for {persona_level} (Attempt {attempt+1})")

        except Exception as e:
            print(f"Exception during LLM query for {persona_level}: {e}")
    
    return {}

def get_negative_trajectories(config_data, all_results):
    for data in all_results:
        data["negative_plans"] = {level: None for level in PERSONAS}
        data["flaws"] = {level: None for level in PERSONAS}
    with ProcessPoolExecutor() as executor:
        future_to_info = {}
        for i, data in enumerate(all_results):
            task_query = data["task_query"]
            for level, pos_plan in data["correct_plans"].items():
                if pos_plan:
                    future = executor.submit(
                        worker_generate_negative_plan,
                        config_data["api_key"],
                        config_data["model_name"],
                        config_data["generation_settings"],
                        task_query,
                        pos_plan,
                        level,
                        PERSONAS[level]
                    )
                    future_to_info[future] = (i, level)

        print(f"--- Submitted {len(future_to_info)} jobs to the process pool. ---")
        for future in as_completed(future_to_info):
            query_index, persona_level = future_to_info[future]
            try:
                result = future.result()
                if result:
                    all_results[query_index]["negative_plans"][persona_level] = result["plan"]
                    all_results[query_index]["flaws"][persona_level] = result["flaws"]
            except Exception as e:
                print(f"Error processing result for query {query_index}, persona {persona_level}: {e}")
    return all_results
