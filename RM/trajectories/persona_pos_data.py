import json
import google.generativeai as genai
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from prompts import POS_PROMPT_TEMPLATE, PERSONAS

def worker_generate_plan_for_persona(
    api_key: str,
    model_name: str,
    generation_config_dict: Dict,
    task_query: str,
    approach: str,
    reasoning: str,
    persona_level: str,
    persona_info: Dict[str, str]
) -> Dict[str, Any]:
    genai.configure(api_key=api_key)
    generation_config_obj = genai.GenerationConfig(**generation_config_dict)
    model = genai.GenerativeModel(model_name, generation_config=generation_config_obj)

    prompt = POS_PROMPT_TEMPLATE.format(
        task_query=task_query,
        approach=approach,
        reasoning=reasoning,
        persona_level=persona_level,
        persona_description=persona_info["description"]
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response.parts:
                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[len("```json"):].strip()
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-len("```")].strip()
            
                try:
                    data = json.loads(raw_text)
                    if isinstance(data, dict) and "plan" in data and "domain" in data:
                        return data
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON for {persona_level} (Attempt {attempt+1})")

        except Exception as e:
            print(f"Exception during LLM query for {persona_level}: {e}")

    return {}

def get_positive_trajectories(config_data, task_queries):
    all_results = [
        {
            "task_query": query['task'],
            "task_id": i,
            "approach": query['approach'],
            "reasoning": query['reasoning'],
            "plan_domains": {level: None for level in PERSONAS},
            "correct_plans": {level: None for level in PERSONAS}
        }
        for i, query in enumerate(task_queries)
    ]
    with ProcessPoolExecutor(max_workers=config_data.get("max_workers", 12)) as executor:
        future_to_info = {}
        for i, query in enumerate(task_queries):
            for level, info in PERSONAS.items():
                future = executor.submit(
                    worker_generate_plan_for_persona,
                    config_data["api_key"],
                    config_data["model_name"],
                    config_data["generation_settings"],
                    query["task"],
                    query["approach"],
                    query["reasoning"],
                    level,
                    info
                )
                future_to_info[future] = (i, level)

        print(f"--- Submitted {len(future_to_info)} jobs to the process pool. ---")
        for future in as_completed(future_to_info):
            query_index, persona_level = future_to_info[future]
            try:
                result = future.result()
                if result:
                    all_results[query_index]["plan_domains"][persona_level] = result["domain"]
                    all_results[query_index]["correct_plans"][persona_level] = result["plan"]
            except Exception as e:
                print(f"Error processing result for query {query_index}, persona {persona_level}: {e}")

    return all_results
