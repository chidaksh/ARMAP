import json
import google.generativeai as genai
from prompts import POS_PROMPT_TEMPLATE

def generate_text_from_prompt(prompt: str, config: dict) -> str:
    api_key = config.get("api_key")
    if not api_key:
        print("API key not found in config. Skipping LLM call.")
        return None
        
    genai.configure(api_key=api_key)
    generation_config_obj = genai.GenerationConfig(**config.get("generation_settings", {}))
    model = genai.GenerativeModel(config["model_name"], generation_config=generation_config_obj)

    max_retries = config['max_retries']
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
                    if isinstance(data, dict) and "plan" in data:
                        return data['plan']
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON for {persona_info['level']} (Attempt {attempt+1})")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                print(f"Final attempt failed for prompt: {prompt}")
                return None
    return None

def generate_single_positive(task_query_data, persona_info, config):
    task_query = task_query_data["task"]
    approach = task_query_data["approach"]
    reasoning = task_query_data["reasoning"]
    
    prompt = POS_PROMPT_TEMPLATE.format(
        task_query=task_query,
        approach=approach,
        reasoning=reasoning,
        persona_level=persona_info['level'],
        persona_description=persona_info["description"]
    )
    
    gpt_response = generate_text_from_prompt(prompt, config)
    if gpt_response:
        return [
            {"from": "human", "value": task_query},
            {"from": "gpt", "value": gpt_response}
        ]
    return None
