import json
import google.generativeai as genai
from prompts import FLAW_GENERATION_PROMPT_TEMPLATE, NEG_PROMPT_TEMPLATE

def generate_text_from_prompt(prompt: str, config: dict, clean_text: bool) -> str:
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
            if not clean_text:
                response = model.generate_content(prompt)
                return response.text.strip()
            else:
                response = model.generate_content(prompt)
                if response.parts:
                    raw_text = response.text.strip()
                    if raw_text.startswith("```json"):
                        raw_text = raw_text[len("```json"):].strip()
                    if raw_text.endswith("```"):
                        raw_text = raw_text[:-len("```")].strip()
                    try:
                        data = json.loads(raw_text)
                        if isinstance(data, dict):
                            if "plan" in data:
                                return data['plan']
                            elif 'degradation_ideas' in data:
                                return data['degradation_ideas']

                    except json.JSONDecodeError:
                        print(f"Failed to decode JSON for {persona_level} (Attempt {attempt+1})")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                print(f"Final attempt failed for prompt: {prompt}")
                return None
    return None

def generate_flaw(task_query_data, persona_info, positive_convo, config):
    task_query = task_query_data['task']
    
    positive_trajectory_text = ""
    if positive_convo and len(positive_convo) > 1:
        positive_trajectory_text = positive_convo[1]['value']

    if not positive_trajectory_text:
        print(f"Warning: Could not extract positive trajectory for flaw generation for persona {persona_info['level']}")
        return None

    flaw_prompt = FLAW_GENERATION_PROMPT_TEMPLATE.format(
        task_query=task_query,
        correct_plan=positive_trajectory_text,
        persona_level=persona_info['level'],
        persona_description=persona_info["description"]
    )
    generated_flaws = generate_text_from_prompt(flaw_prompt, config, clean_text=True)
    return generated_flaws

def generate_single_negative(task_query_data, persona_info, positive_convo, flaws, config):
    task_query = task_query_data['task']
    
    positive_trajectory_text = ""
    if positive_convo and len(positive_convo) > 1 and positive_convo[1]["from"] == "gpt":
        positive_trajectory_text = positive_convo[1]["value"]
    else:
        print(f"Warning: Could not extract positive trajectory for persona {persona_info['level']}")
        return None

    neg_prompt = NEG_PROMPT_TEMPLATE.format(
        task_query=task_query,
        correct_plan=positive_trajectory_text,
        persona_level=persona_info['level'],
        persona_description=persona_info["description"],
        flaws=flaws
    )
    neg_response = generate_text_from_prompt(neg_prompt, config, clean_text=True)
    
    if neg_response:
        return [
            {"from": "human", "value": task_query},
            {"from": "gpt", "value": neg_response}
        ]
        
    print(f"Failed to generate final negative trajectory for persona {persona_info['level']}")
    return None