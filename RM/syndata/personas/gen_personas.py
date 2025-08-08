import os
import json
import random
import numpy as np
import google.generativeai as genai
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

PERSONA_PROMPTS = {
    0: "Generate a 4-5 sentences describing a company's technical infrastructure that relies entirely on heuristics and manual rules for its decision-making processes, with no machine learning models. This is a Level 0 ML maturity company. Ensure that the text contains <500 tokens.",
    1: "Generate a 4-5 sentences describing about a company's technical infrastructure that uses simple, classic machine learning models like XGBoost, logistic regression, or basic classifiers for specific tasks like churn prediction or recommendations. This is a Level 1 ML maturity company. Ensure that the text contains <500 tokens.",
    2: "Generate a 4-5 sentences describing about a company's sophisticated technical infrastructure that uses advanced machine learning systems like multi-armed bandits, reinforcement learning, or multi-objective optimization models for core business functions. This is a Level 2 ML maturity company. Ensure that the text contains <500 tokens."
}

def load_config(config_path='syndata/personas/config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

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
            return response.text.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                print(f"Final attempt failed for prompt: {prompt}")
                return None
            # time.sleep(2 ** attempt)
    return None

def generate_descriptions_for_persona(persona_level: int, num_descriptions: int, config: dict) -> list:
    """
    Generates a specified number of persona descriptions for a given level.
    """
    prompt = PERSONA_PROMPTS.get(persona_level)
    if not prompt:
        print(f"No prompt found for persona level {persona_level}")
        return []

    results = []
    # This loop is now effectively for retries within a single description generation task
    # The main generation loop is now in the ProcessPoolExecutor
    for _ in range(num_descriptions):
        description = generate_text_from_prompt(prompt, config)
        if description:
            results.append(description)
    return results

def run_generation_for_persona(args):
    persona_level, config = args
    prompt = PERSONA_PROMPTS[persona_level]
    description = generate_text_from_prompt(prompt, config)
    return persona_level, description

def analyze_text_similarity(output_dir, company_personas_data):
    """
    Analyze text similarity between all generated persona files.
    Returns similarity matrix and writes detailed analysis to file.
    """
    # 1. Set device to CUDA if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for similarity analysis.")

    # 2. Load model directly onto the specified device
    print("Loading lightweight text encoder model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    texts = []
    file_info = []
    
    print("Encoding text files...")
    for item in company_personas_data:
        file_path = os.path.join(output_dir, f"company_{item['company_id']}.txt")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            texts.append(text)
            file_info.append({
                'company_id': item['company_id'],
                'persona_label': item['persona_label'],
                'file_path': file_path,
                'text_preview': text[:100] + "..." if len(text) > 100 else text
            })
    
    # 3. Generate embeddings on the GPU, convert to tensor
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
    
    # 4. Calculate cosine similarities on the GPU
    print(f"Calculating similarities on {device}...")
    similarities = util.cos_sim(embeddings, embeddings)

    # 5. Move similarities to CPU for numpy operations and reporting
    similarities_cpu = similarities.cpu().numpy()
    
    # Generate similarity report
    similarity_report_path = os.path.join(os.path.dirname(output_dir), "persona_similarity_analysis.txt")
    
    with open(similarity_report_path, 'w', encoding='utf-8') as f:
        f.write("PERSONA TEXT SIMILARITY ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files analyzed: {len(file_info)}\n")
        f.write(f"Model used: all-MiniLM-L6-v2\n")
        f.write(f"Similarity metric: Cosine similarity\n\n")
        
        # Overall statistics
        upper_triangle = similarities_cpu[np.triu_indices_from(similarities_cpu, k=1)]
        f.write(f"Overall similarity statistics:\n")
        f.write(f"  Mean similarity: {np.mean(upper_triangle):.4f}\n")
        f.write(f"  Std similarity: {np.std(upper_triangle):.4f}\n")
        f.write(f"  Min similarity: {np.min(upper_triangle):.4f}\n")
        f.write(f"  Max similarity: {np.max(upper_triangle):.4f}\n\n")
        
        # Per-persona-level statistics
        f.write("Similarity by persona level:\n")
        for level in [0, 1, 2]:
            level_indices = [i for i, info in enumerate(file_info) if info['persona_label'] == level]
            if len(level_indices) > 1:
                level_similarities = []
                for i in range(len(level_indices)):
                    for j in range(i+1, len(level_indices)):
                        idx1, idx2 = level_indices[i], level_indices[j]
                        level_similarities.append(similarities_cpu[idx1][idx2])
                
                if level_similarities:
                    f.write(f"  Level {level}: Mean={np.mean(level_similarities):.4f}, "
                           f"Std={np.std(level_similarities):.4f}, "
                           f"Count={len(level_similarities)} pairs\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("TOP-10 MOST SIMILAR FILES FOR EACH FILE\n")
        f.write("=" * 50 + "\n\n")
        
        # For each file, find top-10 most similar files
        for i, info in enumerate(file_info):
            f.write(f"Company {info['company_id']} (Persona Level {info['persona_label']}):\n")
            f.write(f"Text preview: {info['text_preview']}\n")
            f.write("Top 10 most similar files:\n")
            
            # Get similarity scores for this file (excluding self)
            sim_scores = [(j, similarities_cpu[i][j]) for j in range(len(file_info)) if i != j]
            sim_scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (j, score) in enumerate(sim_scores[:10], 1):
                similar_info = file_info[j]
                f.write(f"  {rank:2d}. Company {similar_info['company_id']} "
                       f"(Level {similar_info['persona_label']}) - "
                       f"Similarity: {score:.4f}\n")
                f.write(f"      Preview: {similar_info['text_preview']}\n")
            
            f.write("\n" + "-" * 40 + "\n\n")
    
    print(f"Similarity analysis saved to: {similarity_report_path}")
    return None

def main():
    config = load_config()

    total_descriptions = config["total_descriptions"]
    num_personas = len(PERSONA_PROMPTS)
    descriptions_per_persona = total_descriptions // num_personas
    remainder = total_descriptions % num_personas

    output_dir = config["output_dir"]
    json_output_path = config["json_output_path"]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Cleaning up existing files in {output_dir}...")
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    tasks = []
    persona_levels = list(PERSONA_PROMPTS.keys())
    persona_counts = {level: descriptions_per_persona for level in persona_levels}
    for _ in range(remainder):
        chosen_level = random.choice(persona_levels)
        persona_counts[chosen_level] += 1

    print(f"Generating a total of {total_descriptions} descriptions with the following distribution:")
    for level, count in persona_counts.items():
        print(f"  - Persona Level {level}: {count} descriptions")
        for _ in range(count):
            tasks.append((level, config))

    all_results = {level: [] for level in persona_levels}
    with ProcessPoolExecutor(max_workers=config["max_workers"]) as executor:
        future_to_persona = {executor.submit(run_generation_for_persona, task): i for i, task in enumerate(tasks)}
        
        for future in tqdm(as_completed(future_to_persona), total=len(tasks), desc="Generating Persona Descriptions"):
            try:
                level, result = future.result()
                if result:
                    all_results[level].append(result)
            except Exception as exc:
                print(f'A generation task failed with an exception: {exc}')

    print("\nPersona description generation complete.")

    company_personas_data = []
    company_counter = 1
    
    file_paths = []
    for persona_level, descriptions in sorted(all_results.items()):
        for desc in descriptions:
            file_name = f"company_{company_counter}.txt"
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(desc)

            relative_path = os.path.join(*output_dir.split(os.sep)[1:], file_name)

            company_personas_data.append({
                "company_id": company_counter,
                "knowledge_file_path": relative_path,
                "persona_label": persona_level
            })
            file_paths.append(file_path)
            company_counter += 1

    analyze_text_similarity(output_dir, company_personas_data)
    # print("Similarities between generated persona files:")
    # print(similarities)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(company_personas_data, f, indent=2)

    print(f"\nData generation complete. {company_counter - 1} text files created.")
    print(f"'{json_output_path}' has been updated.")

if __name__ == "__main__":
    main()