import argparse
from collections import defaultdict
from decimal import Decimal
import json
import os
import random
from tqdm import tqdm
import re
import pdb
import sys 

goal_attributes = ['asin', 'category', 'query', 'name', 'product_category', 'instruction_text', 'attributes', 'price_upper', 'goal_options', 'weight']
goal_prompt_attributes = ['category', 'query', 'name', 'product_category', 'attributes', 'price_upper', 'goal_options']
goal_attributes_v2 = ['asin', 'instruction', 'attributes', 'options', 'instruction_attributes', 'instruction_options', 'assignment_id', 'worker_id']
goal_prompt_attributes_v2 = ['attributes', 'options', 'instruction_attributes', 'instruction_options']
product_attributes = ['name', 'full_description', 'pricing', 'images', 'product_category', 'average_rating', 'small_description', 'model', 'customization_options', 'asin', 'category', 'query', 'page', 'Title', 'Description', 'Reviews', 'Rating', 'BulletPoints', 'Price', 'options', 'option_to_image', 'Attributes', 'MainImage']
product_prompt_attributes = ['name', 'product_category', 'category', 'query', 'Title', 'Description']

def set_debugger():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="configuration to data samples"
    )
    
    # lm config
    parser.add_argument("--provider", type=str, default="local")
    parser.add_argument("--model", type=str, default="llama370B")
    parser.add_argument("--mode", type=str, default="completion")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=float, default=None)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=3840,
    )
    parser.add_argument("--lm_candidate_count", type=int, default=1)
    parser.add_argument("--text_only", type=int, default=1)

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)
    parser.add_argument("--step_id", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="output/webshop_synthesis_v2")
    parser.add_argument("--sample_product_num", type=int, default=10000)
    parser.add_argument("--original_data_path", type=str, default="data_generation/data_original/items_shuffle.json")
    parser.add_argument("--original_data_dir", type=str, default="data_generation/data_original")
    parser.add_argument("--test_goal_path", type=str, default="output/webshop_synthesis_v1/test_goal.json")
    parser.add_argument("--stage2_output_dir", type=str, default="output/webshop_synthesis_v2/raw_responses")
    parser.add_argument("--stage3_output_path", type=str, default="output/webshop_synthesis_v2/syn_ins_v1.json")
    parser.add_argument("--stage4_output_path", type=str, default="output/webshop_synthesis_v2/syn_instruction_triplet.json")
    parser.add_argument("--stage4_input_track_path1", type=str, default="/home/zfchen/zfchen/AgentBenchDev/outputs/llama370B_v1/Local/webshop-train/runs.jsonl")
    parser.add_argument("--stage4_input_track_path2", type=str, default="/home/zfchen/zfchen/AgentBenchDev/outputs/llama38B_v1/Local/webshop-train/runs.jsonl")
    
    # prompt config
    parser.add_argument("--prompt_shot", type=int, default=10)
    parser.add_argument("--stage2_prompt_path", type=str, default="output/webshop_prompts/stage2/simple_prompt_10shot_v2.txt")
    
    args = parser.parse_args()
    return args


def build_prompt_v1():
    args = config()
    prompt_dir = os.path.dirname(args.stage2_prompt_path)
    if not os.path.isdir(prompt_dir):
        os.makedirs(prompt_dir)
    smp_goal_path = os.path.join(args.output_dir, "train_sample_goal.json")
    with open(smp_goal_path, "r") as fh:
        smp_goal_list = json.load(fh)
    prompt_str = "You are an autonomous intelligent agent tasked with proposing a search goal for a shopping website based on the target product's attribute and description.\nHere are a few examples:\n"
    
    product_starter = "Product Detials:\n"
    goal_prefix = "Predicted goal for shopping"
    for idx, ins_info in enumerate(smp_goal_list):
        if idx >=args.prompt_shot:
            break
        prompt_str +=product_starter
        for attr in goal_prompt_attributes:
            tmp_str = "%s: %s\n"%(attr, str(ins_info[attr]))
            prompt_str +=tmp_str 
        tmp_str = "%s: %s\n\n"%(goal_prefix, ins_info["instruction_text"])
        prompt_str +=tmp_str
    mid_str = "Now make a new prediction based on the below attributes of a product.\nProdut Detials:\n"
    prompt_str +=mid_str
    prompt_str +="__PRODUCT_INFO__\n"
    prompt_str +=goal_prefix+":"
    print(prompt_str)
    text_fh = open(args.stage2_prompt_path, "w")
    text_fh.write(prompt_str)

# copy from webshop environment
def load_products(args, num_products=None, human_goals=True):
    
    BASE_DIR = args.original_data_dir
    
    HUMAN_ATTR_PATH = os.path.join(BASE_DIR, 'items_human_ins.json')
    DEFAULT_FILE_PATH = os.path.join(BASE_DIR, 'items_shuffle.json')
    DEFAULT_ATTR_PATH = os.path.join(BASE_DIR, 'items_ins_v2.json')
    
    # TODO: move to preprocessing step -> enforce single source of truth
    with open(DEFAULT_FILE_PATH) as f:
        products = json.load(f)
    print('Products loaded.')
    products = clean_product_keys(products)
   
    import pdb
    pdb.set_trace()
    
    all_reviews = dict()
    all_ratings = dict()

    if human_goals:
        with open(HUMAN_ATTR_PATH) as f:
            human_attributes = json.load(f)
    with open(DEFAULT_ATTR_PATH) as f:
        attributes = json.load(f)
    print('Attributes loaded.')

    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    if num_products is not None:
        # using item_shuffle.json, we assume products already shuffled
        products = products[:num_products]
    for i, p in tqdm(enumerate(products), total=len(products)):
        asin = p['asin']
        if asin == 'nan' or len(asin) > 10:
            continue

        if asin in asins:
            continue
        else:
            asins.add(asin)

        products[i]['category'] = p['category']
        products[i]['query'] = p['query']
        products[i]['product_category'] = p['product_category']

        products[i]['Title'] = p['name']
        products[i]['Description'] = p['full_description']
        products[i]['Reviews'] = all_reviews.get(asin, [])
        products[i]['Rating'] = all_ratings.get(asin, 'N.A.')
        for r in products[i]['Reviews']:
            if 'score' not in r:
                r['score'] = r.pop('stars')
            if 'review' not in r:
                r['body'] = ''
            else:
                r['body'] = r.pop('review')
        products[i]['BulletPoints'] = p['small_description'] \
            if isinstance(p['small_description'], list) else [p['small_description']]

        pricing = p.get('pricing')
        if pricing is None or not pricing:
            pricing = [100.0]
            price_tag = '$100.0'
        else:
            pricing = [
                float(Decimal(re.sub(r'[^\d.]', '', price)))
                for price in pricing.split('$')[1:]
            ]
            if len(pricing) == 1:
                price_tag = f"${pricing[0]}"
            else:
                price_tag = f"${pricing[0]} to ${pricing[1]}"
                pricing = pricing[:2]
        products[i]['pricing'] = pricing
        products[i]['Price'] = price_tag

        options = dict()
        customization_options = p['customization_options']
        option_to_image = dict()
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_name = option_name.lower()

                option_values = []
                for option_content in option_contents:
                    option_value = option_content['value'].strip().replace('/', ' | ').lower()
                    option_image = option_content.get('image', None)

                    option_values.append(option_value)
                    option_to_image[option_value] = option_image
                options[option_name] = option_values
        products[i]['options'] = options
        products[i]['option_to_image'] = option_to_image

        # without color, size, price, availability
        if asin in attributes and 'attributes' in attributes[asin]:
            products[i]['Attributes'] = attributes[asin]['attributes']
        else:
            products[i]['Attributes'] = ['DUMMY_ATTR']
            
        if human_goals:
            if asin in human_attributes:
                products[i]['instructions'] = human_attributes[asin]
        else:
            products[i]['instruction_text'] = \
                attributes[asin].get('instruction', None)

            products[i]['instruction_attributes'] = \
                attributes[asin].get('instruction_attributes', None)

        products[i]['MainImage'] = p['images'][0]
        products[i]['query'] = p['query'].lower().strip()

        all_products.append(products[i])

    for p in all_products:
        for a in p['Attributes']:
            attribute_to_asins[a].add(p['asin'])

    product_item_dict = {p['asin']: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    return all_products, product_item_dict, product_prices, attribute_to_asins


def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product['asin']
        pricing = product['pricing']
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[asin] = price
    return product_prices


def clean_product_keys(products):
    for product in products:
        product.pop('product_information', None)
        product.pop('brand', None)
        product.pop('brand_url', None)
        product.pop('list_price', None)
        product.pop('availability_quantity', None)
        product.pop('availability_status', None)
        product.pop('total_reviews', None)
        product.pop('total_answered_questions', None)
        product.pop('seller_id', None)
        product.pop('seller_name', None)
        product.pop('fulfilled_by_amazon', None)
        product.pop('fast_track_message', None)
        product.pop('aplus_present', None)
        product.pop('small_description_old', None)
    print('Keys cleaned.')
    return products

def build_prompt_v2():
    args = config()
    prompt_dir = os.path.dirname(args.stage2_prompt_path)
    if not os.path.isdir(prompt_dir):
        os.makedirs(prompt_dir)
    smp_prod_path = os.path.join(args.output_dir, "step1_sample_10000.json")
    with open(smp_prod_path, "r") as fh:
        smp_info_dict = json.load(fh)
    prompt_str = "You are an autonomous intelligent agent tasked with proposing a search goal for a shopping website based on the target product's attribute and description.\nHere are a few examples:\n"
    
    product_starter = "Product Detials:\n"
    goal_prefix = "Predicted goal for shopping"
    
    for idx, ins_info in enumerate(smp_info_dict["train_instructions"]):
        if idx >=args.prompt_shot:
            break
        prompt_str +=product_starter
        for attr in product_prompt_attributes:
            tmp_str = "%s: %s\n"%(attr, str(ins_info[attr]))
            prompt_str +=tmp_str 
        for attr_v2 in goal_prompt_attributes_v2:
            tmp_str = "%s: %s\n"%(attr_v2, str(ins_info["instructions"][0][attr_v2]))
            prompt_str +=tmp_str 
        tmp_str = "%s: %s\n\n"%(goal_prefix, ins_info["instructions"][0]["instruction"])
        prompt_str +=tmp_str
    mid_str = "Now make a new prediction based on the below attributes of a product.\nProdut Detials:\n"
    prompt_str +=mid_str
    prompt_str +="__PRODUCT_INFO__\n"
    prompt_str +=goal_prefix+":"
    print(prompt_str)
    text_fh = open(args.stage2_prompt_path, "w")
    text_fh.write(prompt_str)

def analyze_eval():
    result_json = "/home/zfchen/zfchen/AgentBenchDev/outputs/webshop_dev_0/Local/webshop-dev/runs.jsonl"
    with open(result_json, "r") as fh:
        task_info_list1 = list(fh)
    for idx, task_info in enumerate(task_info_list1):
        task_info = json.loads(task_info)
        import pdb
        pdb.set_trace()
       
def analyze_data():
    #result_json = "/home/zfchen/zfchen/AgentBenchDev/outputs/llama370B_v1_10k_fix/Local/webshop-train/runs.jsonl"
    #result_json = "/home/zfchen/zfchen/AgentBenchDev/outputs/mixtral7B_v1_10k_fix/Local/webshop-train/runs.jsonl"
    result_json = "/home/zfchen/zfchen/AgentBenchDev/outputs/mixtral7B_v1_10k_fix/Local/webshop-train/runs.jsonl"
    data = []
    count = 0
    with open(result_json, "r") as fh:
        for line in fh:
            #if count>=2783:
            #    continue
            tmp = json.loads(line)
            print(tmp["index"])
            print(count)
            count +=1
    import pdb
    pdb.set_trace()
        
if __name__=="__main__":
    #build_prompt_v2()
    #analyze_eval()
    analyze_data()
