PERSONAS = {
    "Level-0": {
        "name": "Heuristic-Driven",
        "description": "This company operates without any machine learning models. Decisions are based on business rules, heuristics, and basic analytics (e.g., counts, averages). They have access to raw data but lack the infrastructure or expertise for predictive modeling."
    },
    "Level-1": {
        "name": "Simple-ML",
        "description": "This company has a small data science team and uses basic machine learning models, such as logistic regression, gradient boosting (like XGBoost or LightGBM), or random forests. They can build and deploy simple classifiers or regression models but do not have experience with complex systems."
    },
    "Level-2": {
        "name": "Advanced-ML",
        "description": "This company is a leader in machine learning, similar to tech giants like Spotify, Netflix, or TikTok. They have a mature MLOps infrastructure and employ sophisticated techniques, including multi-armed bandits, deep learning-based recommender systems and multi-objective optimization."
    }
}

POS_PROMPT_TEMPLATE = """
You are an expert strategy consultant providing a correct, actionable plan to address the following ML-related task query.
- Task Query: {task_query}

Company's ML Maturity Profile:
- ML Maturity Level: {persona_level}
- Description: {persona_description}

Below is a possible approach and how this approach can be useful in answering the query. Note that this approach is designed for a company with ML maturity level 2.
- Approach: {approach}
- Reasoning: {reasoning}

Your Task:
Based on the company's profile and their limited capabilities, generate a single, concrete, and correct plan to address the query. The plan should be realistic for them to implement. Do not mention about company's ML maturity in the response, only use it as a reference while crafting the response. Although the possible approach is designed for a company with ML maturity level 2, your final plan should be designed for the company's ML maturity level.  

Output Format:
Return the output as a valid JSON object shown below:

{{
    "domain": "<ML Domain label for the plan generated>",
    "plan": "<Your plan here>"
}}
"""

FLAW_GENERATION_PROMPT_TEMPLATE = """
You are an expert red-teamer tasked with identifying subtle weaknesses in ML project plans. Your goal is to generate flaws that can be injected into the correct plan for the following ML-related task query.
- Task Query: {task_query}

Company's ML Maturity Profile:
- ML Maturity Level: {persona_level}
- Description: {persona_description}

Correct Plan: {correct_plan}

Your Task:
Generate 1-2 subtle flaws that can be injected into the provided plan to make it less preferred while still being appropriate for the company's ML maturity level. The steps should be short (1-2 sentences) and concise.

Ensure:
- The flaws are subtle and not factually wrong
- A modified plan with these flaws injected into the correct plan, is still appropriate for the company's ML maturity level.

Output Format:
Return only a numbered list of the flaws.
[1. <Flaw 1>, 2. <Flaw 2>]
"""

NEG_PROMPT_TEMPLATE = """
You are an expert in identifying common pitfalls in ML project planning. Your goal is to take a correct plan and generate a plan that is less preferred for the following ML-related task query.
- Task Query: {task_query}

Company's ML Maturity Profile:
- ML Maturity Level: {persona_level}
- Description: {persona_description}

Correct Plan: {correct_plan}

Inject the following flaws in the plan:
{flaws}

Your Task:
Based on the company's profile, provided plan, and the flaws to be introduced, generate a new negative plan that is less preferred but appropriate for their maturity level. Do not mention the company's ML maturity in the response.

Output Format:
Return the output as a valid JSON object shown below:

{{
    "plan": "<Your new, subtly flawed negative plan here>"
}} 
"""