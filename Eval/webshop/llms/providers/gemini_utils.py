"""Tools to generate from Gemini prompts."""

import random
import time
from typing import Any
from typing import Optional

from google.api_core.exceptions import InvalidArgument
from vertexai.preview.generative_models import (
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
)
from data_generation.utils import config
import os

"""
Use genai to call models for simplicity
"""
args = config()


# if args.policy_model_type=="gemini":
import  google.generativeai as genai
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

"""
import pprint
for model in genai.list_models():
    pprint.pprint(model)
import pdb
pdb.set_trace()
"""

if args.text_only:
    model = genai.GenerativeModel("gemini-pro")
else:
    model = genai.GenerativeModel("gemini-pro-vision")

def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 1,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (InvalidArgument,),
):
    #max_retries: int = 10,
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:

                print(e)

                # Increment retries
                num_retries += 1
                print("Number retries: %d/%d\n"%(num_retries, max_retries))

                # Check if max retries has been reached
                if num_retries > max_retries:
                    print("Unexpected errors from APIs, type 1\n")
                    print("Number retries: %d/%d\n"%(num_retries, max_retries))
                    if kwargs['candidate_count']==1:
                        return ""
                    return ["" for idx in range(kwargs['candidate_count'])]
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:

                print(e)
                # Increment retries
                num_retries += 1
                print("Unexpected errors from APIs, type 2\n")
                print("Number retries: %d/%d\n"%(num_retries, max_retries))
                #import pdb
                #pdb.set_trace()
                # Check if max retries has been reached
                if num_retries > max_retries:
                    if kwargs['candidate_count']==1:
                        return ""
                    return ["" for idx in range(kwargs['candidate_count'])]
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                # Sleep for the delay
                time.sleep(delay)
    return wrapper


@retry_with_exponential_backoff
def generate_from_gemini_completion(
    prompt: list,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    candidate_count = 1,
    top_k = None
) -> str:
    del engine
    safety_config = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    # sample multiple paths candidate_count times since gemini only supports 1 sequence a time
    if top_k is None:
        generation_config=dict(
            candidate_count=candidate_count,
            max_output_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature)
    else:
        generation_config=dict(
            candidate_count=candidate_count,
            max_output_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature)
    if candidate_count==1:
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        try:
            answer = response.text
        except:
            answer = ""
        return answer
    else:
        answer_list = []
        for smp_idx in range(candidate_count):
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            try:
                answer = response.text
            except:
                answer = ""
            answer_list.append(answer)
        #import pdb;pdb.set_trace()
        # print(answer_list)
        return answer_list


@retry_with_exponential_backoff
# debug only
def fake_generate_from_gemini_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: Optional[str] = None
) -> str:
    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer
