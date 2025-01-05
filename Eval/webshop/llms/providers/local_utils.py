"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
from typing import Any
from typing import Optional

import aiolimiter
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions,
)
from data_generation.utils import config

load_dotenv()
client = Client(credentials=Credentials.from_env())

args = config()

for model in client.model.list(limit=100).results:
    print(model.model_dump(include=["name", "id"]))

if args.model=="mixtralMoe":
    model_id="mistralai/mixtral-8x7b-instruct-v01"
    #model_id="ibm-mistralai/mixtral-8x7b-instruct-v01-q"
elif args.model=="mixtral7B":
    model_id="mistralai/mistral-7b-instruct-v0-2"
elif args.model=="llama38B":
    model_id="meta-llama/llama-3-8b-instruct"
elif args.model=="llama370B":
    model_id="meta-llama/llama-3-70b-instruct"

def generate_from_local_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    stop_token: Optional[str] = None
) -> str:
    if stop_token is not None:
        for response in client.text.generation.create(
                model_id=model_id,
                inputs=[prompt],
                parameters=TextGenerationParameters(
                    max_new_tokens=max_new_tokens,
                    temperature = temperature,
                    top_p = top_p,
                    stop_sequences = [stop_token],
                    return_options=TextGenerationReturnOptions(
                        input_text=True,
                    ),
                ),
            ):
            result = response.results[0]
            answer = result.generated_text
    else:
        parameters=TextGenerationParameters(max_new_tokens=max_new_tokens,
                    temperature = temperature,
                    top_p = top_p,
                    return_options=TextGenerationReturnOptions(
                        input_text=True,
                    ),
                )
        for response in client.text.generation.create(
                model_id=model_id,
                inputs=[prompt],
                parameters = parameters,
            ):
            result = response.results[0]
            answer = result.generated_text
    return answer
