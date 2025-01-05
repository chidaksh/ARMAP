import argparse
from typing import Any
from typing import Optional
from vertexai.preview.generative_models import Image
from PIL import Image

from llms import (
    generate_from_gemini_completion,
    #generate_from_huggingface_completion,
    #generate_from_openai_chat_completion,
    #generate_from_openai_completion,
    #generate_from_cogagent_completion,
    generate_from_openai_azure_chat_completion,
    generate_from_local_completion,
    lm_config,
)

#APIInput = str | list[Any] | dict[str, Any]
APIInput = Any

def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    extra=None,
    reward=False,
) -> str:
    response: str


    provider = lm_config.provider
    if reward:
        provider = lm_config.reward_model_type

    if provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif provider == "openai_azure":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_azure_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
            )
        else:
            raise ValueError(
                f"Azure OpenAI models do not support mode {lm_config.mode}"
            )
    elif provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif provider == "google":
        assert isinstance(prompt, list)
        #assert all(
        #    [isinstance(p, str) or isinstance(p, Image) for p in prompt]
        #)

        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
            candidate_count=lm_config.gen_config["candidate_count"],
        )
    elif provider == "cogagent":
        # assert isinstance(prompt, list)
        #assert all(
        #    [isinstance(p, str) or isinstance(p, Image) for p in prompt]
        #)

        response = generate_from_cogagent_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
            candidate_count=lm_config.gen_config["candidate_count"],
            extra=extra,
            reward=reward,
        )
    elif provider == "local":
        if lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_local_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_new_tokens=lm_config.gen_config["max_new_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
            )
        else:
            raise ValueError("Only completion is supported for ibm models\n.")
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    return response
