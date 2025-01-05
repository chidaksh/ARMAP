"""Config for language models."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LMConfig:
    """A config for a language model.

    Attributes:
        provider: The name of the API provider.
        model: The name of the model.
        model_cls: The Python class corresponding to the model, mostly for
             Hugging Face transformers.
        tokenizer_cls: The Python class corresponding to the tokenizer, mostly
            for Hugging Face transformers.
        mode: The mode of the API calls, e.g., "chat" or "generation".
    """

    provider: str
    model: str
    model_cls: type | None = None
    tokenizer_cls: type | None = None
    mode: str | None = None
    gen_config: dict[str, Any] = dataclasses.field(default_factory=dict)


def construct_llm_config(args: argparse.Namespace) -> LMConfig:
    
    # todo: fix this bug:
    print(args.stop_token)
    args.stop_token="\n" if args.stop_token=="\\n" else args.stop_token
    print(args.stop_token)
    
    llm_config = LMConfig(
        provider=args.provider, model=args.model, mode=args.mode
    )
    if args.provider in ["openai", "google", "openai_azure"]:
        llm_config.gen_config["temperature"] = args.temperature
        llm_config.gen_config["top_p"] = args.top_p
        llm_config.gen_config["context_length"] = args.context_length
        llm_config.gen_config["max_tokens"] = args.max_tokens
        llm_config.gen_config["stop_token"] = args.stop_token
        llm_config.gen_config["max_obs_length"] = args.max_obs_length
        llm_config.gen_config["max_retry"] = args.max_retry
        # for diverse generation
        if args.top_k is not None:
            llm_config.gen_config["top_k"] = args.top_k
        # for mcts decoding
        llm_config.gen_config["candidate_count"] = args.lm_candidate_count
    elif args.provider == "huggingface":
        llm_config.gen_config["temperature"] = args.temperature
        llm_config.gen_config["top_p"] = args.top_p
        llm_config.gen_config["max_new_tokens"] = args.max_tokens
        llm_config.gen_config["stop_sequences"] = (
            [args.stop_token] if args.stop_token else None
        )
        llm_config.gen_config["max_obs_length"] = args.max_obs_length
        llm_config.gen_config["model_endpoint"] = args.model_endpoint
        llm_config.gen_config["max_retry"] = args.max_retry
    elif args.provider == "local":
        llm_config.gen_config["max_new_tokens"] = args.max_tokens
        llm_config.gen_config["temperature"] = args.temperature
        llm_config.gen_config["top_p"] = args.top_p
        llm_config.gen_config["context_length"] = args.context_length
        llm_config.gen_config["max_obs_length"] = 0 
        llm_config.gen_config["stop_token"] = args.stop_token
        llm_config.gen_config["max_retry"] = args.max_retry
    else:
        raise NotImplementedError(f"provider {args.provider} not implemented")
    return llm_config
