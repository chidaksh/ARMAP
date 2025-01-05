"""This module is adapt from https://github.com/zeno-ml/zeno-build"""
from .providers.gemini_utils import generate_from_gemini_completion
#from .providers.hf_utils import generate_from_huggingface_completion
"""
from .providers.openai_utils import (
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
)
from .providers.cogagent_utils import generate_from_cogagent_completion
"""
from .providers.openai_azure_utils import generate_from_openai_azure_chat_completion
from .providers.local_utils import generate_from_local_completion
from .utils import call_llm

__all__ = [
    "generate_from_openai_completion",
    "generate_from_openai_chat_completion",
    "generate_from_huggingface_completion",
    "generate_from_gemini_completion",
    "generate_from_cogagent_completion",
    "generate_from_openai_azure_chat_completion",
    "call_llm",
]
