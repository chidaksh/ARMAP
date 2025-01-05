import openai
import logging
import backoff

from .base import LMAgent

logger = logging.getLogger("agent_frame")


class OpenAILMAgent(LMAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        if "api_base" in config:
            openai.api_base = config['api_base']
        if "api_key" in config:
            openai.api_key = config['api_key']

    def __call__(self, messages) -> str:
        # Prepend the prompt with the system message
        response = openai.ChatCompletion.create(
            model=self.config["model_name"],
            messages=messages,
            max_tokens=self.config.get("max_tokens", 512),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
            top_p=self.config.get("top_p", 1),
        )
        return response.choices[0].message["content"]
