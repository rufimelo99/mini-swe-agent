import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import transformers
import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig

logger = logging.getLogger("local_model")

class HuggingFaceModel(LitellmModel):
    config: LitellmModelConfig

    def __init__(self, **kwargs):
        self.config = LitellmModelConfig(
            **kwargs
        )
        self.cost = 0.0
        self.n_calls = 0
        import transformers
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_name)
        logger.info(f"Initialized local DeepSeekModel with model {self.config.model_name}")

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        logger.info("Querying model with messages: %s", messages)
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant: "
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=kwargs.get("max_tokens", 256))
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_content = response.split("assistant:")[-1].strip()
        return {"choices": [{"message": {"role": "assistant", "content": response_content}}]}
    
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)
        self.n_calls += 1
        cost = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost)
        return response

    def _calculate_cost(self, response: dict) -> float:
        # Get the number of tokens
        num_tokens = len(response.get("choices", [])[0].get("message", {}).get("content", "").split())
        return num_tokens
    

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
