import logging
import os
from dataclasses import asdict
from typing import Any, Optional

import requests
import torch
import yaml
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("local_model")

#### Do not change this without changing the API. I'm not importing the API here for now.


def read_yaml_config(file_path: str):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# TODO(RUI): change this
config = read_yaml_config("config.yaml")
GATEWAY_URL = config.get("server_url")


class LLMPrompt(BaseModel):
    prompt: str
    max_tokens: int
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0


##################################################


class HuggingFaceModel(LitellmModel):
    config: LitellmModelConfig

    def __init__(self, **kwargs):
        self.config = LitellmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

    # @retry(
    #     stop=stop_after_attempt(10),
    #     wait=wait_exponential(multiplier=1, min=4, max=60),
    #     before_sleep=before_sleep_log(logger, logging.WARNING),
    # )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        logger.info("Querying model with messages: %s", messages)
        text = (
            "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            + "\nassistant: "
        )

        # Will have to remove the gpt2 from the endpoint
        logger.info(f"Sending request to {GATEWAY_URL}/generate/gpt2")
        prompt = LLMPrompt(prompt=text, max_tokens=50)
        response = requests.post(
            f"{GATEWAY_URL}/generate/gpt2", json=prompt.model_dump(), timeout=30
        )
        logger.info(
            f"Received response from {GATEWAY_URL}/generate/gpt2", response=response
        )
        response.raise_for_status()
        response = response.json().get("text", "")
        response_content = response.split("assistant:")[-1].strip()
        return {
            "choices": [{"message": {"role": "assistant", "content": response_content}}]
        }

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)
        self.n_calls += 1
        cost = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost)
        return {
            "content": response or "",  # type: ignore
        }

    def _calculate_cost(self, response: dict) -> float:
        return 1  # Example cost calculation

    # def get_template_vars(self) -> dict[str, Any]:
    #     return asdict(self.config) | {
    #         "n_model_calls": self.n_calls,
    #         "model_cost": self.cost,
    #     }
