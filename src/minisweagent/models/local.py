import logging
from dataclasses import asdict
from typing import Any

import torch
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("local_model")


def get_device(verbose: bool = True) -> str:
    """
    Get the device to be used for training.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if verbose:
        logger.info("Getting device.", device=device)
    return device


class HuggingFaceModel(LitellmModel):
    config: LitellmModelConfig

    def __init__(self, **kwargs):
        self.config = LitellmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

        DEVICE = get_device()
        if DEVICE == "cuda":
            bnb_config = BitsAndBytesConfig(
                # Load the model with 4-bit quantization
                load_in_4bit=True,
                # Use double quantization
                bnb_4bit_use_double_quant=True,
                # Use 4-bit Normal Float for storing the base model weights in GPU memory
                bnb_4bit_quant_type="nf4",
                # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            output_hidden_states=True,
            output_attentions=True,
            attn_implementation="eager",  # <- crucial for output_attentions
        ).to(DEVICE)

        logger.info(
            f"Initialized local DeepSeekModel with model {self.config.model_name}"
        )

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        logger.info("Querying model with messages: %s", messages)
        prompt = (
            "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            + "\nassistant: "
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move inputs to the correct device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_content = response.split("assistant:")[-1].strip()
        return {
            "choices": [{"message": {"role": "assistant", "content": response_content}}]
        }

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)
        self.n_calls += 1
        cost = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost)
        return response

    def _calculate_cost(self, response: dict) -> float:
        # Get the number of tokens
        num_tokens = len(
            response.get("choices", [])[0].get("message", {}).get("content", "").split()
        )
        return num_tokens

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }
