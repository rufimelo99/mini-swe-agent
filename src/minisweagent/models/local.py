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

        DEVICE = "cpu" #get_device()
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
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            output_hidden_states=True,
            output_attentions=True,
            attn_implementation="eager",  # <- crucial for output_attentions
            device_map="auto" if DEVICE == "cuda" else None,
        )

        logger.info(
            f"Initialized local DeepSeekModel with model {self.config.model_name}"
        )

    # @retry(
    #     stop=stop_after_attempt(10),
    #     wait=wait_exponential(multiplier=1, min=4, max=60),
    #     before_sleep=before_sleep_log(logger, logging.WARNING),
    # )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        logger.info("Querying model with messages: %s", messages)
        prompt = (
            "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            + "\nassistant: "
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        tokenizer = self.tokenizer
        model = self.model

        # 1) Establish model context window (GPT-2 uses 1024)
        model_ctx = getattr(model.config, "max_position_embeddings", None)
        if model_ctx is None:
            model_ctx = getattr(model.config, "n_positions", None)
        if model_ctx is None or model_ctx > 10**6:
            # fall back to tokenizer hint if config is unset/huge
            model_ctx = getattr(tokenizer, "model_max_length", 1024)

        # 2) Choose a generation budget (you already pass this in); default if None
        generation_budget = kwargs.get("generation_budget", 128)

        # 3) Compute max allowed input length
        max_input_len = max(1, model_ctx - generation_budget)

        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask", None)

        # 4) If over the budget, truncate FROM THE LEFT (keep most recent tokens)
        cur_len = input_ids.shape[-1]
        if cur_len > max_input_len:
            input_ids = input_ids[:, -max_input_len:]
            if attn_mask is not None:
                attn_mask = attn_mask[:, -max_input_len:]

        # 5) Ensure padding/eos for GPT-2 (it has no pad by default)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id

        # 6) Move to device and update inputs
        inputs = {
            "input_ids": input_ids.to(model.device),
            "attention_mask": attn_mask.to(model.device) if attn_mask is not None else None,
        }
        generation_budget = min(generation_budget, max(1, model_ctx - cur_len))

        # 7) Generate with max_new_tokens (do NOT pass max_length)
        outputs = model.generate(
            **{k: v for k, v in inputs.items() if v is not None},
            max_new_tokens=generation_budget,
            do_sample=True,           # or False
            temperature=0.7,
            top_p=0.9,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Model raw response: {response}")
        return response

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)
        self.n_calls += 1
        cost = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost)
        return {
            "content": response or "",  # type: ignore
        }

    def _calculate_cost(self, response: dict) -> float:
        # Get the number of tokens
        return len(self.tokenizer.tokenize(response))  # Example cost calculation

    # def get_template_vars(self) -> dict[str, Any]:
    #     return asdict(self.config) | {
    #         "n_model_calls": self.n_calls,
    #         "model_cost": self.cost,
    #     }
