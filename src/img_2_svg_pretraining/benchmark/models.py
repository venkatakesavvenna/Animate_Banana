"""Registry of vision-language models under benchmark for Stage 1
(architecture diagram image -> TikZ).

All repo IDs below were confirmed to exist on the HuggingFace Hub (HTTP 200
on the Hub API) and to be tagged `image-text-to-text` where checked.
Inference runs via plain `transformers` (AutoModelForImageTextToText /
AutoProcessor + `.generate()`), not vLLM -- see infer.py. `attn_implementation`
defaults to "sdpa" (torch's native scaled-dot-product-attention, no compiled
CUDA extension needed) since this container's flash-attn build is tied to a
different CUDA toolkit version than the installed torch and can't be used;
override per-model if a given architecture doesn't support sdpa.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str               # short id used in filenames / CLI --model flag
    hf_repo: str             # exact HuggingFace repo id
    trust_remote_code: bool = False
    attn_implementation: str = "flash_attention_2"  # "sdpa" | "eager" | "flash_attention_2"
    dtype: str = "bfloat16"


MODELS: dict[str, ModelSpec] = {
    "qwen-3vl-8b": ModelSpec(
        name="qwen-3vl-8b",
        hf_repo="Qwen/Qwen3-VL-8B-Instruct",
    ),
    "qwen-3.5-9b": ModelSpec(
        name="qwen-3.5-9b",
        hf_repo="Qwen/Qwen3.5-9B",
    ),
    "qwen-3.5-4b": ModelSpec(
        name="qwen-3.5-4b",
        hf_repo="Qwen/Qwen3.5-4B",
    ),
    "gemma-4-12b": ModelSpec(
        name="gemma-4-12b",
        hf_repo="google/gemma-4-12B-it",
        attn_implementation="sdpa",
    ),
    "gemma-4-e4b": ModelSpec(
        name="gemma-4-e4b",
        hf_repo="google/gemma-4-E4B-it",
        attn_implementation="sdpa",
    ),
    "kimi-vl": ModelSpec(
        name="kimi-vl",
        hf_repo="moonshotai/Kimi-VL-A3B-Instruct",
        trust_remote_code=True,
    ),
    "phi-4": ModelSpec(
        name="phi-4",
        hf_repo="microsoft/Phi-4-multimodal-instruct",
        trust_remote_code=True,
    ),
}


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(MODELS)}")
    spec = MODELS[name]
    if spec.hf_repo == "REPLACE_ME":
        raise ValueError(
            f"Model '{name}' has no hf_repo configured yet -- edit benchmark/models.py"
        )
    return spec
