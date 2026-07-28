from __future__ import annotations

import os

import torch

from tests.support.matrix import GEMMA3_SPEC, MOLMO_D_SPEC, QWEN25_SPEC, VLMCheckpointSpec


LEGACY_RUNTIME_FLAGS = {
    "gpu": ("TRAINING_RUN_GPU_TESTS", "TRAINING_RUN_GPU_TRAINER_TESTS"),
    "ddp": ("TRAINING_RUN_DDP_TESTS",),
    "dataparallel": ("TRAINING_RUN_DATAPARALLEL_TESTS",),
}


def comma_env_values(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [value.strip() for value in raw.split(",") if value.strip()]


def runtime_enabled(runtime_name: str) -> bool:
    selected = comma_env_values("TRAINING_TEST_SELECTED_RUNTIMES", "")
    if selected:
        return runtime_name in selected

    if runtime_name == "cpu":
        return not any(runtime_enabled(name) for name in ("gpu", "ddp", "dataparallel"))

    for env_var in LEGACY_RUNTIME_FLAGS.get(runtime_name, ()):
        if os.environ.get(env_var, "0") == "1":
            return True
    return False


def trainer_backends_for_spec(spec: VLMCheckpointSpec, env_prefix: str) -> list[str]:
    if spec.family == "gemmavlm":
        return comma_env_values(f"TRAINING_GEMMA_{env_prefix}_ATTN_BACKENDS", "eager")
    if spec.family == "molmovlm":
        return comma_env_values(f"TRAINING_MOLMO_{env_prefix}_ATTN_BACKENDS", "default")
    return comma_env_values(f"TRAINING_QWEN_{env_prefix}_ATTN_BACKENDS", "flash_attention_2")


def trainer_backend_env_var(spec: VLMCheckpointSpec, suffix: str) -> str:
    if spec.family == "gemmavlm":
        return f"TRAINING_GEMMA_TRAINER_{suffix}_ATTN"
    if spec.family == "molmovlm":
        return f"TRAINING_MOLMO_TRAINER_{suffix}_ATTN"
    return f"TRAINING_QWEN_TRAINER_{suffix}_ATTN"


def gpu_wrapper_backends_for_spec(spec: VLMCheckpointSpec) -> list[str]:
    if spec.family == "gemmavlm":
        return comma_env_values("TRAINING_GEMMA_TEST_ATTN_BACKENDS", "eager")
    if spec.family == "molmovlm":
        return comma_env_values("TRAINING_MOLMO_TEST_ATTN_BACKENDS", "default")
    return comma_env_values("TRAINING_QWEN_TEST_ATTN_BACKENDS", "flash_attention_2")


def build_runtime_vlm_args_kwargs(spec: VLMCheckpointSpec, attn_implementation: str) -> dict:
    return {
        "attn_implementation": attn_implementation,
        "tune_mm_llm": False,
        "tune_mm_mlp": True,
        "tune_mm_vision": False,
        "tune_mm_lm_head": False,
    }


def select_vlm_batch(spec: VLMCheckpointSpec, batch: dict, device: torch.device) -> dict:
    if spec.family == GEMMA3_SPEC.family:
        keys = {"input_ids", "attention_mask", "labels", "pixel_values", "token_type_ids"}
    elif spec.family == MOLMO_D_SPEC.family:
        keys = {"input_ids", "attention_mask", "labels", "molmo_images", "image_input_idx", "image_masks"}
    else:
        keys = {
            "input_ids",
            "attention_mask",
            "labels",
            "pixel_values",
            "image_grid_thw",
            "position_ids",
        }

    selected = {}
    for key in keys:
        if key in batch and torch.is_tensor(batch[key]):
            selected[key] = batch[key].to(device)
    return selected


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def representative_model_ref_for_spec(
    spec: VLMCheckpointSpec,
    qwen25_model_id: str | None,
    gemma3_model_id: str,
    molmo_d_model_id: str | None = None,
) -> str | None:
    if spec.slug == QWEN25_SPEC.slug:
        return qwen25_model_id
    if spec.slug == GEMMA3_SPEC.slug:
        return gemma3_model_id
    if spec.slug == MOLMO_D_SPEC.slug:
        return molmo_d_model_id
    return None
