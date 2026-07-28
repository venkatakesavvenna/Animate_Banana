from __future__ import annotations

import json
import os
from typing import Any, Union

from img_2_svg_pretraining.training.training_core.models.vlm_sam import VLMSam
from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry, DatasetRegistry, SAMDataModuleRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataArguments, DataModule, ModelConfig, SamModelArguments, VLMArguments

# Import for auto-registry side effects — covers all supported VLM families and SAM variants.
from img_2_svg_pretraining.training.training_core.data_modules.vlms.gemma import gemma_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.data_modules.vlms.qwen import qwen_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.data_modules.vlms.molmo import molmo_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.data_modules.sam.sam1 import sam1_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.data_modules.sam.noop import noop_sam_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.vlms.gemma import gemma_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.vlms.qwen import qwen_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.vlms.molmo import molmo_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.sam.sam1 import sam1_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.datasets.layout import custom_prompt  # noqa: F401


DEFAULT_VLM_FAMILY = "qwenvlm"
DEFAULT_MODEL_FAMILY_NAME = "qwen2.5vl"
DEFAULT_SAM_VERSION = "sam1"
DEFAULT_SAM_IMAGE_SIZE = 1024


def read_sam_version_from_checkpoint(checkpoint_path: str) -> str:
    """Read ``sam_version`` from a checkpoint's ``config.json`` without loading weights.

    ``ModelConfig`` stores ``sam_version`` as a plain attribute, so it is
    always present in any checkpoint's ``config.json`` produced by img_2_svg_pretraining training.

    Args:
        checkpoint_path: Path to the HF checkpoint directory.

    Returns:
        The ``sam_version`` string (e.g. ``"sam1"`` or ``"none"``).

    Raises:
        FileNotFoundError: If ``config.json`` is not found.
        KeyError:          If ``sam_version`` key is absent from the config.
    """
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"checkpoint config not found at {config_path!r}; "
            "ensure checkpoint_path points to an HF checkpoint directory."
        )
    with open(config_path) as f:
        cfg = json.load(f)
    if "sam_version" not in cfg:
        raise KeyError(
            f"'sam_version' key not found in {config_path!r}. "
            "Is this an img_2_svg_pretraining checkpoint?"
        )
    return str(cfg["sam_version"])


def add_seg_token(tokenizer):
    tokenizer.add_tokens("[SEG]")
    tokenizer.padding_side = "left"
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    return tokenizer, seg_token_idx


def build_inference_data_module(
    dataset_name: str,
    model_name_or_path: str,
    vlm_family: str = DEFAULT_VLM_FAMILY,
    model_family_name: str = DEFAULT_MODEL_FAMILY_NAME,
    sam_version: str = DEFAULT_SAM_VERSION,
    seed: int = 42,
    sam_image_size: int = DEFAULT_SAM_IMAGE_SIZE,
    dataset_kwargs: dict[str, Any] | None = None,
) -> tuple[DataArguments, DataModule]:
    dataset_kwargs = dict(dataset_kwargs or {})
    sam_data_module = SAMDataModuleRegistry.get_sam_data_module(sam_version)

    data_args: DataArguments = DatasetRegistry.get_dataset(
        dataset_name,
        get_sam_source_fn=sam_data_module.format_source,
        debug_path="",
        seed=seed,
        sam_image_size=sam_image_size,
        **dataset_kwargs,
    )
    data_module: DataModule = DataModuleRegistry.get_module(
        vlm_family,
        data_args=data_args,
        change_tokenizer_fn=add_seg_token,
        sam_collator=sam_data_module.get_collator(),
        model_name=model_family_name,
        model_path=model_name_or_path,
    )
    return data_args, data_module


def build_inference_model_config(
    model_name_or_path: str,
    vlm_family: str,
    model_family_name: str,
    sam_version: str,
    sam_checkpoint: str | None = None,
    attn_implementation: str | None = None,
) -> ModelConfig:
    """Build a ModelConfig for inference.

    For ``sam_version='none'`` (VLM-only) no SAM args are needed; pass
    ``sam_checkpoint=None`` and SAM args will be omitted from the config.
    """
    vlm_only = sam_version == "none"
    sam_args = None if vlm_only else SamModelArguments(
        tune_image_encoder=False,
        tune_prompt_encoder=False,
        tune_mask_decoder=False,
        checkpoint=sam_checkpoint or "",
        version=sam_version,
    )
    return ModelConfig(
        sam_args=sam_args,
        vlm_args=VLMArguments(
            family=vlm_family,
            model_name_or_path=model_name_or_path,
            model_family_name=model_family_name,
            attn_implementation=attn_implementation,
            tune_mm_llm=False,
            tune_mm_vision=False,
            tune_mm_mlp=False,
            tune_mm_lm_head=False,
        ),
        vlm_family=vlm_family,
        sam_version=sam_version,
    )


def load_inference_model(
    checkpoint_path: str,
    model_name_or_path: str,
    sam_checkpoint: str | None = None,
    device: str = "cuda:0",
    vlm_family: str = DEFAULT_VLM_FAMILY,
    model_family_name: str = DEFAULT_MODEL_FAMILY_NAME,
    sam_version: str = DEFAULT_SAM_VERSION,
    attn_implementation: str | None = None,
    data_module: DataModule | None = None,
) -> tuple[Union[VLMSam, "VLMOnly"], DataModule]:  # type: ignore[name-defined]
    """Load an img_2_svg_pretraining model checkpoint for inference.

    Automatically dispatches to :class:`VLMSam` or :class:`VLMOnly`
    depending on ``sam_version``:

    * ``sam_version != "none"`` → :class:`VLMSam` (VLM + SAM head)
    * ``sam_version == "none"`` → :class:`VLMOnly` (text generation only)

    Args:
        checkpoint_path:      Path to the saved HF checkpoint directory.
        model_name_or_path:   Base VLM model path (for processor/tokenizer).
        sam_checkpoint:       SAM weights path; ignored when ``sam_version="none"``.
        device:               Target device string (e.g. ``"cuda:0"``).
        vlm_family:           VLM family key registered in DataModuleRegistry.
        model_family_name:    Model family name used by the data module.
        sam_version:          ``"sam1"`` or ``"none"``.
        attn_implementation:  Attention backend override (``None`` = auto).
        data_module:          Pre-built DataModule; built on-the-fly if ``None``.

    Returns:
        (model, data_module) tuple — model is on ``device`` and in eval mode.
    """
    vlm_only = sam_version == "none"

    if data_module is None:
        _, data_module = build_inference_data_module(
            dataset_name="custom_prompt",
            model_name_or_path=model_name_or_path,
            vlm_family=vlm_family,
            model_family_name=model_family_name,
            sam_version=sam_version,
            dataset_kwargs={
                "data_list": [{"image": "dummy.png", "prompt": "dummy"}],
                "train": False,
            },
        )

    resolved_attn = attn_implementation
    if resolved_attn is None and not str(device).startswith("cuda"):
        resolved_attn = "eager"

    model_config = build_inference_model_config(
        model_name_or_path=model_name_or_path,
        vlm_family=vlm_family,
        model_family_name=model_family_name,
        sam_version=sam_version,
        sam_checkpoint=sam_checkpoint,
        attn_implementation=resolved_attn,
    )

    if vlm_only:
        from img_2_svg_pretraining.training.training_core.models.vlm_only import VLMOnly
        model = VLMOnly.from_pretrained(
            checkpoint_path,
            config=model_config,
            data_module=data_module,
        )
    else:
        model = VLMSam.from_pretrained(
            checkpoint_path,
            config=model_config,
            data_module=data_module,
        )

    model.eval()
    model.to(device)
    return model, data_module

