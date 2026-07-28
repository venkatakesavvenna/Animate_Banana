"""FastAPI-style inference helpers for img_2_svg_pretraining training.

This module provides two layers of API:

1. **Low-level helpers** (``load_model``, ``run_inference``, ``inference_single``,
   ``inference_batch``): thin wrappers for callers that manage the model
   and data module themselves.  These remain backward-compatible with the
   ``test_mock_fastapi.py`` contract.

2. **High-level runner** (``InferenceRunner``): the preferred API for new code.
   Re-exported here for convenience so callers can do::

       from img_2_svg_pretraining.training.training_core.inference.fastapi import InferenceRunner

Both ``VLMSam`` (returns masks) and ``VLMOnly`` (returns ``None`` for masks)
are supported transparently.  ``run_inference`` branches on
``pred_masks is None`` instead of assuming masks are always present.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from img_2_svg_pretraining.training.training_core.datasets.layout import custom_prompt  # noqa: F401 — dataset registration
from img_2_svg_pretraining.training.training_core.inference.contract import InferenceResult  # re-export
from img_2_svg_pretraining.training.training_core.inference.factory import (
    DEFAULT_MODEL_FAMILY_NAME,
    DEFAULT_SAM_VERSION,
    DEFAULT_VLM_FAMILY,
    build_inference_data_module,
    load_inference_model,
)
from img_2_svg_pretraining.training.training_core.inference.runner import InferenceRunner  # re-export
from img_2_svg_pretraining.training.training_core.inference.utils import mask_to_bbox, move_to_device, prepare_generate_batch
from img_2_svg_pretraining.training.training_core.registry.utils import DataArguments, DataModule
from img_2_svg_pretraining.training.training_core.validation.eval_utils import to_numpy_mask


# ---------------------------------------------------------------------------
# Data-module helpers
# ---------------------------------------------------------------------------

def get_vlm_data_module(
    base_model: str,
    model_family_name: str = DEFAULT_MODEL_FAMILY_NAME,
    vlm_family: str = DEFAULT_VLM_FAMILY,
    sam_version: str = DEFAULT_SAM_VERSION,
    data_list=None,
):
    """Build a data module for inference.

    Args:
        base_model:         Base VLM model path / HF hub id.
        model_family_name:  Variant name used by the data module.
        vlm_family:         VLM family key.
        sam_version:        ``"sam1"`` or ``"none"``.
        data_list:          Optional list of ``{"image", "prompt"}`` dicts;
                            a dummy placeholder is used when ``None``.

    Returns:
        ``(data_args, data_module)`` tuple.
    """
    if not data_list:
        data_list = [{"image": "dummy.png", "prompt": "dummy"}]
    return build_inference_data_module(
        dataset_name="custom_prompt",
        model_name_or_path=base_model,
        vlm_family=vlm_family,
        model_family_name=model_family_name,
        sam_version=sam_version,
        dataset_kwargs={"data_list": data_list, "train": False},
    )


def get_qwen_data_module(
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    model_family_name: str = DEFAULT_MODEL_FAMILY_NAME,
    data_list=None,
):
    """Backward-compatible wrapper for the old Qwen-named helper."""
    return get_vlm_data_module(
        base_model=base_model,
        model_family_name=model_family_name,
        vlm_family=DEFAULT_VLM_FAMILY,
        sam_version=DEFAULT_SAM_VERSION,
        data_list=data_list,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    sam_checkpoint: Optional[str] = None,
    device: str = "cuda:0",
    vlm_family: str = DEFAULT_VLM_FAMILY,
    model_family_name: str = DEFAULT_MODEL_FAMILY_NAME,
    sam_version: str = DEFAULT_SAM_VERSION,
    attn_implementation: Optional[str] = None,
):
    """Load an img_2_svg_pretraining model checkpoint for inference.

    Dispatches to :class:`~img_2_svg_pretraining.training.training_core.models.vlm_sam.VLMSam` or
    :class:`~img_2_svg_pretraining.training.training_core.models.vlm_only.VLMOnly` based on ``sam_version``.

    Args:
        checkpoint_path:    Path to the HF checkpoint directory.
        base_model:         Base VLM model path.
        sam_checkpoint:     SAM weights path; ignored when
                            ``sam_version="none"``.  Defaults to ``None``.
        device:             Target device string (e.g. ``"cuda:0"``).
        vlm_family:         Registered VLM family name.
        model_family_name:  Family-specific model variant name.
        sam_version:        ``"sam1"`` or ``"none"``.
        attn_implementation: Attention backend override.

    Returns:
        Loaded composite model in eval mode on ``device``.
    """
    model, _ = load_inference_model(
        checkpoint_path=checkpoint_path,
        model_name_or_path=base_model,
        sam_checkpoint=sam_checkpoint,
        device=device,
        vlm_family=vlm_family,
        model_family_name=model_family_name,
        sam_version=sam_version,
        attn_implementation=attn_implementation,
    )
    print(f"Model loaded from {checkpoint_path} on {device}")
    return model


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------

def prepare_input_batch(
    data_list: List[Dict[str, Union[str, Image.Image]]],
    batch_size: int = 1,
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    model_family_name: str = DEFAULT_MODEL_FAMILY_NAME,
    vlm_family: str = DEFAULT_VLM_FAMILY,
    sam_version: str = DEFAULT_SAM_VERSION,
):
    """Prepare a batch of image+prompt pairs for model input.

    .. note::
        This function rebuilds the full data module on every call.
        For repeated inference calls, prefer
        :class:`~img_2_svg_pretraining.training.training_core.inference.runner.InferenceRunner` which
        builds the data module only once.

    Args:
        data_list:          List of ``{"image", "prompt"}`` dicts.
        batch_size:         DataLoader batch size.
        base_model:         Base VLM model path.
        model_family_name:  Variant name.
        vlm_family:         VLM family key.
        sam_version:        ``"sam1"`` or ``"none"``.

    Returns:
        ``(data_args, data_module, first_batch)`` triple.
    """
    from torch.utils.data import DataLoader

    data_args, data_module = get_vlm_data_module(
        base_model=base_model,
        model_family_name=model_family_name,
        vlm_family=vlm_family,
        sam_version=sam_version,
        data_list=data_list,
    )
    val_dataset = data_module.Dataloader
    collator = data_module.Collator

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, batch_size),
        collate_fn=collator,
    )
    first_batch = next(iter(val_loader))
    return data_args, data_module, first_batch


# ---------------------------------------------------------------------------
# Class-name extraction helper
# ---------------------------------------------------------------------------

def get_layouts(decoded_layouts, extract_layout_fn):
    """Apply ``extract_layout_fn`` to each decoded string.

    Args:
        decoded_layouts:  List of decoded prediction strings.
        extract_layout_fn: ``(str) -> List[str]`` callable.

    Returns:
        List of class-name lists.
    """
    return [extract_layout_fn(s) for s in decoded_layouts]


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    batch: Dict,
    data_module: DataModule,
    data_args: DataArguments,
    device: str = "cuda:0",
) -> Tuple[List[str], List[List[Dict]]]:
    """Run inference on a prepared batch.

    Supports both VLM+SAM (masks present) and VLM-only (``pred_masks is None``)
    checkpoints.  When ``pred_masks is None``, ``detections`` entries contain
    ``{"cls_name": str, "bbox": None}`` — no ``"mask_np"`` key.

    Args:
        model:       Loaded composite model (VLMSam or VLMOnly).
        batch:       Prepared input batch dict.
        data_module: Data module (for tokenizer access).
        data_args:   DataArguments (for ``extract_from_labels_fn``).
        device:      Device string.

    Returns:
        ``(preds, batch_detections)`` where ``batch_detections`` is a list of
        per-image detection lists.
    """
    tokenizer = data_module.processor.tokenizer
    extract_layout_fn = getattr(data_args, "extract_from_labels_fn", None)

    batch = move_to_device(batch, device)
    generate_batch = prepare_generate_batch(batch, tokenizer=tokenizer)

    with torch.no_grad():
        preds, pred_masks = model.generate(**generate_batch)

    pred_classes = get_layouts(preds, extract_layout_fn) if extract_layout_fn else [[] for _ in preds]
    batch_detections = []

    for i in range(len(preds)):
        pm = None if pred_masks is None else pred_masks[i]
        classes = pred_classes[i]

        if pm is None:
            # VLM-only: text-only detections, no mask_np
            detections = [{"cls_name": c, "bbox": None} for c in classes]
        else:
            pm_np = to_numpy_mask(pm, threshold=0.0)
            detections = []
            for idx in range(pm_np.shape[0]):
                cls_name = classes[idx] if idx < len(classes) else "UNDEFINED"
                bbox = mask_to_bbox(pm_np[idx])
                detections.append({
                    "bbox": bbox,
                    "cls_name": cls_name,
                    "mask_np": pm_np[idx],
                })

        batch_detections.append(detections)

    return preds, batch_detections


# ---------------------------------------------------------------------------
# Single-image convenience wrapper
# ---------------------------------------------------------------------------

def inference_single(
    image: Union[str, Image.Image],
    prompt: str,
    model,
    device: str = "cuda:0",
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    model_family_name: str = DEFAULT_MODEL_FAMILY_NAME,
    vlm_family: str = DEFAULT_VLM_FAMILY,
    sam_version: str = DEFAULT_SAM_VERSION,
) -> Dict:
    """Run inference on a single image+prompt pair.

    .. note::
        Rebuilds the data module on every call.  For repeated inference, use
        :class:`~img_2_svg_pretraining.training.training_core.inference.runner.InferenceRunner`.

    Args:
        image:    PIL Image or path to an image file.
        prompt:   Text prompt.
        model:    Loaded model (VLMSam or VLMOnly).
        device:   Device string.
        base_model, model_family_name, vlm_family, sam_version:
                  Data-module construction parameters.

    Returns:
        ``{"text": str, "detections": List[dict]}``
    """
    data_list = [{"image": image, "prompt": prompt}]
    data_args, data_module, first_batch = prepare_input_batch(
        data_list,
        batch_size=1,
        base_model=base_model,
        model_family_name=model_family_name,
        vlm_family=vlm_family,
        sam_version=sam_version,
    )
    batch_text, batch_detections = run_inference(model, first_batch, data_module, data_args, device)
    return {"text": batch_text[0], "detections": batch_detections[0]}


# ---------------------------------------------------------------------------
# Script entry point (smoke test / demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FLAT_CHECKPOINT_PATH = "/fsxvision_new/srihari.bandarupalli/DocGrounding/outputs/finetune_full_dataset_90k_pretrained_checkpoint/checkpoints/v2/checkpoint-60000"
    PROMPTABLE_CHECKPOINT_PATH = "/fsxvision_new/srihari.bandarupalli/DocGrounding/outputs/90k_checkpoint_promptable_finetune/checkpoints/v1/checkpoint-60000"
    BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    SAM_CHECKPOINT = "/fsxvision_new/srihari.bandarupalli/DocGrounding/checkpoints/sam_vit_h_4b8939.pth"
    DEVICE = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = load_model(
        checkpoint_path=FLAT_CHECKPOINT_PATH,
        base_model=BASE_MODEL,
        sam_checkpoint=SAM_CHECKPOINT,
        device=DEVICE,
    )

    image_path = "/fsxvision_new/raghuveer.r/Layout-Bench/sroie/shard_000/000000189/pages/sroie_000000189_page_1.png"
    layout_names_to_use = "[<list given by user>]"
    flat_prompt = f"Give me the layout of the document, use the following classes {layout_names_to_use}"
    question = "Which region contains the title of the invoice?"
    promptable_prompt = f"Give me the answer along with layout for this question: {question}"

    result = inference_single(
        image=image_path,
        prompt=promptable_prompt,
        model=model,
        device=DEVICE,
        base_model=BASE_MODEL,
    )

    print(f"Generated text: {result['text']}")
    print(f"Number of detections: {len(result['detections'])}")
    for i, det in enumerate(result["detections"]):
        print(f"  Detection {i}: class={det['cls_name']}, bbox={det['bbox']}")
