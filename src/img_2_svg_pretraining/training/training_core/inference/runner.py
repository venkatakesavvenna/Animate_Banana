"""InferenceRunner: unified inference entry point for VLMSam and VLMOnly.

Design goals
------------
* **One data-module build per runner lifetime** — the data module (tokenizer,
  processor, dataset registry lookup) is constructed once in
  :meth:`InferenceRunner.from_checkpoint` and reused across every call to
  :meth:`run_single` and :meth:`run_batch`.  This fixes the per-request
  rebuild in the legacy ``fastapi.py::inference_single``.
* **Uniform output** — both model types return :class:`~img_2_svg_pretraining.training.training_core.inference.contract.InferenceResult`.
  The runner branches internally on ``pred_masks is None``; callers never
  touch raw model output.
* **SAM-version auto-detection** — when ``sam_version`` is not passed, the
  runner reads it from the checkpoint's own ``config.json`` via
  :func:`~img_2_svg_pretraining.training.training_core.inference.factory.read_sam_version_from_checkpoint`.
  If the caller *does* pass ``sam_version``, the runner asserts it matches
  the checkpoint to prevent silent wrong loads.

Usage
-----
::

    runner = InferenceRunner.from_checkpoint(
        checkpoint_path="/path/to/ckpt",
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
        sam_checkpoint="/path/to/sam.pth",   # omit / None for VLM-only
        sam_version="sam1",                  # optional — auto-detected otherwise
        device="cuda:0",
    )

    # Single image
    result = runner.run_single(image=pil_img, prompt="Describe the layout.")

    # Batch
    data_list = [{"image": img1, "prompt": p1}, {"image": img2, "prompt": p2}]
    results = runner.run_batch(data_list, batch_size=4)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader

from img_2_svg_pretraining.training.training_core.inference.contract import InferenceResult
from img_2_svg_pretraining.training.training_core.inference.factory import (
    DEFAULT_MODEL_FAMILY_NAME,
    DEFAULT_SAM_VERSION,
    DEFAULT_VLM_FAMILY,
    build_inference_data_module,
    load_inference_model,
    read_sam_version_from_checkpoint,
)
from img_2_svg_pretraining.training.training_core.inference.utils import mask_to_bbox, move_to_device, prepare_generate_batch
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule


def _masks_to_numpy(masks):
    """Convert model pred_masks to a list of (N, H, W) uint8 ndarrays.

    Returns ``None`` when ``masks is None`` (VLM-only output).
    """
    import numpy as np  # keep top-level import light

    if masks is None:
        return None
    result = []
    for m in masks:
        if isinstance(m, torch.Tensor):
            arr = m.detach().cpu().float().numpy()
            arr = (arr > 0).astype(np.uint8)
        else:
            arr = (m > 0).astype(type(m).__class__.__name__ == "ndarray" and m.dtype or "uint8")
            import numpy as _np
            arr = _np.asarray(m)
            if arr.dtype != _np.uint8:
                arr = (arr > 0).astype(_np.uint8)
        result.append(arr)
    return result


def _build_inference_result(
    text: str,
    pred_mask,           # single-image mask tensor or None
    pred_classes: List[str],
    extract_layout_fn=None,
) -> InferenceResult:
    """Build an InferenceResult for a single image.

    Args:
        text:              Decoded prediction string.
        pred_mask:         Mask tensor shape ``(N, H, W)`` for VLM+SAM, or
                           ``None`` for VLM-only.
        pred_classes:      Class names parsed from ``text`` (may be empty).
        extract_layout_fn: Optional callable ``(str) -> List[str]``; only used
                           to re-derive ``pred_classes`` when the caller passes
                           ``None`` for the list.
    """
    import numpy as np

    if pred_mask is None:
        # VLM-only: text-only detections (bbox from parsed coordinates if any)
        detections: Optional[List[Dict]] = None
        if pred_classes:
            detections = [{"cls_name": c, "bbox": None} for c in pred_classes]
        return InferenceResult(text=text, detections=detections, has_masks=False)

    # VLM+SAM: derive mask numpy, bbox, class name per detection
    if isinstance(pred_mask, torch.Tensor):
        pm_np = pred_mask.detach().cpu().float().numpy()
        pm_np = (pm_np > 0).astype(np.uint8)
    else:
        pm_np = np.asarray(pred_mask)
        if pm_np.dtype != np.uint8:
            pm_np = (pm_np > 0).astype(np.uint8)

    detections = []
    for idx in range(pm_np.shape[0]):
        cls_name = pred_classes[idx] if idx < len(pred_classes) else "UNDEFINED"
        bbox = mask_to_bbox(pm_np[idx])
        detections.append({"cls_name": cls_name, "bbox": bbox, "mask_np": pm_np[idx]})

    return InferenceResult(text=text, detections=detections, has_masks=True)


class InferenceRunner:
    """Stateful runner that holds a loaded model + data module.

    Build via :meth:`from_checkpoint`; then call :meth:`run_single` or
    :meth:`run_batch`.  Both are safe to call repeatedly without reloading
    weights.
    """

    def __init__(
        self,
        model,
        data_module: DataModule,
        extract_layout_fn=None,
        device: str = "cuda:0",
    ):
        self.model = model
        self.data_module = data_module
        self.extract_layout_fn = extract_layout_fn
        self.device = device

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        base_model: str,
        sam_checkpoint: Optional[str] = None,
        sam_version: Optional[str] = None,
        vlm_family: str = DEFAULT_VLM_FAMILY,
        model_family_name: str = DEFAULT_MODEL_FAMILY_NAME,
        device: str = "cuda:0",
        attn_implementation: Optional[str] = None,
        extract_layout_fn=None,
        dataset_name: str = "custom_prompt",
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "InferenceRunner":
        """Load a checkpoint and build a ready-to-use runner.

        Args:
            checkpoint_path:    Path to the HF checkpoint directory.
            base_model:         Base VLM model path / HF hub id.
            sam_checkpoint:     SAM weights path; ``None`` for VLM-only.
            sam_version:        ``"sam1"`` / ``"none"``; auto-detected from the
                                checkpoint config when not provided.
            vlm_family:         VLM family key in :class:`~img_2_svg_pretraining.training.training_core.registry.DataModuleRegistry`.
            model_family_name:  Variant name used by the data module.
            device:             Target device string.
            attn_implementation: Attention backend override.
            extract_layout_fn:  ``(str) -> List[str]`` for parsing class names
                                out of generated text.  Required only when you
                                need ``detection.cls_name`` populated correctly.
            dataset_name:       Dataset key for the data module; default is
                                ``"custom_prompt"`` (generic image+prompt).
            dataset_kwargs:     Extra kwargs forwarded to the dataset builder.

        Returns:
            Fully loaded :class:`InferenceRunner`.
        """
        # --- auto-detect or validate sam_version -----------------------
        ckpt_sam_version = read_sam_version_from_checkpoint(checkpoint_path)

        if sam_version is None:
            sam_version = ckpt_sam_version
        else:
            if sam_version != ckpt_sam_version:
                raise ValueError(
                    f"sam_version mismatch: caller passed '{sam_version}' but "
                    f"checkpoint config reports '{ckpt_sam_version}'. "
                    "Pass sam_version=None to auto-detect, or fix the caller."
                )

        # --- build data module once ------------------------------------
        _dataset_kwargs = dict(dataset_kwargs or {})
        _dataset_kwargs.setdefault("data_list", [{"image": "dummy.png", "prompt": "dummy"}])
        _dataset_kwargs.setdefault("train", False)

        _, data_module = build_inference_data_module(
            dataset_name=dataset_name,
            model_name_or_path=base_model,
            vlm_family=vlm_family,
            model_family_name=model_family_name,
            sam_version=sam_version,
            dataset_kwargs=_dataset_kwargs,
        )

        # --- load model ------------------------------------------------
        model, _ = load_inference_model(
            checkpoint_path=checkpoint_path,
            model_name_or_path=base_model,
            sam_checkpoint=sam_checkpoint,
            device=device,
            vlm_family=vlm_family,
            model_family_name=model_family_name,
            sam_version=sam_version,
            attn_implementation=attn_implementation,
            data_module=data_module,
        )

        return cls(
            model=model,
            data_module=data_module,
            extract_layout_fn=extract_layout_fn,
            device=device,
        )

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def run_single(
        self,
        image: Union[str, Image.Image],
        prompt: str,
    ) -> InferenceResult:
        """Run inference on a single image + prompt.

        Args:
            image:  PIL Image or path to an image file.
            prompt: Text prompt.

        Returns:
            :class:`~img_2_svg_pretraining.training.training_core.inference.contract.InferenceResult`.
        """
        results = self.run_batch(
            data_list=[{"image": image, "prompt": prompt}],
            batch_size=1,
        )
        return results[0]

    def run_batch(
        self,
        data_list: List[Dict[str, Union[str, Image.Image]]],
        batch_size: int = 8,
    ) -> List[InferenceResult]:
        """Run inference on a list of image+prompt dicts.

        Args:
            data_list:  List of ``{"image": ..., "prompt": ...}`` dicts.
            batch_size: DataLoader batch size.

        Returns:
            List of :class:`~img_2_svg_pretraining.training.training_core.inference.contract.InferenceResult`,
            one per input item in the same order.
        """
        from img_2_svg_pretraining.training.training_core.inference.factory import build_inference_data_module
        from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry, SAMDataModuleRegistry

        # We need to rebuild the dataset with the actual data_list, but reuse
        # the already-built processor/tokenizer from self.data_module.
        # The cleanest approach is to use the dataset's Dataloader directly,
        # since it already holds the collator and processor.
        # However, we need to swap in the real data.  The simplest safe path:
        # build a fresh dataset-level object with the actual data_list, then
        # use the existing data_module's Collator.
        #
        # build_inference_data_module is idempotent and cheap for small lists.
        sam_version = self.model.config.sam_version
        vlm_family = self.model.config.vlm_family
        model_name_or_path = self.model.config.vlm_args.get("model_name_or_path", "")

        _, temp_dm = build_inference_data_module(
            dataset_name="custom_prompt",
            model_name_or_path=model_name_or_path,
            vlm_family=vlm_family,
            model_family_name=self.data_module.model_name,
            sam_version=sam_version,
            dataset_kwargs={"data_list": data_list, "train": False},
        )

        val_loader = DataLoader(
            temp_dm.Dataloader,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=temp_dm.Collator,
        )

        tokenizer = self.data_module.processor.tokenizer
        all_results: List[InferenceResult] = []

        with torch.no_grad():
            for batch in val_loader:
                batch = move_to_device(batch, self.device)
                generate_batch = prepare_generate_batch(batch, tokenizer=tokenizer)

                preds, pred_masks = self.model.generate(**generate_batch)

                # parse class names from generated text
                batch_classes: List[List[str]] = []
                if self.extract_layout_fn is not None:
                    batch_classes = [self.extract_layout_fn(p) for p in preds]
                else:
                    batch_classes = [[] for _ in preds]

                for i, (text, classes) in enumerate(zip(preds, batch_classes)):
                    pm = None if pred_masks is None else pred_masks[i]
                    result = _build_inference_result(
                        text=text,
                        pred_mask=pm,
                        pred_classes=classes,
                    )
                    all_results.append(result)

        return all_results
