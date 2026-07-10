"""Automatic mask generation (AMG) via SAM2 -- image in, masks out, no
point/text prompts needed.

Explored 2026-07-08 as an alternative to point-prompted SAM3
(sam3_tracker.py) and the classical-CV flood-fill approach
(../deprecated_exps/cv_segmentation.py) after both struggled on real
`partial_test_set` diagrams: the classical pipeline's global-Otsu
binarization missed low-contrast/light-colored box borders entirely on
several samples (leaked into background on ~every point), and needed
per-image threshold tuning that doesn't generalize. SAM2's dense-grid AMG
mode found correct box-level masks on every style tested -- flat fill,
gradient/3D-shaded, photo content, low-contrast borders, nested containers
-- with no image-specific tuning, though it also over-segments text glyphs
inside boxes at default settings (filterable by mask area/rank downstream;
not yet implemented here).

Uses the `transformers` `pipeline("mask-generation", ...)` wrapper around
`facebook/sam2.1-hiera-large` rather than a bespoke model/processor call --
confirmed to load and run correctly in this container (~3s load, <2s
inference per image on one H100).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from img_2_svg_pretraining.data_engine.segmentation.models import Sam2Spec


@dataclass
class AmgMaskResult:
    mask: np.ndarray  # HxW bool array, full image resolution
    area: int


class Sam2AmgRunner:
    """Loads the SAM2 mask-generation pipeline once and keeps it resident
    across calls."""

    def __init__(self, spec: Sam2Spec, device: str = "cuda"):
        from transformers import pipeline

        self.spec = spec
        device_index = 0 if device == "cuda" else int(device.split(":")[-1])
        self.generator = pipeline("mask-generation", model=spec.hf_repo, device=device_index)

    def unload(self):
        del self.generator
        import torch
        torch.cuda.empty_cache()

    def generate(
        self, image_path: Path,
        points_per_batch: int = 64, points_per_crop: int = 32,
        pred_iou_thresh: float = 0.7, stability_score_thresh: float = 0.85,
    ) -> list[AmgMaskResult]:
        """Runs AMG and returns masks sorted largest-area-first (callers
        wanting only box-level masks, not individual text glyphs, should
        filter/rank on `area` -- see module docstring)."""
        outputs = self.generator(
            str(image_path), points_per_batch=points_per_batch, points_per_crop=points_per_crop,
            pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh,
        )
        results = []
        for m in outputs["masks"]:
            mask = m.cpu().numpy().astype(bool) if hasattr(m, "cpu") else np.array(m).astype(bool)
            results.append(AmgMaskResult(mask=mask, area=int(mask.sum())))
        results.sort(key=lambda r: -r.area)
        return results
