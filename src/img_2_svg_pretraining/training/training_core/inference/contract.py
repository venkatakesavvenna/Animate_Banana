"""Shared inference output contract for VLMSam and VLMOnly models.

A single ``InferenceResult`` is returned per *image* (not per batch).
Callers branch on ``has_masks`` (or ``result.detections is None``) instead of
using ``hasattr`` / ``try-except`` on the raw model output, making the failure
mode for a broken caller a clean ``TypeError`` rather than a silent wrong branch.

Detection dict shape
--------------------
* VLM-only (``has_masks=False``):
  ``{"bbox": [x1,y1,x2,y2] | None, "cls_name": str}``
  where ``bbox`` comes from text-parsed ``<box>`` coordinates or ``None``.
* VLM+SAM  (``has_masks=True``):
  ``{"bbox": [x1,y1,x2,y2] | None, "cls_name": str, "mask_np": np.ndarray}``
  where ``mask_np`` is a ``(H, W)`` uint8 array and ``bbox`` is derived from
  the mask contour via :func:`~img_2_svg_pretraining.training.training_core.inference.utils.mask_to_bbox`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class InferenceResult:
    """Single-image inference output.

    Attributes:
        text:       Raw decoded prediction string from the VLM.
        detections: Per-detection metadata list, or ``None`` when the model
                    produced no structured detections (e.g. pure captioning).
                    Each dict always contains ``"bbox"`` and ``"cls_name"``;
                    VLM+SAM results additionally carry ``"mask_np"``.
        has_masks:  ``True`` iff this result came from a VLM+SAM checkpoint
                    and ``detections`` entries include ``"mask_np"``.
                    Always ``False`` for VLM-only checkpoints.
    """

    text: str
    detections: Optional[List[Dict]] = field(default=None)
    has_masks: bool = field(default=False)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def pred_classes(self) -> List[str]:
        """Return predicted class names (empty list when no detections)."""
        if self.detections is None:
            return []
        return [d.get("cls_name", "UNDEFINED") for d in self.detections]

    def pred_bboxes(self) -> List[Optional[List[int]]]:
        """Return predicted bounding boxes (empty list when no detections)."""
        if self.detections is None:
            return []
        return [d.get("bbox") for d in self.detections]

    def pred_masks_np(self) -> List[np.ndarray]:
        """Return mask arrays for VLM+SAM results (empty list for VLM-only)."""
        if not self.has_masks or self.detections is None:
            return []
        return [d["mask_np"] for d in self.detections if "mask_np" in d]
