"""Point- and box-prompted segmentation via SAM3.

Uses `Sam3TrackerModel`/`Sam3TrackerProcessor` (the point/box-promptable
per-object variant), not `Sam3Model`/`Sam3Processor` (the text-prompted
"segment all instances of a concept" variant) -- confirmed against the
model's own README (`facebook/sam3`, fetched 2026-07-07): the tracker
variant takes `input_points`/`input_labels` or `input_boxes` and returns one
mask per object.

Each Molmo point becomes one positive click for its own object -- points/
boxes are segmented one-at-a-time (not batched as multiple objects in one
call) since mask cache paths are keyed per point/node id and we need results
streamed one at a time so a single failing point doesn't block the rest.

`segment_box` (box-prompted) was added 2026-07-08 alongside
`boxing/gemma4.py`: a bare point prompt was found to under/over-segment on
real diagrams (see segmentation/classical_cv.py and sam2_amg.py docstrings),
so the new flow is pointing -> Gemma4 grounds each point hint into a tight
box -> that box is fed here as a box prompt, which should constrain the
segmented region far more reliably than a single click.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

from img_2_svg_pretraining.data_engine.segmentation.models import Sam3Spec


@dataclass
class MaskResult:
    mask: np.ndarray          # HxW bool array, full image resolution
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixel coords
    score: float


def _mask_to_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


class Sam3Runner:
    """Loads SAM3 once and keeps it resident on GPU across calls."""

    def __init__(self, spec: Sam3Spec, device: str = "cuda"):
        self.spec = spec
        self.device = device
        self.model = Sam3TrackerModel.from_pretrained(spec.hf_repo).to(device).eval()
        self.processor = Sam3TrackerProcessor.from_pretrained(spec.hf_repo)

    def unload(self):
        del self.model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def segment_point(self, image_path: Path, x: float, y: float) -> MaskResult:
        image = Image.open(image_path).convert("RGB")
        input_points = [[[[x, y]]]]
        input_labels = [[[1]]]
        inputs = self.processor(
            images=image, input_points=input_points, input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"],
        )[0]
        # masks: [num_objects, num_masks_per_object, H, W]; single point/object,
        # multimask_output=False -> take the sole mask.
        mask = masks[0, 0].numpy().astype(bool)
        score = float(outputs.iou_scores[0, 0, 0].cpu()) if hasattr(outputs, "iou_scores") else 1.0
        return MaskResult(mask=mask, bbox=_mask_to_bbox(mask), score=score)

    @torch.no_grad()
    def segment_box(self, image_path: Path, box: tuple[float, float, float, float]) -> MaskResult:
        """box: (x0, y0, x1, y1) in pixel coordinates, e.g. from
        `boxing/gemma4.py::Gemma4BoxRunner`."""
        image = Image.open(image_path).convert("RGB")
        # input_boxes must be exactly 3 levels: [image, box, coords] --
        # unlike input_points (4 levels: [image, object, point, coords]),
        # confirmed against Sam3TrackerProcessor's own validation error.
        input_boxes = [[list(box)]]
        inputs = self.processor(
            images=image, input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"],
        )[0]
        mask = masks[0, 0].numpy().astype(bool)
        score = float(outputs.iou_scores[0, 0, 0].cpu()) if hasattr(outputs, "iou_scores") else 1.0
        return MaskResult(mask=mask, bbox=_mask_to_bbox(mask), score=score)


def save_mask(mask: np.ndarray, cache_dir: Path, sample_id: str, node_id: str) -> Path:
    out_dir = cache_dir / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:8]
    path = out_dir / f"{node_id}_{digest}.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)
    return path
