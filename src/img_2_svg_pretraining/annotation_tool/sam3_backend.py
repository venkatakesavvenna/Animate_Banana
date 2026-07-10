"""SAM3 interactive backend: embed once per image, predict per click.

Self-hosted `Sam3TrackerModel`/`Sam3TrackerProcessor` (transformers,
`facebook/sam3`) -- the point/box-promptable SAM2-compatible interactive
mode. NEVER the text/concept-prompted `Sam3Model`: "node" is a structural
role with no consistent visual signature, so concept prompting doesn't work
for this task (decision recorded in the design doc, backend choice confirmed
with the user 2026-07-10; no data leaves the machine).

The embed/predict split (spec section 9): `ensure_embedded()` runs the
expensive vision encoder once per image and caches the result; every
subsequent click only runs the cheap prompt-encoder/mask-decoder pass with
`image_embeddings` passed in. Both the model load and each actual embedding
compute emit a log line -- acceptance criteria 1-2 are verified by counting
those lines (one per process / one per image, never one per click).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

logger = logging.getLogger("annotation_tool.sam3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

DEFAULT_SAM3_REPO = "facebook/sam3"


@dataclass
class SegmentResult:
    mask: np.ndarray            # HxW bool, full image resolution
    score: float                # model's IoU self-estimate


class Sam3InteractiveBackend:
    """Holds the model plus the embedding of exactly ONE image (the one
    currently open in the app). One reviewer per process (spec section 12),
    so a single-slot embedding cache is all that's needed."""

    def __init__(self, hf_repo: str = DEFAULT_SAM3_REPO, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        self.model = Sam3TrackerModel.from_pretrained(hf_repo).to(self.device).eval()
        self.processor = Sam3TrackerProcessor.from_pretrained(hf_repo)
        logger.info("SAM3_LOAD repo=%s device=%s took=%.1fs (must appear once per process)",
                    hf_repo, self.device, time.perf_counter() - t0)
        self._embedded_image_id: str | None = None
        self._image: Image.Image | None = None
        self._image_embeddings = None

    # -- embedding guard ----------------------------------------------------

    def ensure_embedded(self, image_id: str, image: Image.Image) -> None:
        """No-op when `image_id` is already embedded; otherwise run the vision
        encoder once and cache. Must fire once per image, not per click."""
        if self._embedded_image_id == image_id:
            return
        t0 = time.perf_counter()
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self._image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        self._image = image
        self._embedded_image_id = image_id
        logger.info("SAM3_EMBED image_id=%s took=%.2fs (must appear once per image)",
                    image_id, time.perf_counter() - t0)

    # -- per-click prediction -------------------------------------------------

    @torch.no_grad()
    def segment(
        self,
        image_id: str,
        points: list[tuple[float, float, bool]] | None = None,
        box: tuple[float, float, float, float] | None = None,
    ) -> SegmentResult:
        """Predict one mask for the instance described by `points`
        ((x, y, positive) tuples) and/or `box` (x0, y0, x1, y1). At least one
        prompt must be given. `image_id` must already be embedded -- raising
        instead of silently re-embedding keeps the criteria-1/2 log counts
        honest."""
        if self._embedded_image_id != image_id:
            raise RuntimeError(
                f"segment() called for {image_id!r} but embedded image is "
                f"{self._embedded_image_id!r} -- call ensure_embedded() first")
        if not points and box is None:
            raise ValueError("need at least one point or a box prompt")

        kwargs = {}
        if points:
            kwargs["input_points"] = [[[[x, y] for x, y, _ in points]]]
            kwargs["input_labels"] = [[[int(pos) for _, _, pos in points]]]
        if box is not None:
            # This transformers version's Sam3TrackerProcessor validates boxes
            # as a 3-level nest: [image, box, coords] (checked against the
            # installed 5.3.0 -- the doc's warning about prompt-shape drift
            # between versions is real; older code here used 4 levels).
            kwargs["input_boxes"] = [[list(box)]]

        # The processor call re-runs image *pre*-processing (resize) so it can
        # rescale prompt coords, but pixel_values are dropped: the expensive
        # vision-encoder pass is replaced by the cached image_embeddings.
        inputs = self.processor(images=self._image, return_tensors="pt", **kwargs)
        inputs.pop("pixel_values")
        original_sizes = inputs.pop("original_sizes")
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}
        outputs = self.model(
            **inputs, image_embeddings=self._image_embeddings, multimask_output=False,
        )
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), original_sizes,
        )[0]
        mask = masks[0, 0].numpy().astype(bool)
        score = (float(outputs.iou_scores[0, 0, 0].cpu())
                 if hasattr(outputs, "iou_scores") else 1.0)
        return SegmentResult(mask=mask, score=score)
