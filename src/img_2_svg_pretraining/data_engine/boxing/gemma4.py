"""Point-hint -> bounding-box grounding via Gemma4.

Sits between the pointing stage and segmentation: pointing gives one
approximate (x, y) hint per detected object, but a single point isn't
enough to prompt a promptable segmentation model (SAM2/SAM3) for a tight
mask -- point prompts on their own were found to under/over-segment on real
diagrams (see segmentation/classical_cv.py and segmentation/sam2_amg.py
docstrings for what was tried before this). Grounding each hint into an
explicit bounding box first, then feeding that box to SAM2/SAM3 as a box
prompt, is expected to be far more robust than a bare point, since a box
directly constrains the object's extent instead of relying on the promptable
model's own region-growing heuristics from a single click.

Reuses the `benchmark.models` registry (same chat-VLM loading pattern as
edges.py/codegen.py) -- Gemma4 is a general chat VLM, not a
detection-specialist model, so this leans on Gemma4's general visual
grounding ability via prompting rather than a task-specific head.

First pass: rasters only (embedded photo/plot regions), per explicit
instruction -- node boxing can follow once this is validated.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from img_2_svg_pretraining.benchmark.models import ModelSpec
from img_2_svg_pretraining.data_engine.pointing.common import PointResult

RASTER_BOX_PROMPT = """You are given a technical diagram image and a list of hint points, \
each marking approximately where one embedded raster graphic (photo, icon, logo, \
illustration, screenshot, or chart -- NOT a plain geometric node/box) is located.

HINT POINTS (pixel coordinates in this image, one per raster region):
{points}

For each hint point, determine the tight bounding box of the raster graphic region \
it falls on or nearest to. Respond ONLY with a JSON list of objects, one per hint \
point, in the same order: [{{"object_id": <id>, "box": [x0, y0, x1, y1]}}, ...] \
where [x0, y0, x1, y1] are pixel coordinates (top-left, bottom-right) of that \
region's tight bounding box. If a hint point does not correspond to any raster \
graphic, omit it from the list.
"""


@dataclass
class BoxResult:
    object_id: int
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixel coords


def _format_points(points: list[PointResult]) -> str:
    return "\n".join(f"- id {p.object_id}: ({p.x:.1f}, {p.y:.1f})" for p in points)


def _parse_boxes(raw_output: str, valid_ids: set[int]) -> list[BoxResult]:
    text = raw_output.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        entries = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    results = []
    for entry in entries:
        try:
            object_id = int(entry["object_id"])
            x0, y0, x1, y1 = (float(v) for v in entry["box"])
        except (KeyError, TypeError, ValueError):
            continue
        if object_id not in valid_ids:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        results.append(BoxResult(object_id=object_id, box=(x0, y0, x1, y1)))
    return results


class Gemma4BoxRunner:
    """Loads one Gemma4 chat VLM (any `benchmark.models.MODELS` gemma entry)
    and keeps it resident on GPU across calls."""

    def __init__(self, spec: ModelSpec, device: str = "cuda"):
        self.spec = spec
        self.device = device
        dtype = getattr(torch, spec.dtype)
        self.processor = AutoProcessor.from_pretrained(
            spec.hf_repo, trust_remote_code=spec.trust_remote_code,
        )
        # device_map=device streams weights straight to GPU during loading
        # instead of materializing the full model on CPU then `.to(device)`
        # transferring it in one shot -- the latter was observed to OOM a
        # single 80GB H100 loading gemma-4-31b (~62GB in bf16) even though
        # the model comfortably fits once resident, because the load-then-
        # transfer path briefly needs headroom for both the CPU copy being
        # read and the GPU copy being written plus allocator fragmentation
        # from one giant transfer.
        self.model = AutoModelForImageTextToText.from_pretrained(
            spec.hf_repo,
            trust_remote_code=spec.trust_remote_code,
            attn_implementation=spec.attn_implementation,
            dtype=dtype,
            device_map=device,
        ).eval()

    def unload(self):
        del self.model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def box_rasters(
        self, image_path: Path, points: list[PointResult], max_new_tokens: int = 2048,
    ) -> list[BoxResult]:
        if not points:
            return []

        image = Image.open(image_path).convert("RGB")
        prompt = RASTER_BOX_PROMPT.format(points=_format_points(points))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw_output = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

        valid_ids = {p.object_id for p in points}
        return _parse_boxes(raw_output, valid_ids)
