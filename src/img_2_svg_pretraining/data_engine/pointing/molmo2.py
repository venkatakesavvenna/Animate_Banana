"""Node/raster-region discovery via `allenai/Molmo2-8B`, the general-purpose
Molmo2 VLM, which also supports pointing (trained on point/track data among
other tasks).

Simpler integration than molmo_point.py's MolmoPointRunner: points come back
as plain text (`<points ... coords="idx x y idx x y ..."/>`, coordinates
scaled by 1000 against image width/height) inside normal `.generate()`
output, decoded with a regex -- no special logits processor or metadata
plumbing needed. Confirmed against the model's own README's "Multi-Image
Point QA" example (fetched 2026-07-07), adapted here for the single-image
case.

Same `transformers==4.57.1` venv requirement as molmo_point.py -- see that
module's docstring for why.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from img_2_svg_pretraining.data_engine.pointing.common import NODE_QUERY, RASTER_QUERY, PointResult
from img_2_svg_pretraining.data_engine.pointing.models import Molmo2Spec

# Regexes from allenai/Molmo2-8B's README ("Multi-Image Point QA" section),
# adapted for the single-image case (no frame/image-index disambiguation
# needed since every call here has exactly one image).
_COORD_REGEX = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"/?>')
_POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def _extract_points(text: str, image_w: int, image_h: int) -> list[PointResult]:
    results = []
    for coord_match in _COORD_REGEX.finditer(text):
        for point_match in _POINTS_REGEX.finditer(coord_match.group(1)):
            object_id, x_raw, y_raw = point_match.groups()
            x = float(x_raw) / 1000 * image_w
            y = float(y_raw) / 1000 * image_h
            if 0 <= x <= image_w and 0 <= y <= image_h:
                results.append(PointResult(object_id=int(object_id), x=x, y=y))
    return results


class Molmo2PointRunner:
    """Loads the general-purpose Molmo2-8B model and keeps it resident on
    GPU across calls. Simpler than MolmoPointRunner: plain generate() +
    regex extraction, no special logits processor or metadata plumbing."""

    def __init__(self, spec: Molmo2Spec, device: str = "cuda"):
        self.spec = spec
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            spec.hf_repo,
            trust_remote_code=spec.trust_remote_code,
            dtype="auto",
            device_map=device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            spec.hf_repo,
            trust_remote_code=spec.trust_remote_code,
        )

    def unload(self):
        del self.model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def point(self, image_path: Path, query: str, max_new_tokens: int = 2048) -> list[PointResult]:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_tokens = output_ids[0, inputs["input_ids"].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return _extract_points(generated_text, image.width, image.height)

    def point_nodes(self, image_path: Path) -> list[PointResult]:
        return self.point(image_path, NODE_QUERY)

    def point_rasters(self, image_path: Path) -> list[PointResult]:
        return self.point(image_path, RASTER_QUERY)


def make_point_runner(spec: Molmo2Spec, device: str = "cuda"):
    """Dispatches to the right runner class based on which checkpoint is
    registered -- MolmoPoint and Molmo2 need different generation/extraction
    code paths, so callers shouldn't have to know which one a given
    --pointing-model key maps to."""
    if "molmopoint" in spec.hf_repo.lower():
        from img_2_svg_pretraining.data_engine.pointing.molmo_point import MolmoPointRunner
        return MolmoPointRunner(spec, device=device)
    return Molmo2PointRunner(spec, device=device)
