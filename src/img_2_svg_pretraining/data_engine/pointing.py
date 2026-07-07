"""Node/raster-region discovery via point-prompted generation.

Two backends are supported (see models.py for the registry, run_pointing.py
for how to select one via --pointing-model):

- `MolmoPointRunner` (`allenai/MolmoPoint-8B`): the pointing-specialist
  checkpoint. Doesn't output pixel coordinates directly -- it generates
  special point tokens that must be decoded with per-request metadata from
  the processor (`return_pointing_metadata=True`) via
  `model.extract_image_points(...)`. Confirmed against the model's own
  README (fetched 2026-07-07) -- the exact call sequence
  (`build_logit_processor_from_inputs`, `post_process_image_text_to_text`,
  `extract_image_points`) is required for generation to produce valid point
  tokens at all, not just for parsing.
- `Molmo2PointRunner` (`allenai/Molmo2-8B`): the general-purpose Molmo2 VLM,
  which also supports pointing (trained on point/track data among other
  tasks). Much simpler integration: points come back as plain text
  (`<points ... coords="idx x y idx x y ..."/>`, coordinates scaled by 1000
  against image width/height) inside normal `.generate()` output, decoded
  with a regex -- no special logits processor or metadata plumbing needed.
  Confirmed against the model's own README's "Multi-Image Point QA" example
  (fetched 2026-07-07), adapted here for the single-image case.

Both expose the same `point_nodes`/`point_rasters` interface (two prompt
modes: one query per call, "point to X" -- run once with a nodes/boxes
query, once with a raster/photo-region query) and return `PointResult`.

IMPORTANT: Both models' bundled remote code only works against
`transformers==4.57.1` (their READMEs' pinned version) -- 5.13.0, installed
in the project's main `img_2_svg_pretraining` venv, breaks in multiple
places (rope init, weight init, processor kwarg handling for MolmoPoint at
least) because the remote code predates transformers' 5.x rewrite of those
internals. Rather than monkeypatch around each break (fragile against
whatever else differs) or downgrade the shared venv (would risk breaking
SAM3/benchmark, which are only verified against 5.13.0), this module must be
run from the dedicated `/environments/molmo_point` venv
(`--system-site-packages` for torch, with `transformers==4.57.1` pinned on
top) -- see run_pointing.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from img_2_svg_pretraining.data_engine.models import Molmo2Spec

# Node Queries version - 1: DO NOT CHANGE CLAUDE
# NODE_QUERY = (
#     "Point to every diagram shape (geometric objects such as rectangles, "
#     "rounded rectangles, circles, ellipses, cylinders, and polygons)"
# )
# RASTER_QUERY = (
#     "Point to every embedded graphic (including raster images, icons, "
#     "logos, illustrations, screenshots, and charts)."
# )

# Node Queries version - 2: DO NOT CHANGE CLAUDE
NODE_QUERY = (
    "Count all of the nodes in this diagram."
)
RASTER_QUERY = (
    "Count all of the rasterized images (including icons) in this diagram."
)

@dataclass
class PointResult:
    object_id: int
    x: float
    y: float


class MolmoPointRunner:
    """Loads the MolmoPoint pointing-specialist checkpoint and keeps it
    resident on GPU across calls."""

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
            padding_side="left",
        )

    def unload(self):
        del self.model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def point(self, image_path: Path, query: str, max_new_tokens: int = 400) -> list[PointResult]:
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
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            return_pointing_metadata=True,
        )
        metadata = inputs.pop("metadata")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16):
            output = self.model.generate(
                **inputs,
                logits_processor=self.model.build_logit_processor_from_inputs(inputs),
                max_new_tokens=max_new_tokens,
            )

        generated_tokens = output[:, inputs["input_ids"].size(1):]
        generated_text = self.processor.post_process_image_text_to_text(
            generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )[0]
        raw_points = self.model.extract_image_points(
            generated_text,
            metadata["token_pooling"],
            metadata["subpatch_mapping"],
            metadata["image_sizes"],
        )
        # raw_points: list of [object_id, image_num, x, y]; single image per
        # call here, so image_num is always 0.
        points = np.array(raw_points) if len(raw_points) else np.zeros((0, 4))
        return [
            PointResult(object_id=int(row[0]), x=float(row[2]), y=float(row[3]))
            for row in points
        ]

    def point_nodes(self, image_path: Path) -> list[PointResult]:
        return self.point(image_path, NODE_QUERY)

    def point_rasters(self, image_path: Path) -> list[PointResult]:
        return self.point(image_path, RASTER_QUERY)


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
    code paths (see module docstring), so callers shouldn't have to know
    which one a given --pointing-model key maps to."""
    if "molmopoint" in spec.hf_repo.lower():
        return MolmoPointRunner(spec, device=device)
    return Molmo2PointRunner(spec, device=device)
