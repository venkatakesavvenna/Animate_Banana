"""Node/raster-region discovery via `allenai/MolmoPoint-8B`, the
pointing-specialist checkpoint.

Doesn't output pixel coordinates directly -- it generates special point
tokens that must be decoded with per-request metadata from the processor
(`return_pointing_metadata=True`) via `model.extract_image_points(...)`.
Confirmed against the model's own README (fetched 2026-07-07) -- the exact
call sequence (`build_logit_processor_from_inputs`,
`post_process_image_text_to_text`, `extract_image_points`) is required for
generation to produce valid point tokens at all, not just for parsing.

IMPORTANT: this model's bundled remote code only works against
`transformers==4.57.1` (its README's pinned version) -- 5.13.0, installed in
the project's main `img_2_svg_pretraining` venv, breaks in multiple places
(rope init, weight init, processor kwarg handling) because the remote code
predates transformers' 5.x rewrite of those internals. Rather than
monkeypatch around each break (fragile against whatever else differs) or
downgrade the shared venv (would risk breaking SAM3/benchmark, which are
only verified against 5.13.0), this module must be run from the dedicated
`/environments/molmo_point` venv (`--system-site-packages` for torch, with
`transformers==4.57.1` pinned on top).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from img_2_svg_pretraining.data_engine.pointing.common import NODE_QUERY, RASTER_QUERY, PointResult
from img_2_svg_pretraining.data_engine.pointing.models import Molmo2Spec


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
