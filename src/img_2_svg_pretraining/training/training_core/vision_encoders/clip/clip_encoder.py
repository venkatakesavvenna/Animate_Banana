from __future__ import annotations

import torch

from img_2_svg_pretraining.training.training_core.registry.registry import VisionEncoderRegistry
from img_2_svg_pretraining.training.training_core.vision_encoders.base import VisionEncoderBase


class CLIPVisionEncoder(VisionEncoderBase):
    """Wraps transformers.CLIPVisionModel.

    Used for OpenAI CLIP checkpoints (e.g. openai/clip-vit-large-patch14-336).
    MetaCLIP uses the same HF class and is handled by metaclip_encoder.py.
    """

    def __init__(self, checkpoint: str):
        super().__init__()
        from transformers import CLIPImageProcessor, CLIPVisionModel

        self._model = CLIPVisionModel.from_pretrained(checkpoint)
        proc = CLIPImageProcessor.from_pretrained(checkpoint)
        size = proc.size
        if isinstance(size, dict):
            image_size = size.get("shortest_edge") or size.get("height") or 224
        else:
            image_size = int(size)
        self._preprocessor_config = {
            "image_mean": list(proc.image_mean),
            "image_std": list(proc.image_std),
            "image_size": int(image_size),
        }

    @property
    def embed_dim(self) -> int:
        return int(self._model.config.hidden_size)

    @property
    def preprocessor_config(self) -> dict:
        return self._preprocessor_config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._model(pixel_values=pixel_values).last_hidden_state


@VisionEncoderRegistry.register_encoder("clip")
def get_clip_encoder(checkpoint: str, **_kwargs) -> CLIPVisionEncoder:
    return CLIPVisionEncoder(checkpoint)
