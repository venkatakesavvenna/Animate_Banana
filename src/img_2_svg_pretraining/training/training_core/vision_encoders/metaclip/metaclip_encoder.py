from __future__ import annotations

import torch

from img_2_svg_pretraining.training.training_core.registry.registry import VisionEncoderRegistry
from img_2_svg_pretraining.training.training_core.vision_encoders.base import VisionEncoderBase


class MetaCLIPVisionEncoder(VisionEncoderBase):
    """Wraps transformers.CLIPVisionModel for Meta's MetaCLIP checkpoints.

    MetaCLIP v1 (metaclip-l14-fullcc2.5b, metaclip-h14-fullcc2.5b) and
    MetaCLIP v2 (metaclip-2-worldwide-huge-quickgelu) both use the HF
    CLIPVisionModel class, but warrant separate registry keys because of the
    different expected embed_dims (1024 vs 1280).
    """

    def __init__(self, checkpoint: str):
        super().__init__()
        from transformers import CLIPVisionModel, CLIPImageProcessor

        # Use CLIPVisionModel directly to avoid loading the text encoder.
        # AutoModel.from_pretrained on a full CLIP checkpoint triggers a
        # text-encoder shape mismatch when the config.json was saved with
        # transformers<4.40 and the text_hidden_size field is misread.
        self._model = CLIPVisionModel.from_pretrained(checkpoint)
        try:
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
        except Exception:
            # Fallback to CLIP canonical values when the checkpoint does not ship
            # a standalone image processor config.
            self._preprocessor_config = {
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "image_size": 224,
            }

    @property
    def embed_dim(self) -> int:
        cfg = getattr(self._model, "config", None)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        raise AttributeError("MetaCLIP model does not expose a vision hidden_size")

    @property
    def preprocessor_config(self) -> dict:
        return self._preprocessor_config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._model(pixel_values=pixel_values).last_hidden_state


@VisionEncoderRegistry.register_encoder("metaclip")
def get_metaclip_encoder(checkpoint: str, **_kwargs) -> MetaCLIPVisionEncoder:
    return MetaCLIPVisionEncoder(checkpoint)


@VisionEncoderRegistry.register_encoder("metaclip2")
def get_metaclip2_encoder(checkpoint: str, **_kwargs) -> MetaCLIPVisionEncoder:
    return MetaCLIPVisionEncoder(checkpoint)
