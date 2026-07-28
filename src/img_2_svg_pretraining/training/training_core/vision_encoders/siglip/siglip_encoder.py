from __future__ import annotations

import torch

from img_2_svg_pretraining.training.training_core.registry.registry import VisionEncoderRegistry
from img_2_svg_pretraining.training.training_core.vision_encoders.base import VisionEncoderBase


def _preprocessor_config_from_siglip_processor(proc) -> dict:
    size = getattr(proc, "size", None) or {}
    if isinstance(size, dict):
        image_size = size.get("height") or size.get("shortest_edge") or 224
    else:
        image_size = int(size)
    mean = getattr(proc, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(proc, "image_std", [0.5, 0.5, 0.5])
    return {
        "image_mean": list(mean),
        "image_std": list(std),
        "image_size": int(image_size),
    }


class SigLIPVisionEncoder(VisionEncoderBase):
    """Wraps transformers.SiglipVisionModel (SigLIP v1)."""

    def __init__(self, checkpoint: str):
        super().__init__()
        from transformers import AutoProcessor, SiglipVisionModel

        self._model = SiglipVisionModel.from_pretrained(checkpoint)
        proc = AutoProcessor.from_pretrained(checkpoint)
        image_proc = getattr(proc, "image_processor", proc)
        self._preprocessor_config = _preprocessor_config_from_siglip_processor(image_proc)

    @property
    def embed_dim(self) -> int:
        return int(self._model.config.hidden_size)

    @property
    def preprocessor_config(self) -> dict:
        return self._preprocessor_config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._model(pixel_values=pixel_values).last_hidden_state


class SigLIP2VisionEncoder(VisionEncoderBase):
    """Wraps transformers.Siglip2VisionModel (SigLIP v2).

    SigLIP2 uses a different HF class (Siglip2VisionModel) and should not
    be confused with SigLIP v1 (SiglipVisionModel).
    """

    def __init__(self, checkpoint: str):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        # The official HF SigLIP2 checkpoints are published for AutoModel loading.
        # Direct Siglip2VisionModel loading can mismatch patch-embed weights on the
        # transformers build used in this environment.
        loaded_model = AutoModel.from_pretrained(checkpoint)
        self._model = getattr(loaded_model, "vision_model", loaded_model)
        proc = AutoProcessor.from_pretrained(checkpoint)
        image_proc = getattr(proc, "image_processor", proc)
        self._preprocessor_config = _preprocessor_config_from_siglip_processor(image_proc)

    @property
    def embed_dim(self) -> int:
        cfg = getattr(self._model, "config", None)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        raise AttributeError("SigLIP2 model does not expose a vision hidden_size")

    @property
    def preprocessor_config(self) -> dict:
        return self._preprocessor_config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._model(pixel_values=pixel_values).last_hidden_state


@VisionEncoderRegistry.register_encoder("siglip")
def get_siglip_encoder(checkpoint: str, **_kwargs) -> SigLIPVisionEncoder:
    return SigLIPVisionEncoder(checkpoint)


@VisionEncoderRegistry.register_encoder("siglip2")
def get_siglip2_encoder(checkpoint: str, **_kwargs) -> SigLIP2VisionEncoder:
    return SigLIP2VisionEncoder(checkpoint)
