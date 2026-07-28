from __future__ import annotations

import torch

from img_2_svg_pretraining.training.training_core.registry.registry import VisionEncoderRegistry
from img_2_svg_pretraining.training.training_core.vision_encoders.base import VisionEncoderBase


def _require_open_clip():
    try:
        import open_clip
        return open_clip
    except ImportError:
        raise ImportError(
            "OpenVisionEncoder requires open_clip_torch. "
            "Install it with: pip install open_clip_torch"
        )


def _preprocessor_config_from_transform(preprocess) -> dict:
    """Extract image_mean and image_std from an open_clip torchvision Compose."""
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    image_size = 224
    transforms = getattr(preprocess, "transforms", [])
    for t in transforms:
        if hasattr(t, "mean"):
            mean = tuple(t.mean)
        if hasattr(t, "std"):
            std = tuple(t.std)
        # torchvision Resize — access size attribute
        if hasattr(t, "size") and not hasattr(t, "mean"):
            sz = t.size
            if isinstance(sz, int):
                image_size = sz
            elif isinstance(sz, (list, tuple)) and len(sz) > 0:
                image_size = int(sz[0])
    return {
        "image_mean": list(mean),
        "image_std": list(std),
        "image_size": int(image_size),
    }


class OpenVisionEncoder(VisionEncoderBase):
    """Wraps open_clip models for the UCSC-VLAA OpenVision collection.

    OpenVision checkpoints are NOT loadable via HF transformers — they require
    the open_clip library. Example usage:

        enc = VisionEncoderRegistry.get_encoder(
            "openvision",
            checkpoint="UCSC-VLAA/openvision-vit-large-patch14-224",
            arch="ViT-L-14",
        )

    The `arch` argument is the open_clip architecture string (e.g. "ViT-L-14").
    If not provided, an auto-detection from the checkpoint name is attempted.
    """

    def __init__(self, checkpoint: str, arch: str | None = None):
        super().__init__()
        import json
        import os

        open_clip = _require_open_clip()

        if os.path.isdir(checkpoint):
            # Local HF snapshot directory.  open_clip's hf-hub: loader strips the
            # prefix and calls download_pretrained_from_hf(local_path), which fails
            # because huggingface_hub.hf_hub_download rejects absolute paths as
            # repo IDs.  Instead, read the config + weights directly from disk.
            config_path = os.path.join(checkpoint, "open_clip_config.json")
            weights_path = os.path.join(checkpoint, "open_clip_pytorch_model.bin")

            with open(config_path) as f:
                oc_cfg = json.load(f)

            model_name = f"_training_openvision_{os.path.basename(checkpoint)}"
            # Inject into open_clip's internal config registry (idempotent).
            import open_clip.factory as _oc_factory
            _oc_factory._MODEL_CONFIGS[model_name] = oc_cfg["model_cfg"]

            preprocess_cfg = oc_cfg.get("preprocess_cfg", {})
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=weights_path,
                image_mean=tuple(preprocess_cfg["mean"]) if "mean" in preprocess_cfg else None,
                image_std=tuple(preprocess_cfg["std"]) if "std" in preprocess_cfg else None,
            )
        else:
            # HF repo ID: use the standard hf-hub: prefix.
            self._model, self._preprocess = open_clip.create_model_from_pretrained(
                f"hf-hub:{checkpoint}",
            )

        self._model.eval()
        self._embed_dim = int(self._model.visual.output_dim)
        self._preprocessor_config = _preprocessor_config_from_transform(self._preprocess)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def preprocessor_config(self) -> dict:
        return self._preprocessor_config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._model.encode_image(pixel_values)


@VisionEncoderRegistry.register_encoder("openvision")
def get_openvision_encoder(checkpoint: str, arch: str | None = None, **_kwargs) -> OpenVisionEncoder:
    return OpenVisionEncoder(checkpoint, arch=arch)


@VisionEncoderRegistry.register_encoder("openvision2")
def get_openvision2_encoder(checkpoint: str, arch: str | None = None, **_kwargs) -> OpenVisionEncoder:
    return OpenVisionEncoder(checkpoint, arch=arch)
