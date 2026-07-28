from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from img_2_svg_pretraining.training.training_core.registry.registry import VisionEncoderRegistry
from img_2_svg_pretraining.training.training_core.vision_encoders.adapter_utils import (
    normalize_encoder_output,
    qwen_images_to_patch_tokens,
)
from img_2_svg_pretraining.training.training_core.vision_encoders.base import VisionEncoderBase


class ExtractedVisionEncoder(VisionEncoderBase):
    """Vision encoder that wraps a module extracted from a VLM.

    Weights are stored in ``weights.pt`` and preprocessing metadata in
    ``preprocessor.json`` under the saved directory.  Use
    ``ExtractedVisionEncoder.from_saved(path)`` to reload a previously
    extracted encoder.
    """

    def __init__(
        self,
        module: nn.Module,
        embed_dim: int,
        preprocessor_config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__()
        self._module = module
        self._embed_dim = embed_dim
        self._preprocessor_config = preprocessor_config
        self._metadata = metadata or {}

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def preprocessor_config(self) -> dict:
        return self._preprocessor_config

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        source_family = self._metadata.get("source_vlm_family")
        if source_family == "qwenvlm":
            return self._forward_qwen(pixel_values)
        if source_family == "molmovlm":
            return self._forward_molmo(pixel_values)
        return normalize_encoder_output(self._module(pixel_values))

    def save(self, directory: str | Path) -> None:
        """Persist weights and preprocessor metadata to ``directory``."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._module.cpu(), path / "module.pt")
        torch.save(self._module.state_dict(), path / "weights.pt")
        with open(path / "preprocessor.json", "w") as fh:
            json.dump(
                {
                    "embed_dim": self._embed_dim,
                    "metadata": self._metadata,
                    **self._preprocessor_config,
                },
                fh,
                indent=2,
            )

    @classmethod
    def from_saved(cls, directory: str | Path) -> "ExtractedVisionEncoder":
        """Reload an encoder saved with ``save()``."""
        path = Path(directory)
        with open(path / "preprocessor.json") as fh:
            meta = json.load(fh)

        embed_dim = int(meta.pop("embed_dim"))
        metadata = meta.pop("metadata", {})
        preprocessor_config = meta

        module_path = path / "module.pt"
        if module_path.exists():
            try:
                module = torch.load(module_path, map_location="cpu", weights_only=False)
                return cls(
                    module=module,
                    embed_dim=embed_dim,
                    preprocessor_config=preprocessor_config,
                    metadata=metadata,
                )
            except ModuleNotFoundError:
                # Remote-code modules (e.g. MolmoVisionBackbone) serialize their class
                # as transformers_modules.*; the class is only importable after the
                # trust_remote_code registration step.  If we have the source checkpoint
                # we can register the remote code via AutoConfig (config-only, no weights)
                # and then retry the load without touching the full model.
                source_checkpoint = metadata.get("source_checkpoint")
                if source_checkpoint:
                    try:
                        from transformers import AutoConfig
                        AutoConfig.from_pretrained(source_checkpoint, trust_remote_code=True)
                        module = torch.load(module_path, map_location="cpu", weights_only=False)
                        return cls(
                            module=module,
                            embed_dim=embed_dim,
                            preprocessor_config=preprocessor_config,
                            metadata=metadata,
                        )
                    except Exception:
                        pass
                if not (path / "weights.pt").exists():
                    raise

        weights_path = path / "weights.pt"
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        module = cls._reconstruct_module_from_source(metadata)
        if module is None:
            module = _StateModule(state)
        else:
            module.load_state_dict(state, strict=False)
        return cls(
            module=module,
            embed_dim=embed_dim,
            preprocessor_config=preprocessor_config,
            metadata=metadata,
        )

    @staticmethod
    def _reconstruct_module_from_source(metadata: dict[str, Any]) -> nn.Module | None:
        source_family = metadata.get("source_vlm_family")
        source_checkpoint = metadata.get("source_checkpoint")
        source_model_family_name = metadata.get("source_model_family_name")
        if not source_family or not source_checkpoint:
            return None

        from img_2_svg_pretraining.training.training_core.builders.extract_vision_encoder import _FAMILY_VE_ATTR, _get_any_nested_attr
        from img_2_svg_pretraining.training.training_core.registry.registry import VLMModelRegistry
        from img_2_svg_pretraining.training.training_core.registry.utils import VLMArguments

        vlm_wrapper = VLMModelRegistry.get_model(
            source_family,
            vlm_args=VLMArguments(
                family=source_family,
                model_name_or_path=source_checkpoint,
                model_family_name=source_model_family_name,
            ),
            training_args=None,
            gradient_checkpointing=False,
            bf16=False,
        )
        vlm_wrapper.eval()
        attr_paths = _FAMILY_VE_ATTR[source_family]
        return _get_any_nested_attr(vlm_wrapper, attr_paths)


@VisionEncoderRegistry.register_encoder("extracted")
def get_extracted_encoder(checkpoint: str, **_kwargs) -> ExtractedVisionEncoder:
    return ExtractedVisionEncoder.from_saved(checkpoint)


class _StateModule(nn.Module):
    """Minimal nn.Module that holds a state dict without a fixed architecture."""

    def __init__(self, state_dict: dict):
        super().__init__()
        for name, param in state_dict.items():
            self.register_buffer(name.replace(".", "_"), param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "_StateModule is a state holder only. "
            "To run a forward pass, recreate the full module from the original VLM."
        )


def _resolve_qwen_attr(module: nn.Module, *attr_names: str):
    for attr_name in attr_names:
        if hasattr(module, attr_name):
            return getattr(module, attr_name)
    raise AttributeError(f"Qwen extracted encoder is missing all of: {attr_names}")


def _resolve_molmo_grid(module: nn.Module) -> tuple[int, int]:
    config = getattr(module, "config", None)
    vision_cfg = getattr(config, "vision_backbone", None) if config is not None else None
    if vision_cfg is None:
        raise AttributeError("Molmo extracted encoder is missing config.vision_backbone")
    grid_h, grid_w = getattr(vision_cfg, "image_num_patch")
    return int(grid_h), int(grid_w)


def _resolve_molmo_patch_size(module: nn.Module) -> int:
    config = getattr(module, "config", None)
    vision_cfg = getattr(config, "vision_backbone", None) if config is not None else None
    if vision_cfg is None:
        return 14
    return int(getattr(vision_cfg, "image_patch_size", 14))


def _molmo_patchify_images(pixel_values: torch.Tensor, module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_values.ndim != 4:
        raise ValueError(f"Expected BCHW images, got shape {tuple(pixel_values.shape)}")

    batch, channels, height, width = pixel_values.shape
    grid_h, grid_w = _resolve_molmo_grid(module)
    if height % grid_h != 0 or width % grid_w != 0:
        patch_size = _resolve_molmo_patch_size(module)
        required_h = grid_h * patch_size
        required_w = grid_w * patch_size
        pixel_values = F.interpolate(
            pixel_values.float(),
            size=(required_h, required_w),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=pixel_values.dtype)
        batch, channels, height, width = pixel_values.shape

    patch_h = height // grid_h
    patch_w = width // grid_w
    patches = pixel_values.reshape(batch, channels, grid_h, patch_h, grid_w, patch_w)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.reshape(batch, 1, grid_h * grid_w, channels * patch_h * patch_w)
    image_masks = torch.ones((batch, 1, grid_h * grid_w), device=pixel_values.device, dtype=pixel_values.dtype)
    return patches, image_masks


def _stack_if_uniform(chunks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not chunks:
        raise ValueError("Expected at least one chunk to stack")
    if len({tuple(chunk.shape) for chunk in chunks}) != 1:
        raise ValueError("Cannot stack non-uniform extracted encoder chunks")
    return torch.stack(list(chunks), dim=0)


def _split_qwen_outputs(hidden_states: torch.Tensor, grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    split_sizes = [
        int(value)
        for value in (grid_thw.prod(dim=-1) // (spatial_merge_size * spatial_merge_size)).detach().cpu().tolist()
    ]
    chunks = torch.split(hidden_states, split_sizes, dim=0)
    return _stack_if_uniform(chunks)


def _forward_qwen(self: ExtractedVisionEncoder, pixel_values: torch.Tensor) -> torch.Tensor:
    patch_embed = _resolve_qwen_attr(self._module, "patch_embed")
    patch_tokens, grid_thw = qwen_images_to_patch_tokens(
        pixel_values,
        patch_size=int(getattr(patch_embed, "patch_size")),
        temporal_patch_size=int(getattr(patch_embed, "temporal_patch_size")),
        in_channels=int(getattr(patch_embed, "in_channels")),
    )
    hidden_states = self._module(patch_tokens, grid_thw=grid_thw)
    spatial_merge_size = int(_resolve_qwen_attr(self._module, "spatial_merge_size"))
    return _split_qwen_outputs(hidden_states, grid_thw, spatial_merge_size)


def _forward_molmo(self: ExtractedVisionEncoder, pixel_values: torch.Tensor) -> torch.Tensor:
    images, image_masks = _molmo_patchify_images(pixel_values, self._module)
    image_features, _ = self._module(images, image_masks)
    if image_features.ndim == 4 and image_features.shape[1] == 1:
        image_features = image_features[:, 0]
    return normalize_encoder_output(image_features)


ExtractedVisionEncoder._forward_qwen = _forward_qwen  # type: ignore[attr-defined]
ExtractedVisionEncoder._forward_molmo = _forward_molmo  # type: ignore[attr-defined]
