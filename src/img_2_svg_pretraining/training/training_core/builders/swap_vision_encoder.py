"""Swap a VLM's internal vision backbone with a standalone VisionEncoder.

When the new encoder has a different embed_dim from the VLM's internal projector,
a small nn.Linear adapter is inserted between the new encoder's output and the
VLM's projector input.  The adapter is returned separately so the caller can
assign it to a dedicated optimizer param group.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling

from img_2_svg_pretraining.training.training_core.models.vlms.base import VLMBase
from img_2_svg_pretraining.training.training_core.vision_encoders.adapter_utils import (
    cast_tensor_to_module_dtype,
    normalize_encoder_output,
    qwen_patch_tokens_to_images,
    resize_token_sequence,
)
from img_2_svg_pretraining.training.training_core.vision_encoders.base import VisionEncoderBase

log = logging.getLogger(__name__)


class GemmaVisionTowerAdapter(nn.Module):
    def __init__(self, encoder: VisionEncoderBase, target_projector: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.target_projector = target_projector

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        pixel_values = cast_tensor_to_module_dtype(pixel_values, self.encoder)
        tokens = normalize_encoder_output(self.encoder(pixel_values))
        patches_per_image = int(getattr(self.target_projector, "patches_per_image"))
        resized = [
            resize_token_sequence(sample, target_h=patches_per_image, target_w=patches_per_image)
            for sample in tokens
        ]
        last_hidden_state = torch.stack(resized, dim=0)
        last_hidden_state = cast_tensor_to_module_dtype(last_hidden_state, self.target_projector)
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state)


class QwenVisionTowerAdapter(nn.Module):
    def __init__(
        self,
        encoder: VisionEncoderBase,
        merger: nn.Module,
        pre_merger_adapter: nn.Linear | None,
        *,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        spatial_merge_size: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.merger = merger
        self.pre_merger_adapter = pre_merger_adapter
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.spatial_merge_size = spatial_merge_size
        self.dtype = getattr(encoder, "dtype", torch.float32)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if grid_thw is None:
            raise ValueError("QwenVisionTowerAdapter requires grid_thw")
        images = qwen_patch_tokens_to_images(
            pixel_values,
            grid_thw,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=self.in_channels,
        )
        target_image_size = getattr(self.encoder, "preprocessor_config", {}).get("image_size")
        if target_image_size and images.shape[-2:] != (target_image_size, target_image_size):
            images = F.interpolate(
                images.float(),
                size=(int(target_image_size), int(target_image_size)),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=images.dtype)
        images = cast_tensor_to_module_dtype(images, self.encoder)
        tokens = normalize_encoder_output(self.encoder(images))
        resized = []
        for sample_tokens, thw in zip(tokens, grid_thw):
            _t, grid_h, grid_w = [int(v) for v in thw.detach().cpu().tolist()]
            target_h = grid_h
            target_w = grid_w
            resized.append(resize_token_sequence(sample_tokens, target_h=target_h, target_w=target_w))
        hidden_states = torch.cat(resized, dim=0)
        if self.pre_merger_adapter is not None:
            hidden_states = cast_tensor_to_module_dtype(hidden_states, self.pre_merger_adapter)
            hidden_states = self.pre_merger_adapter(hidden_states)
        hidden_states = cast_tensor_to_module_dtype(hidden_states, self.merger)
        return self.merger(hidden_states)


class MolmoVisionBackboneAdapter(nn.Module):
    def __init__(
        self,
        encoder: VisionEncoderBase,
        image_projector: nn.Module,
        *,
        pre_projector_adapter: nn.Linear | None,
        target_h: int,
        target_w: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.image_projector = image_projector
        self.pre_projector_adapter = pre_projector_adapter
        self.target_h = target_h
        self.target_w = target_w

    def forward(self, images: torch.Tensor, image_masks: torch.Tensor):
        if images.ndim == 4:
            batch_size, num_crops, num_patch, patch_dim = images.shape
            grid_size = int(math.isqrt(num_patch))
            patch_size = int(math.isqrt(patch_dim // 3))
            if grid_size * grid_size != num_patch or patch_size * patch_size * 3 != patch_dim:
                raise ValueError(
                    f"Cannot unpatchify Molmo crops with shape {tuple(images.shape)} into RGB images"
                )
            flat_images = images.reshape(batch_size * num_crops, grid_size, grid_size, 3, patch_size, patch_size)
            flat_images = flat_images.permute(0, 3, 1, 4, 2, 5).reshape(
                batch_size * num_crops,
                3,
                grid_size * patch_size,
                grid_size * patch_size,
            )
        elif images.ndim == 5:
            batch_size, num_crops = images.shape[:2]
            flat_images = images.reshape(batch_size * num_crops, *images.shape[2:])
        else:
            raise ValueError(
                f"MolmoVisionBackboneAdapter expects [B, crops, num_patch, patch_dim] "
                f"or [B, crops, C, H, W], got {tuple(images.shape)}"
            )

        batch_size, num_crops = int(batch_size), int(num_crops)

        target_image_size = getattr(self.encoder, "preprocessor_config", {}).get("image_size")
        if target_image_size and flat_images.shape[-2:] != (int(target_image_size), int(target_image_size)):
            flat_images = F.interpolate(
                flat_images.float(),
                size=(int(target_image_size), int(target_image_size)),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=flat_images.dtype)

        flat_images = cast_tensor_to_module_dtype(flat_images, self.encoder)
        tokens = normalize_encoder_output(self.encoder(flat_images))
        resized = [
            resize_token_sequence(sample, target_h=self.target_h, target_w=self.target_w)
            for sample in tokens
        ]
        image_features = torch.stack(resized, dim=0)
        if self.pre_projector_adapter is not None:
            image_features = cast_tensor_to_module_dtype(image_features, self.pre_projector_adapter)
            image_features = self.pre_projector_adapter(image_features)
        image_features = cast_tensor_to_module_dtype(image_features, self.image_projector)
        image_features = self.image_projector(image_features)
        image_features = image_features.reshape(batch_size, num_crops, self.target_h * self.target_w, -1)
        if image_masks is not None and image_masks.ndim == 2 and tuple(image_masks.shape) == (batch_size, num_crops):
            image_features = image_features * image_masks.unsqueeze(-1).unsqueeze(-1).to(image_features.dtype)
        return image_features, None


# ---------------------------------------------------------------------------
# Projector introspection helpers
# ---------------------------------------------------------------------------


def _get_qwen_visual(vlm_wrapper: VLMBase) -> nn.Module:
    if hasattr(vlm_wrapper.qwen, "model") and hasattr(vlm_wrapper.qwen.model, "visual"):
        return vlm_wrapper.qwen.model.visual
    return vlm_wrapper.qwen.visual

def _get_projector(vlm_wrapper: VLMBase, vlm_family: str) -> nn.Module:
    if vlm_family == "qwenvlm":
        return _get_qwen_visual(vlm_wrapper).merger
    if vlm_family == "gemmavlm":
        return vlm_wrapper.gemma.multi_modal_projector
    if vlm_family == "molmovlm":
        return _get_molmo_vision_backbone(vlm_wrapper).image_projector
    raise ValueError(f"swap_vision_encoder: unsupported vlm_family '{vlm_family}'")


def _get_projector_in_dim(vlm_wrapper: VLMBase, vlm_family: str) -> int:
    proj = _get_projector(vlm_wrapper, vlm_family)
    if vlm_family == "qwenvlm":
        # Qwen merger has a layer norm whose normalized_shape gives the input dim.
        if hasattr(proj, "ln_q") and hasattr(proj.ln_q, "normalized_shape"):
            return int(proj.ln_q.normalized_shape[-1])
    if vlm_family == "gemmavlm":
        # Gemma3MultiModalProjector: RMSNorm + AvgPool, no Linear.
        # Input dim is the vision encoder output dim, read from the norm.
        if hasattr(proj, "mm_soft_emb_norm") and hasattr(proj.mm_soft_emb_norm, "normalized_shape"):
            return int(proj.mm_soft_emb_norm.normalized_shape[-1])
        if hasattr(proj, "mm_soft_emb_norm") and hasattr(proj.mm_soft_emb_norm, "weight"):
            return int(proj.mm_soft_emb_norm.weight.shape[0])
    # Generic: search all sub-modules (recursive) for the first nn.Linear.
    for _name, mod in proj.named_modules():
        if hasattr(mod, "in_features"):
            return int(mod.in_features)
    # Fallback: projector itself is a Linear.
    if hasattr(proj, "in_features"):
        return int(proj.in_features)
    raise AttributeError(
        f"Cannot determine in_dim for {vlm_family} projector {type(proj).__name__}. "
        "Provide the dimension manually."
    )


def _set_projector(vlm_wrapper: VLMBase, vlm_family: str, new_proj: nn.Module) -> None:
    if vlm_family == "qwenvlm":
        vlm_wrapper.qwen.visual.merger = new_proj
        if hasattr(vlm_wrapper.qwen, "model") and hasattr(vlm_wrapper.qwen.model, "visual"):
            vlm_wrapper.qwen.model.visual.merger = new_proj
    elif vlm_family == "gemmavlm":
        vlm_wrapper.gemma.multi_modal_projector = new_proj
        if hasattr(vlm_wrapper.gemma, "model") and hasattr(vlm_wrapper.gemma.model, "multi_modal_projector"):
            vlm_wrapper.gemma.model.multi_modal_projector = new_proj
    elif vlm_family == "molmovlm":
        _get_molmo_vision_backbone(vlm_wrapper).image_projector = new_proj
    else:
        raise ValueError(f"swap_vision_encoder: unsupported vlm_family '{vlm_family}'")


def _set_vision_encoder(vlm_wrapper: VLMBase, vlm_family: str, new_enc: VisionEncoderBase) -> None:
    if vlm_family == "qwenvlm":
        vlm_wrapper.qwen.visual = new_enc
        if hasattr(vlm_wrapper.qwen, "model") and hasattr(vlm_wrapper.qwen.model, "visual"):
            vlm_wrapper.qwen.model.visual = new_enc
    elif vlm_family == "gemmavlm":
        vlm_wrapper.gemma.vision_tower = new_enc
        if hasattr(vlm_wrapper.gemma, "model") and hasattr(vlm_wrapper.gemma.model, "vision_tower"):
            vlm_wrapper.gemma.model.vision_tower = new_enc
    elif vlm_family == "molmovlm":
        molmo_parent, attr_name = _get_molmo_vision_backbone_parent(vlm_wrapper)
        setattr(molmo_parent, attr_name, new_enc)
    else:
        raise ValueError(f"swap_vision_encoder: unsupported vlm_family '{vlm_family}'")


def _get_molmo_vision_backbone_parent(vlm_wrapper: VLMBase) -> tuple[object, str]:
    if hasattr(vlm_wrapper.molmo, "vision_backbone"):
        return vlm_wrapper.molmo, "vision_backbone"
    if hasattr(vlm_wrapper.molmo, "model") and hasattr(vlm_wrapper.molmo.model, "vision_backbone"):
        return vlm_wrapper.molmo.model, "vision_backbone"
    raise AttributeError("Molmo wrapper does not expose a vision_backbone on either root or .model")


def _get_molmo_vision_backbone(vlm_wrapper: VLMBase) -> nn.Module:
    parent, attr_name = _get_molmo_vision_backbone_parent(vlm_wrapper)
    return getattr(parent, attr_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def swap_vision_encoder(
    vlm_wrapper: VLMBase,
    new_encoder: VisionEncoderBase,
    vlm_family: str,
) -> tuple[VLMBase, nn.Linear | None]:
    """Replace the VLM's internal vision backbone with new_encoder.

    If the encoder's embed_dim differs from the projector's expected input
    dimension, a ``nn.Linear(new_dim → original_dim)`` adapter is inserted
    before the projector.  The adapter is stored as
    ``vlm_wrapper.vision_adapter`` and returned as the second element of the
    tuple so callers can route it to a separate optimizer param group.

    Parameters
    ----------
    vlm_wrapper:
        A ``VLMBase`` instance (QwenModel, GemmaModel, or MolmoModel).
    new_encoder:
        The ``VisionEncoderBase`` instance to install.
    vlm_family:
        Registry family key: ``"qwenvlm"``, ``"gemmavlm"``, or ``"molmovlm"``.

    Returns
    -------
    (vlm_wrapper, adapter_or_None)
        The same vlm_wrapper with the encoder replaced (in-place modification),
        plus the inserted adapter (or None if dimensions matched).
    """
    if vlm_family == "qwenvlm":
        qwen_visual = _get_qwen_visual(vlm_wrapper)
        patch_embed = getattr(qwen_visual, "patch_embed", None)
        if patch_embed is None:
            original_dim = _get_projector_in_dim(vlm_wrapper, vlm_family)
            adapter: nn.Linear | None = None
            if new_encoder.embed_dim != original_dim:
                adapter = nn.Linear(new_encoder.embed_dim, original_dim, bias=True)
                nn.init.kaiming_uniform_(adapter.weight)
                nn.init.zeros_(adapter.bias)
                old_proj = _get_projector(vlm_wrapper, vlm_family)
                _set_projector(vlm_wrapper, vlm_family, nn.Sequential(adapter, old_proj))
            _set_vision_encoder(vlm_wrapper, vlm_family, new_encoder)
            vlm_wrapper.__dict__["vision_adapter"] = adapter
            return vlm_wrapper, adapter

        merger = _get_projector(vlm_wrapper, vlm_family)
        original_dim = int(getattr(patch_embed, "embed_dim", getattr(qwen_visual.config, "hidden_size")))
        new_dim = new_encoder.embed_dim
        adapter: nn.Linear | None = None
        if new_dim != original_dim:
            adapter = nn.Linear(new_dim, original_dim, bias=True)
            nn.init.kaiming_uniform_(adapter.weight)
            nn.init.zeros_(adapter.bias)
            log.warning(
                "VE dim mismatch: new=%d, original=%d. "
                "Inserted nn.Linear adapter inside the Qwen visual adapter. "
                "Train adapter first, then unfreeze the full model.",
                new_dim,
                original_dim,
            )

        target_encoder = QwenVisionTowerAdapter(
            new_encoder,
            merger=merger,
            pre_merger_adapter=adapter,
            patch_size=int(getattr(patch_embed, "patch_size")),
            temporal_patch_size=int(getattr(patch_embed, "temporal_patch_size")),
            in_channels=int(getattr(patch_embed, "in_channels")),
            spatial_merge_size=int(getattr(qwen_visual, "spatial_merge_size")),
        )
        _set_vision_encoder(vlm_wrapper, vlm_family, target_encoder)
        vlm_wrapper.__dict__["vision_adapter"] = adapter
        return vlm_wrapper, adapter

    if vlm_family == "molmovlm":
        original_backbone = _get_molmo_vision_backbone(vlm_wrapper)
        image_projector = _get_projector(vlm_wrapper, vlm_family)
        original_dim = _get_projector_in_dim(vlm_wrapper, vlm_family)
        new_dim = new_encoder.embed_dim
        adapter: nn.Linear | None = None
        if new_dim != original_dim:
            adapter = nn.Linear(new_dim, original_dim, bias=True)
            nn.init.kaiming_uniform_(adapter.weight)
            nn.init.zeros_(adapter.bias)
            log.warning(
                "VE dim mismatch: new=%d, original=%d. "
                "Inserted nn.Linear adapter inside the Molmo vision adapter. "
                "Train adapter first, then unfreeze the full model.",
                new_dim,
                original_dim,
            )

        target_h, target_w = original_backbone.config.llm_patches_per_crop()
        target_encoder = MolmoVisionBackboneAdapter(
            new_encoder,
            image_projector=image_projector,
            pre_projector_adapter=adapter,
            target_h=int(target_h),
            target_w=int(target_w),
        )
        _set_vision_encoder(vlm_wrapper, vlm_family, target_encoder)
        vlm_wrapper.__dict__["vision_adapter"] = adapter
        return vlm_wrapper, adapter

    original_dim = _get_projector_in_dim(vlm_wrapper, vlm_family)
    new_dim = new_encoder.embed_dim
    target_encoder = _build_target_vision_adapter(vlm_wrapper, vlm_family, new_encoder)

    adapter: nn.Linear | None = None
    if new_dim != original_dim:
        adapter = nn.Linear(new_dim, original_dim, bias=True)
        nn.init.kaiming_uniform_(adapter.weight)
        nn.init.zeros_(adapter.bias)

        old_proj = _get_projector(vlm_wrapper, vlm_family)
        new_proj = nn.Sequential(adapter, old_proj)
        _set_projector(vlm_wrapper, vlm_family, new_proj)
        log.warning(
            "VE dim mismatch: new=%d, original=%d. "
            "Inserted nn.Linear adapter before %s projector. "
            "Train adapter first, then unfreeze the full model.",
            new_dim,
            original_dim,
            vlm_family,
        )

    _set_vision_encoder(vlm_wrapper, vlm_family, target_encoder)
    # Keep a convenience handle without registering the adapter a second time as a
    # top-level submodule; it already lives inside the swapped projector path.
    vlm_wrapper.__dict__["vision_adapter"] = adapter
    return vlm_wrapper, adapter


def _build_target_vision_adapter(
    vlm_wrapper: VLMBase,
    vlm_family: str,
    new_encoder: VisionEncoderBase,
) -> nn.Module:
    if vlm_family == "gemmavlm":
        return GemmaVisionTowerAdapter(new_encoder, _get_projector(vlm_wrapper, vlm_family))
    if vlm_family == "qwenvlm":
        qwen_visual = _get_qwen_visual(vlm_wrapper)
        patch_embed = getattr(qwen_visual, "patch_embed", None)
        if patch_embed is None:
            return new_encoder
        spatial_merge_size = int(getattr(qwen_visual, "spatial_merge_size"))
        return QwenVisionTowerAdapter(
            new_encoder,
            patch_size=int(getattr(patch_embed, "patch_size")),
            temporal_patch_size=int(getattr(patch_embed, "temporal_patch_size")),
            in_channels=int(getattr(patch_embed, "in_channels")),
            spatial_merge_size=spatial_merge_size,
        )
    return new_encoder
