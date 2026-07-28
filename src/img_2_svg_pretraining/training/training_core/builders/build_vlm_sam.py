"""Build a fully wired VLMSam composite model from configuration parameters."""
from __future__ import annotations

import argparse
import logging

import torch

from img_2_svg_pretraining.training.training_core.models.vlm_sam import VLMSam
from img_2_svg_pretraining.training.training_core.data_modules.vlms.common import apply_processor_image_override
from img_2_svg_pretraining.training.training_core.registry.registry import (
    DataModuleRegistry,
    SAMDataModuleRegistry,
    VLMModelRegistry,
)
from img_2_svg_pretraining.training.training_core.registry.utils import (
    DataArguments,
    DataModule,
    ModelConfig,
    SamModelArguments,
    VLMArguments,
)

log = logging.getLogger(__name__)


def _stub_change_tokenizer(tokenizer):
    """Minimal tokenizer extension that adds [SEG] for builder context."""
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    return tokenizer, seg_token_idx


def build_vlm_sam(
    vlm_family: str,
    vlm_checkpoint: str,
    sam_version: str,
    sam_checkpoint: str,
    model_family_name: str | None = None,
    vision_encoder: str | None = None,
    vision_encoder_checkpoint: str | None = None,
    vision_encoder_arch: str | None = None,
    attn_implementation: str | None = None,
    gradient_checkpointing: bool = False,
    out_dim: int = 256,
    bf16: bool = True,
) -> tuple["VLMSam", "DataModule"]:
    """Build a VLMSam composite model.

    Parameters
    ----------
    vlm_family:
        Registry key for the VLM family, e.g. ``"qwenvlm"``.
    vlm_checkpoint:
        Local path or HF model ID for the VLM checkpoint.
    model_family_name:
        Family-specific decoder variant name, e.g. ``"qwen2.5vl"`` or
        ``"molmo"``. When omitted, a backward-compatible best-effort default is
        derived from ``vlm_family`` and ``vlm_checkpoint``.
    sam_version:
        Registry key for the SAM version, e.g. ``"sam1"``.
    sam_checkpoint:
        Local path to the SAM checkpoint file.
    vision_encoder:
        Optional VisionEncoderRegistry key to swap the VLM's internal encoder.
        If None, the VLM's pretrained encoder is kept unchanged.
    vision_encoder_checkpoint:
        Checkpoint path / HF model ID for the standalone vision encoder.
        Required when vision_encoder is not None.
    vision_encoder_arch:
        Architecture string forwarded to open_clip (only relevant for
        openvision/openvision2 encoders).
    attn_implementation:
        Optional attention backend override (e.g. ``"flash_attention_2"``).
    gradient_checkpointing:
        Whether to enable VLM-side gradient checkpointing hooks at model load time.
    out_dim:
        SAM prompt embedding dimension expected by the projection MLP.
    bf16:
        Load VLM weights in bfloat16 when True (default).

    Returns
    -------
    (VLMSam, DataModule)
    """
    import os
    from pathlib import Path

    from img_2_svg_pretraining.training.training_core.builders.extract_vision_encoder import _resolve_model_family_name

    # ------------------------------------------------------------------ #
    # Registry side-effect imports
    # ------------------------------------------------------------------ #
    from img_2_svg_pretraining.training.training_core.data_modules.vlms.qwen import qwen_data  # noqa: F401
    from img_2_svg_pretraining.training.training_core.data_modules.vlms.gemma import gemma_data  # noqa: F401
    from img_2_svg_pretraining.training.training_core.data_modules.vlms.molmo import molmo_data  # noqa: F401
    from img_2_svg_pretraining.training.training_core.data_modules.sam.sam1 import sam1_data  # noqa: F401
    from img_2_svg_pretraining.training.training_core.models.vlms.qwen import qwen_model  # noqa: F401
    from img_2_svg_pretraining.training.training_core.models.vlms.gemma import gemma_model  # noqa: F401
    from img_2_svg_pretraining.training.training_core.models.vlms.molmo import molmo_model  # noqa: F401
    from img_2_svg_pretraining.training.training_core.models.sam.sam1 import sam1_model  # noqa: F401
    from img_2_svg_pretraining.training.training_core.vision_encoders.clip import clip_encoder  # noqa: F401
    from img_2_svg_pretraining.training.training_core.vision_encoders.siglip import siglip_encoder  # noqa: F401
    from img_2_svg_pretraining.training.training_core.vision_encoders.metaclip import metaclip_encoder  # noqa: F401
    from img_2_svg_pretraining.training.training_core.vision_encoders.openvision import openvision_encoder  # noqa: F401
    from img_2_svg_pretraining.training.training_core.vision_encoders.extracted import extracted_encoder  # noqa: F401

    resolved_model_family_name = _resolve_model_family_name(
        vlm_family=vlm_family,
        model_family_name=model_family_name,
        vlm_checkpoint=vlm_checkpoint,
    )

    # ------------------------------------------------------------------ #
    # Integrity check: SAM checkpoint
    # ------------------------------------------------------------------ #
    if not Path(sam_checkpoint).exists():
        log.warning("SAM checkpoint not found at %s", sam_checkpoint)

    # ------------------------------------------------------------------ #
    # SAM data module
    # ------------------------------------------------------------------ #
    sam_dm = SAMDataModuleRegistry.get_sam_data_module(sam_version)

    # ------------------------------------------------------------------ #
    # VLM data module (uses a minimal stub DataArguments for builder context)
    # ------------------------------------------------------------------ #
    from datasets import Dataset

    stub_ds = Dataset.from_list([{"image": None, "boxes": [[0, 0, 16, 16]]}])
    data_args = DataArguments(
        dataset_path="builder_stub",
        ds=stub_ds,
        seed=0,
        get_source=lambda source, **kw: (source, {}),
        get_source_kwargs={},
        extract_from_labels_fn=lambda _: [],
        layout_classes=[],
    )
    data_module: DataModule = DataModuleRegistry.get_module(
        vlm_family,
        data_args=data_args,
        change_tokenizer_fn=_stub_change_tokenizer,
        sam_collator=sam_dm.get_collator(),
        model_name=resolved_model_family_name,
        model_path=vlm_checkpoint,
    )

    # ------------------------------------------------------------------ #
    # Build composite model
    # ------------------------------------------------------------------ #
    config = ModelConfig(
        sam_args=SamModelArguments(
            tune_image_encoder=False,
            tune_prompt_encoder=False,
            tune_mask_decoder=True,
            checkpoint=sam_checkpoint,
            version=sam_version,
        ),
        vlm_args=VLMArguments(
            family=vlm_family,
            model_name_or_path=vlm_checkpoint,
            model_family_name=resolved_model_family_name,
            attn_implementation=attn_implementation,
            gradient_checkpointing=gradient_checkpointing,
            tune_mm_llm=False,
            tune_mm_vision=False,
            tune_mm_mlp=True,
            tune_mm_lm_head=True,
        ),
        vlm_family=vlm_family,
        sam_version=sam_version,
        out_dim=out_dim,
    )
    model = VLMSam(config=config, data_module=data_module)

    # ------------------------------------------------------------------ #
    # Optional vision encoder swap
    # ------------------------------------------------------------------ #
    if vision_encoder is not None:
        if not vision_encoder_checkpoint:
            raise ValueError(
                "vision_encoder_checkpoint is required when vision_encoder is set"
            )
        from img_2_svg_pretraining.training.training_core.registry.registry import VisionEncoderRegistry
        from img_2_svg_pretraining.training.training_core.builders.swap_vision_encoder import swap_vision_encoder

        enc_kwargs: dict = {"checkpoint": vision_encoder_checkpoint}
        if vision_encoder_arch:
            enc_kwargs["arch"] = vision_encoder_arch
        enc = VisionEncoderRegistry.get_encoder(vision_encoder, **enc_kwargs)
        apply_processor_image_override(data_module.processor, getattr(enc, "preprocessor_config", None))
        model.backbone, adapter = swap_vision_encoder(model.backbone, enc, vlm_family)
        if adapter is not None:
            log.info("VE swap inserted adapter: %s", adapter)

    # ------------------------------------------------------------------ #
    # Integrity checks (logged as warnings, not hard failures)
    # ------------------------------------------------------------------ #
    hidden_size = model.backbone.hidden_size
    proj_in = model.text_hidden_fcs_layout[0][0].in_features
    if hidden_size != proj_in:
        log.warning(
            "hidden_size mismatch: backbone=%d, text_hidden_fcs_layout in_features=%d",
            hidden_size,
            proj_in,
        )
    if model.seg_token_idx <= 0:
        log.warning("seg_token_idx=%d — tokenizer extension may not have run", model.seg_token_idx)

    return model, data_module


def main():
    parser = argparse.ArgumentParser(description="Build a VLMSam composite model.")
    parser.add_argument("--vlm_family", required=True)
    parser.add_argument("--vlm_checkpoint", required=True)
    parser.add_argument("--model_family_name", default=None)
    parser.add_argument("--sam_version", default="sam1")
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--vision_encoder", default=None)
    parser.add_argument("--vision_encoder_checkpoint", default=None)
    parser.add_argument("--vision_encoder_arch", default=None)
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--out_dim", type=int, default=256)
    args = parser.parse_args()

    model, _ = build_vlm_sam(
        vlm_family=args.vlm_family,
        vlm_checkpoint=args.vlm_checkpoint,
        model_family_name=args.model_family_name,
        sam_version=args.sam_version,
        sam_checkpoint=args.sam_checkpoint,
        vision_encoder=args.vision_encoder,
        vision_encoder_checkpoint=args.vision_encoder_checkpoint,
        vision_encoder_arch=args.vision_encoder_arch,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=args.gradient_checkpointing,
        out_dim=args.out_dim,
    )
    print(f"Built VLMSam: backbone.hidden_size={model.backbone.hidden_size}")
    print(f"              sam_head.prompt_embed_dim={model.sam_head.prompt_embed_dim}")
    if hasattr(model.backbone, "vision_adapter") and model.backbone.vision_adapter is not None:
        print(f"              vision_adapter={model.backbone.vision_adapter}")


if __name__ == "__main__":
    main()
