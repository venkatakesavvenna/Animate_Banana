"""Extract and save a VLM's internal vision backbone as a standalone VisionEncoder."""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
from transformers import AutoProcessor

from img_2_svg_pretraining.training.training_core.registry.registry import VLMModelRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import VLMArguments
from img_2_svg_pretraining.training.training_core.vision_encoders.extracted.extracted_encoder import ExtractedVisionEncoder

log = logging.getLogger(__name__)


# Family → (attribute path, embed_dim probe attribute)
_FAMILY_VE_ATTR = {
    "qwenvlm": ("qwen.visual",),
    "gemmavlm": ("gemma.vision_tower",),
    "molmovlm": ("molmo.vision_backbone", "molmo.model.vision_backbone"),
}

_DEFAULT_MODEL_FAMILY_BY_VLM_FAMILY = {
    "qwenvlm": "qwen2.5vl",
    "gemmavlm": "gemma3",
    "molmovlm": "molmo",
}


def _get_nested_attr(obj, attr_path: str):
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def _get_any_nested_attr(obj, attr_paths: tuple[str, ...]):
    last_exc: Exception | None = None
    for attr_path in attr_paths:
        try:
            return _get_nested_attr(obj, attr_path)
        except AttributeError as exc:
            last_exc = exc
    raise AttributeError(
        f"None of the attribute paths were present: {attr_paths}"
    ) from last_exc


def _infer_embed_dim(module, vlm_family: str) -> int:
    if vlm_family == "molmovlm":
        # The vision backbone's image_projector maps ViT features → LLM hidden dim.
        # Read from w2 (the output linear of the SwiGLU MLP) for the true output dim.
        projector = getattr(module, "image_projector", None)
        if projector is not None:
            w2 = getattr(projector, "w2", None)
            if w2 is not None and hasattr(w2, "out_features"):
                return int(w2.out_features)
            if hasattr(projector, "out_features"):  # nn.Linear case
                return int(projector.out_features)
    if vlm_family == "qwenvlm":
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "out_hidden_size"):
            return int(config.out_hidden_size)
    # For Qwen visual and Gemma vision_tower, hidden_size is on the config.
    for attr in ("config.hidden_size", "config.embed_dim"):
        try:
            obj = module
            for part in attr.split("."):
                obj = getattr(obj, part)
            return int(obj)
        except AttributeError:
            continue
    # Last resort: count output parameters from a tiny forward pass.
    raise AttributeError(
        f"Cannot determine embed_dim for {vlm_family} vision module. "
        "Set it manually by passing embed_dim= to ExtractedVisionEncoder."
    )


def _extract_image_size(image_processor) -> int:
    # Molmo-style processors use base_image_input_size instead of size/crop_size
    base_input_size = getattr(image_processor, "base_image_input_size", None)
    if base_input_size is not None:
        if isinstance(base_input_size, (list, tuple)):
            return int(base_input_size[0])
        return int(base_input_size)

    size = getattr(image_processor, "size", None)
    crop_size = getattr(image_processor, "crop_size", None)

    for candidate in (crop_size, size):
        if isinstance(candidate, dict):
            if "height" in candidate:
                return int(candidate["height"])
            if "width" in candidate:
                return int(candidate["width"])
            if "shortest_edge" in candidate:
                value = int(candidate["shortest_edge"])
                return int(math.sqrt(value)) if value > 4096 else value
        elif isinstance(candidate, int):
            return int(candidate)

    for attr_name in ("max_pixels", "min_pixels"):
        value = getattr(image_processor, attr_name, None)
        if value:
            return int(math.sqrt(int(value)))

    return 224


def _infer_preprocessor_config(vlm_family: str, vlm_checkpoint: str) -> dict:
    processor = AutoProcessor.from_pretrained(
        vlm_checkpoint,
        trust_remote_code=(vlm_family != "qwenvlm"),
    )
    image_processor = getattr(processor, "image_processor", processor)
    mean = list(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]))
    std = list(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]))
    return {
        "image_mean": mean,
        "image_std": std,
        "image_size": _extract_image_size(image_processor),
    }


def _resolve_model_family_name(
    vlm_family: str,
    model_family_name: str | None,
    vlm_checkpoint: str,
) -> str:
    if model_family_name:
        return model_family_name

    ckpt_name = Path(vlm_checkpoint.rstrip("/")).name.lower()
    full_ref = vlm_checkpoint.lower()
    if vlm_family == "qwenvlm":
        if "qwen3" in full_ref:
            return "qwen3vl"
        if "qwen2.5" in full_ref:
            return "qwen2.5vl"
        return "qwen2vl"
    return _DEFAULT_MODEL_FAMILY_BY_VLM_FAMILY.get(vlm_family, ckpt_name)


def extract_vision_encoder(
    vlm_family: str,
    vlm_checkpoint: str,
    encoder_name: str,
    output_dir: str,
    model_family_name: str | None = None,
    preprocessor_config: dict | None = None,
) -> ExtractedVisionEncoder:
    """Load a VLM, extract its vision module, and save it to output_dir.

    Parameters
    ----------
    vlm_family:
        Registry key for the VLM family, e.g. ``"qwenvlm"``.
    vlm_checkpoint:
        Local path or HF model ID for the VLM checkpoint.
    encoder_name:
        Name used for the output sub-directory under output_dir.
    output_dir:
        Root directory where the encoder will be saved.
    preprocessor_config:
        Optional override for preprocessing metadata. If None, a minimal
        config is derived from the VLM model config.

    Returns
    -------
    ExtractedVisionEncoder
        A ready-to-use encoder wrapping the extracted module.
    """
    if vlm_family not in _FAMILY_VE_ATTR:
        raise ValueError(
            f"extract_vision_encoder: unsupported vlm_family '{vlm_family}'. "
            f"Supported: {sorted(_FAMILY_VE_ATTR)}"
        )

    # Ensure registry modules are loaded when this helper is used as a function,
    # not only through the CLI entrypoint.
    from img_2_svg_pretraining.training.training_core.models.vlms.qwen import qwen_model  # noqa: F401
    from img_2_svg_pretraining.training.training_core.models.vlms.gemma import gemma_model  # noqa: F401
    from img_2_svg_pretraining.training.training_core.models.vlms.molmo import molmo_model  # noqa: F401

    log.info("Loading %s VLM from %s for vision encoder extraction", vlm_family, vlm_checkpoint)
    vlm_args = VLMArguments(
        family=vlm_family,
        model_name_or_path=vlm_checkpoint,
        model_family_name=_resolve_model_family_name(vlm_family, model_family_name, vlm_checkpoint),
        attn_implementation=None,
    )
    vlm_wrapper = VLMModelRegistry.get_model(
        vlm_family,
        vlm_args=vlm_args,
        training_args=None,
        gradient_checkpointing=False,
        bf16=False,
    )
    vlm_wrapper.eval()

    attr_paths = _FAMILY_VE_ATTR[vlm_family]
    ve_module = _get_any_nested_attr(vlm_wrapper, attr_paths)
    embed_dim = _infer_embed_dim(ve_module, vlm_family)

    if preprocessor_config is None:
        preprocessor_config = _infer_preprocessor_config(vlm_family, vlm_checkpoint)

    enc = ExtractedVisionEncoder(
        module=ve_module,
        embed_dim=embed_dim,
        preprocessor_config=preprocessor_config,
        metadata={
            "source_vlm_family": vlm_family,
            "source_model_family_name": vlm_args.model_family_name,
            "source_checkpoint": vlm_checkpoint,
        },
    )

    save_path = Path(output_dir) / encoder_name
    enc.save(save_path)
    log.info("Saved extracted vision encoder to %s (embed_dim=%d)", save_path, embed_dim)
    return enc


def main():
    parser = argparse.ArgumentParser(description="Extract a VLM vision backbone to a standalone file.")
    parser.add_argument("--vlm_family", required=True, choices=sorted(_FAMILY_VE_ATTR))
    parser.add_argument("--vlm_checkpoint", required=True)
    parser.add_argument("--model_family_name", default=None)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # Ensure registry modules are loaded.
    from img_2_svg_pretraining.training.training_core.models.vlms.qwen import qwen_model  # noqa: F401
    from img_2_svg_pretraining.training.training_core.models.vlms.gemma import gemma_model  # noqa: F401
    from img_2_svg_pretraining.training.training_core.models.vlms.molmo import molmo_model  # noqa: F401
    from img_2_svg_pretraining.training.training_core.vision_encoders.extracted import extracted_encoder  # noqa: F401

    enc = extract_vision_encoder(
        vlm_family=args.vlm_family,
        vlm_checkpoint=args.vlm_checkpoint,
        model_family_name=args.model_family_name,
        encoder_name=args.encoder_name,
        output_dir=args.output_dir,
    )
    print(f"Extracted encoder embed_dim={enc.embed_dim}")
    print(f"Saved to {Path(args.output_dir) / args.encoder_name}")


if __name__ == "__main__":
    main()
