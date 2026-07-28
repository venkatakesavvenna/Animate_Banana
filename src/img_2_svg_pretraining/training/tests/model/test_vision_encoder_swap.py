from __future__ import annotations

import pytest
import torch

from tests.support.builders import build_vlm_sam_model, load_vlm_data_module, make_synthetic_image
from tests.support.matrix import (
    GEMMA3_SPEC,
    MOLMO_D_SPEC,
    MOLMO_O_SPEC,
    QWEN25_SPEC,
    resolve_model_ref,
    resolve_sam1_checkpoint,
)
from tests.support.runtime import runtime_enabled


def _skip_if_unavailable(model_ref, label: str):
    if not model_ref:
        pytest.skip(f"{label} checkpoint not set or path does not exist")
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
def test_swap_siglip_into_qwen_vlm_forward_passes(qwen25_model_id, sam1_checkpoint):
    """SigLIP-SO400M (1152 dim) swap into Qwen2.5-VL (also expects 1152) — no adapter."""
    _skip_if_unavailable(qwen25_model_id, "QWEN25")
    if not sam1_checkpoint:
        pytest.skip("SAM1 checkpoint unavailable")

    from img_2_svg_pretraining.training.training_core.registry.registry import VisionEncoderRegistry
    from img_2_svg_pretraining.training.training_core.builders.swap_vision_encoder import swap_vision_encoder
    from img_2_svg_pretraining.training.training_core.vision_encoders.siglip import siglip_encoder  # noqa: F401

    device = torch.device("cuda:0")
    data_module = load_vlm_data_module(
        spec=QWEN25_SPEC,
        model_ref=qwen25_model_id,
        image=make_synthetic_image(),
    )
    model = build_vlm_sam_model(
        spec=QWEN25_SPEC,
        model_ref=qwen25_model_id,
        data_module=data_module,
        sam_checkpoint=sam1_checkpoint,
    )

    enc = VisionEncoderRegistry.get_encoder(
        "siglip",
        checkpoint="google/siglip-so400m-patch14-384",
    )
    model.backbone, adapter = swap_vision_encoder(model.backbone, enc, QWEN25_SPEC.family)

    # Qwen2.5-VL's merger already expects 1152 — dims match, no adapter needed.
    assert adapter is None
    model.to(device)

    batch = data_module.Collator([data_module.Dataloader[0]])
    batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    outputs = model(**batch_device)

    assert "loss" in outputs
    assert torch.isfinite(outputs["loss"])


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
def test_swap_clip_into_qwen_vlm_adapter_inserted(qwen25_model_id, sam1_checkpoint):
    """CLIP ViT-L (1024) swap into Qwen2.5-VL (expects 1152) — adapter inserted."""
    _skip_if_unavailable(qwen25_model_id, "QWEN25")
    if not sam1_checkpoint:
        pytest.skip("SAM1 checkpoint unavailable")

    from img_2_svg_pretraining.training.training_core.registry.registry import VisionEncoderRegistry
    from img_2_svg_pretraining.training.training_core.builders.swap_vision_encoder import swap_vision_encoder
    from img_2_svg_pretraining.training.training_core.vision_encoders.clip import clip_encoder  # noqa: F401

    device = torch.device("cuda:0")
    data_module = load_vlm_data_module(
        spec=QWEN25_SPEC,
        model_ref=qwen25_model_id,
        image=make_synthetic_image(),
    )
    model = build_vlm_sam_model(
        spec=QWEN25_SPEC,
        model_ref=qwen25_model_id,
        data_module=data_module,
        sam_checkpoint=sam1_checkpoint,
    )

    enc = VisionEncoderRegistry.get_encoder(
        "clip",
        checkpoint="openai/clip-vit-large-patch14-336",
    )
    model.backbone, adapter = swap_vision_encoder(model.backbone, enc, QWEN25_SPEC.family)

    assert isinstance(adapter, torch.nn.Linear)
    assert adapter.in_features == 1024
    assert adapter.out_features != 1024  # should be the projector's expected dim
    model.to(device)

    batch = data_module.Collator([data_module.Dataloader[0]])
    batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    outputs = model(**batch_device)

    assert "loss" in outputs
    assert torch.isfinite(outputs["loss"])


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
def test_extract_qwen_visual_tower_and_reload(qwen25_model_id, large_artifact_dir):
    """Extract Qwen2.5-VL visual tower, save, reload, verify embed_dim."""
    _skip_if_unavailable(qwen25_model_id, "QWEN25")

    from img_2_svg_pretraining.training.training_core.builders.extract_vision_encoder import extract_vision_encoder
    from img_2_svg_pretraining.training.training_core.vision_encoders.extracted.extracted_encoder import ExtractedVisionEncoder

    enc = extract_vision_encoder(
        vlm_family=QWEN25_SPEC.family,
        vlm_checkpoint=qwen25_model_id,
        model_family_name=QWEN25_SPEC.model_family_name,
        encoder_name="qwen_visual",
        output_dir=str(large_artifact_dir),
    )
    assert enc.embed_dim > 0

    cfg = enc.preprocessor_config
    assert "image_mean" in cfg
    assert "image_std" in cfg
    assert "image_size" in cfg

    reloaded = ExtractedVisionEncoder.from_saved(large_artifact_dir / "qwen_visual")
    assert reloaded.embed_dim == enc.embed_dim
    assert reloaded.metadata["source_model_family_name"] == QWEN25_SPEC.model_family_name


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.parametrize(
    "spec",
    (QWEN25_SPEC, GEMMA3_SPEC, MOLMO_O_SPEC, MOLMO_D_SPEC),
    ids=lambda spec: f"{spec.slug}-extract-reload",
)
def test_extract_and_reload_source_encoder_path_for_matrix_sources(spec, large_artifact_dir):
    model_ref = resolve_model_ref(spec)
    _skip_if_unavailable(model_ref, spec.slug)

    from img_2_svg_pretraining.training.training_core.builders.extract_vision_encoder import extract_vision_encoder
    from img_2_svg_pretraining.training.training_core.registry.registry import VisionEncoderRegistry

    encoder_name = f"{spec.slug}_visual"
    saved_dir = large_artifact_dir / encoder_name
    enc = extract_vision_encoder(
        vlm_family=spec.family,
        vlm_checkpoint=model_ref,
        model_family_name=spec.model_family_name,
        encoder_name=encoder_name,
        output_dir=str(large_artifact_dir),
    )
    reloaded = VisionEncoderRegistry.get_encoder("extracted", checkpoint=str(saved_dir))

    assert saved_dir.exists()
    assert (saved_dir / "module.pt").exists()
    assert (saved_dir / "preprocessor.json").exists()
    assert reloaded.embed_dim == enc.embed_dim
    assert reloaded.metadata["source_vlm_family"] == spec.family
    assert reloaded.metadata["source_model_family_name"] == spec.model_family_name
