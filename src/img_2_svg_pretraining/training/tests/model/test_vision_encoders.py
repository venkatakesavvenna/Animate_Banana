from __future__ import annotations

import pytest
import torch

from img_2_svg_pretraining.training.training_core.registry.registry import VisionEncoderRegistry
from tests.support.runtime import runtime_enabled


def _skip_gpu_if_needed():
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")


def _run_encoder_smoke(enc_key: str, checkpoint: str, expected_dim: int, arch: str | None = None):
    _skip_gpu_if_needed()

    kwargs = {"checkpoint": checkpoint}
    if arch is not None:
        kwargs["arch"] = arch

    enc = VisionEncoderRegistry.get_encoder(enc_key, **kwargs)
    enc.eval()

    assert enc.embed_dim == expected_dim, f"embed_dim mismatch: {enc.embed_dim} != {expected_dim}"

    cfg = enc.preprocessor_config
    assert "image_mean" in cfg and len(cfg["image_mean"]) == 3
    assert "image_std" in cfg and len(cfg["image_std"]) == 3
    assert "image_size" in cfg and int(cfg["image_size"]) > 0

    image_size = int(cfg["image_size"])
    dummy = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        out = enc(dummy)
    # Output must be at least 3-D: [B, tokens, dim]
    assert out.shape[-1] == expected_dim, f"output last dim {out.shape[-1]} != {expected_dim}"
    assert out.ndim >= 2


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.parametrize("checkpoint,expected_dim", [
    ("openai/clip-vit-large-patch14-336", 1024),
    ("openai/clip-vit-base-patch32", 512),
])
def test_clip_encoder(checkpoint, expected_dim):
    _run_encoder_smoke("clip", checkpoint, expected_dim)


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.parametrize("checkpoint,expected_dim", [
    ("google/siglip-so400m-patch14-384", 1152),
    ("google/siglip-base-patch16-224", 768),
])
def test_siglip_encoder(checkpoint, expected_dim):
    _run_encoder_smoke("siglip", checkpoint, expected_dim)


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.parametrize("checkpoint,expected_dim", [
    ("google/siglip2-so400m-patch14-384", 1152),
])
def test_siglip2_encoder(checkpoint, expected_dim):
    _run_encoder_smoke("siglip2", checkpoint, expected_dim)


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.parametrize("enc_key,checkpoint,expected_dim", [
    ("metaclip", "facebook/metaclip-l14-fullcc2.5b", 1024),
    ("metaclip", "facebook/metaclip-h14-fullcc2.5b", 1280),
    ("metaclip2", "facebook/metaclip-2-worldwide-huge-quickgelu", 1280),
])
def test_metaclip_encoder(enc_key, checkpoint, expected_dim):
    _run_encoder_smoke(enc_key, checkpoint, expected_dim)


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.parametrize("enc_key,checkpoint,arch,expected_dim", [
    ("openvision", "UCSC-VLAA/openvision-vit-large-patch14-224", "ViT-L-14", 1024),
    ("openvision", "UCSC-VLAA/openvision-vit-base-patch16-384", "ViT-B-16", 768),
])
def test_openvision_encoder(enc_key, checkpoint, arch, expected_dim):
    try:
        import open_clip  # noqa: F401
    except ImportError:
        pytest.skip("open_clip_torch not installed")
    _run_encoder_smoke(enc_key, checkpoint, expected_dim, arch=arch)
