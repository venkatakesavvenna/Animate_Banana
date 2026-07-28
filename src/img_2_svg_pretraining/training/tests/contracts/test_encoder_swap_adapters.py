from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from img_2_svg_pretraining.training.training_core.data_modules.vlms.common import apply_processor_image_override
from img_2_svg_pretraining.training.training_core.vision_encoders.adapter_utils import (
    qwen_images_to_patch_tokens,
    qwen_patch_tokens_to_images,
    resize_token_sequence,
)
from img_2_svg_pretraining.training.training_core.matrix.encoder_swap_matrix import MATRIX_RUN_SPECS
from scripts.run_encoder_swap_matrix import find_existing_extracted_encoder, prepare_extracted_encoders


def test_apply_processor_image_override_updates_standard_processor_fields():
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            size={"height": 896, "width": 896},
            crop_size={"height": 896, "width": 896},
        )
    )
    apply_processor_image_override(
        processor,
        {
            "image_mean": [0.1, 0.2, 0.3],
            "image_std": [0.9, 0.8, 0.7],
            "image_size": 336,
        },
    )

    image_processor = processor.image_processor
    assert image_processor.image_mean == [0.1, 0.2, 0.3]
    assert image_processor.image_std == [0.9, 0.8, 0.7]
    assert image_processor.size == {"height": 336, "width": 336}
    assert image_processor.crop_size == {"height": 336, "width": 336}


def test_apply_processor_image_override_updates_qwen_pixel_budget_fields():
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            min_pixels=3136,
            max_pixels=12845056,
            size={"shortest_edge": 3136, "longest_edge": 12845056},
        )
    )
    apply_processor_image_override(
        processor,
        {
            "image_mean": [0.2, 0.2, 0.2],
            "image_std": [0.7, 0.7, 0.7],
            "image_size": 112,
        },
    )

    image_processor = processor.image_processor
    assert image_processor.image_mean == [0.2, 0.2, 0.2]
    assert image_processor.image_std == [0.7, 0.7, 0.7]
    assert image_processor.min_pixels == 112 * 112
    assert image_processor.max_pixels == 112 * 112
    assert image_processor.size["shortest_edge"] == 112 * 112
    assert image_processor.size["longest_edge"] == 112 * 112


def test_qwen_patch_roundtrip_restores_original_shape():
    image = torch.arange(3 * 28 * 28, dtype=torch.float32).reshape(1, 3, 28, 28)
    patch_tokens, grid_thw = qwen_images_to_patch_tokens(
        image,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
    )
    reconstructed = qwen_patch_tokens_to_images(
        patch_tokens,
        grid_thw,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
    )

    assert reconstructed.shape == image.shape
    assert torch.allclose(reconstructed, image)


def test_resize_token_sequence_handles_cls_token_and_resizes_grid():
    tokens = torch.randn(24 * 24 + 1, 32)
    resized = resize_token_sequence(tokens, target_h=12, target_w=12)
    assert resized.shape == (12 * 12, 32)


def test_find_existing_extracted_encoder_prefers_latest_run(tmp_path: Path):
    old_path = tmp_path / "20260520T100000Z" / "extracted_encoders" / "extracted-qwen25"
    new_path = tmp_path / "20260520T200000Z" / "extracted_encoders" / "extracted-qwen25"
    (old_path / "module.pt").parent.mkdir(parents=True, exist_ok=True)
    (new_path / "module.pt").parent.mkdir(parents=True, exist_ok=True)
    (old_path / "module.pt").touch()
    (old_path / "preprocessor.json").touch()
    (new_path / "module.pt").touch()
    (new_path / "preprocessor.json").touch()

    assert find_existing_extracted_encoder(tmp_path, "extracted-qwen25") == new_path


def test_find_existing_extracted_encoder_requires_preprocessor_sidecar(tmp_path: Path):
    incomplete = tmp_path / "20260520T100000Z" / "extracted_encoders" / "extracted-qwen25"
    complete = tmp_path / "20260520T200000Z" / "extracted_encoders" / "extracted-qwen25"
    incomplete.mkdir(parents=True, exist_ok=True)
    complete.mkdir(parents=True, exist_ok=True)
    (incomplete / "module.pt").touch()
    (complete / "module.pt").touch()
    (complete / "preprocessor.json").touch()

    assert find_existing_extracted_encoder(tmp_path, "extracted-qwen25") == complete


def test_prepare_extracted_encoders_skips_work_when_subset_has_no_extracted_rows(tmp_path: Path):
    native_only_run = next(spec for spec in MATRIX_RUN_SPECS if spec.encoder_kind == "native")
    paths = prepare_extracted_encoders(
        env={},
        extracted_root=tmp_path / "extracted",
        selected_runs=[native_only_run],
        force_reextract=False,
    )
    assert paths == {}
