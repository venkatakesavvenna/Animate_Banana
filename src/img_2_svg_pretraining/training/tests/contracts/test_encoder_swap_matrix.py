from __future__ import annotations

import inspect
from collections import Counter, defaultdict

from img_2_svg_pretraining.training.training_core.builders.build_vlm_sam import build_vlm_sam
from img_2_svg_pretraining.training.training_core.builders.extract_vision_encoder import extract_vision_encoder
from img_2_svg_pretraining.training.training_core.matrix.encoder_swap_matrix import (
    DECODER_SPECS,
    ENCODER_SPECS,
    MATRIX_MODES,
    MATRIX_RUN_SPECS,
    MATRIX_TRAINING_DEFAULTS,
)


def test_builders_accept_explicit_model_family_name():
    build_signature = inspect.signature(build_vlm_sam)
    extract_signature = inspect.signature(extract_vision_encoder)

    assert "model_family_name" in build_signature.parameters
    assert "model_family_name" in extract_signature.parameters


def test_encoder_swap_matrix_expands_to_expected_run_count():
    assert len(DECODER_SPECS) == 4
    assert len(MATRIX_MODES) == 2
    assert len(MATRIX_RUN_SPECS) == 80


def test_matrix_omits_same_decoder_same_extracted_duplicates():
    emitted = {(run.decoder_slug, run.encoder_slug) for run in MATRIX_RUN_SPECS}
    assert ("qwen25", "extracted-qwen25") not in emitted
    assert ("gemma3", "extracted-gemma3") not in emitted
    assert ("molmo7b-o", "extracted-molmo7bo") not in emitted
    assert ("molmo7b-d", "extracted-molmo7bd") not in emitted


def test_matrix_keeps_nine_non_native_rows_per_decoder():
    counts: Counter[str] = Counter()
    for run in MATRIX_RUN_SPECS:
        if run.encoder_slug != "native" and run.mode == "vlm_only":
            counts[run.decoder_slug] += 1

    assert counts == Counter(
        {
            "qwen25": 9,
            "gemma3": 9,
            "molmo7b-o": 9,
            "molmo7b-d": 9,
        }
    )


def test_openvision2_stays_registry_only_and_is_not_emitted():
    encoder_slugs = {encoder.slug for encoder in ENCODER_SPECS}
    emitted_slugs = {run.encoder_slug for run in MATRIX_RUN_SPECS}

    assert "openvision" in encoder_slugs
    assert "openvision2" not in encoder_slugs
    assert "openvision2" not in emitted_slugs


def test_molmo_d_remains_molmovlm_in_matrix():
    run_specs = [run for run in MATRIX_RUN_SPECS if run.decoder_slug == "molmo7b-d"]
    assert run_specs
    assert all(run.vlm_family == "molmovlm" for run in run_specs)
    assert all(run.model_family_name == "molmo" for run in run_specs)


def test_matrix_training_defaults_match_requested_real_run_profile():
    trainer_defaults = MATRIX_TRAINING_DEFAULTS["trainer"]
    dataset_defaults = MATRIX_TRAINING_DEFAULTS["dataset_kwargs"]
    loss_by_mode = MATRIX_TRAINING_DEFAULTS["loss_by_mode"]

    assert dataset_defaults["streaming"] is True
    assert dataset_defaults["sample_limit"] == 8
    assert trainer_defaults["max_steps"] == 3
    assert trainer_defaults["eval_steps"] == 3
    assert trainer_defaults["per_device_train_batch_size"] == 1
    assert loss_by_mode["vlm_only"]["bce_loss_weight"] == 0.0
    assert loss_by_mode["sam1"]["bce_loss_weight"] == 1.0


def test_every_decoder_emits_one_native_row_per_mode():
    grouped: dict[str, set[str]] = defaultdict(set)
    for run in MATRIX_RUN_SPECS:
        if run.encoder_slug == "native":
            grouped[run.decoder_slug].add(run.mode)

    assert grouped == {
        "qwen25": {"vlm_only", "sam1"},
        "gemma3": {"vlm_only", "sam1"},
        "molmo7b-o": {"vlm_only", "sam1"},
        "molmo7b-d": {"vlm_only", "sam1"},
    }


def test_matrix_run_names_are_unique():
    run_names = [run.run_name for run in MATRIX_RUN_SPECS]
    assert len(run_names) == len(set(run_names))
