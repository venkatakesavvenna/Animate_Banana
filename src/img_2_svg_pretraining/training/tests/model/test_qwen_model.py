import os

import pytest
import torch

from img_2_svg_pretraining.training.training_core.registry.registry import VLMModelRegistry
from tests.support.builders import build_vlm_args
from tests.support.matrix import QWEN25_SPEC, resolve_model_ref, selected_qwen_specs
from tests.support.runtime import runtime_enabled


def _large_model_tests_enabled() -> bool:
    return os.environ.get("TRAINING_RUN_LARGE_MODEL_TESTS", "0") == "1"


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.parametrize("spec", selected_qwen_specs(), ids=lambda spec: spec.slug)
def test_qwen_wrapper_loads_actual_model(spec):
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    model_ref = resolve_model_ref(spec)
    if not model_ref or not _large_model_tests_enabled():
        pytest.skip(f"No enabled checkpoint for {spec.slug}")

    model = VLMModelRegistry.get_model(
        spec.family,
        vlm_args=build_vlm_args(spec=spec, model_ref=model_ref),
    ).to(torch.device("cuda:0"))

    assert model.hidden_size > 0
    assert hasattr(model, "qwen")
    assert next(model.parameters()).is_cuda


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
def test_qwen_representative_wrapper_loads_cached_model():
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    model_ref = resolve_model_ref(QWEN25_SPEC)
    if not model_ref or not _large_model_tests_enabled():
        pytest.skip("Representative Qwen2.5 checkpoint unavailable")

    model = VLMModelRegistry.get_model(
        QWEN25_SPEC.family,
        vlm_args=build_vlm_args(spec=QWEN25_SPEC, model_ref=model_ref),
    ).to(torch.device("cuda:0"))

    assert model.hidden_size > 0
    assert model.model_type == "qwen2.5vl"
    assert next(model.parameters()).is_cuda
