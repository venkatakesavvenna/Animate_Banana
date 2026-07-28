import pytest
import torch

from img_2_svg_pretraining.training.training_core.registry.registry import VLMModelRegistry
from tests.support.builders import build_vlm_args
from tests.support.matrix import GEMMA3_SPEC
from tests.support.runtime import runtime_enabled


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
def test_gemma_wrapper_loads_actual_model(gemma3_model_id):
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    model = VLMModelRegistry.get_model(
        GEMMA3_SPEC.family,
        vlm_args=build_vlm_args(spec=GEMMA3_SPEC, model_ref=gemma3_model_id),
    ).to(torch.device("cuda:0"))

    assert model.hidden_size > 0
    assert hasattr(model, "gemma")
    assert next(model.parameters()).is_cuda
