import pytest
import torch

from img_2_svg_pretraining.training.training_core.registry.registry import VLMModelRegistry
from tests.support.builders import build_vlm_args, load_vlm_data_module, make_synthetic_image
from tests.support.matrix import selected_representative_trainer_specs
from tests.support.runtime import (
    build_runtime_vlm_args_kwargs,
    gpu_wrapper_backends_for_spec,
    representative_model_ref_for_spec,
    runtime_enabled,
    select_vlm_batch,
)


def _gpu_wrapper_cases():
    cases = []
    for spec in selected_representative_trainer_specs():
        for backend in gpu_wrapper_backends_for_spec(spec):
            cases.append((spec, backend))
    return cases


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.parametrize(
    ("spec", "attn_implementation"),
    _gpu_wrapper_cases(),
    ids=[f"{spec.slug}-{backend}" for spec, backend in _gpu_wrapper_cases()],
)
def test_vlm_wrapper_runs_single_gpu_train_step(
    spec,
    attn_implementation,
    qwen25_model_id,
    gemma3_model_id,
):
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    model_ref = representative_model_ref_for_spec(spec, qwen25_model_id, gemma3_model_id)
    if not model_ref:
        pytest.skip(f"Representative checkpoint unavailable for {spec.slug}")

    data_module = load_vlm_data_module(spec, model_ref, make_synthetic_image())
    batch = data_module.Collator([data_module.Dataloader[0]])

    device = torch.device("cuda:0")
    model = VLMModelRegistry.get_model(
        spec.family,
        vlm_args=build_vlm_args(
            spec=spec,
            model_ref=model_ref,
            **build_runtime_vlm_args_kwargs(spec, attn_implementation),
        ),
        gradient_checkpointing=False,
        bf16=torch.cuda.is_bf16_supported(),
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**select_vlm_batch(spec, batch, device))
    outputs.loss.backward()
    optimizer.step()

    assert float(outputs.loss.detach().cpu()) > 0
