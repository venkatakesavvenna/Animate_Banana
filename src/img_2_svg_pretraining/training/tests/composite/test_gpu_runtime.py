import pytest
import torch

from tests.support.builders import build_vlm_sam_model, load_vlm_data_module, make_synthetic_image
from tests.support.matrix import selected_representative_trainer_specs
from tests.support.runtime import (
    build_runtime_vlm_args_kwargs,
    gpu_wrapper_backends_for_spec,
    move_batch_to_device,
    representative_model_ref_for_spec,
    runtime_enabled,
)


def _gpu_composite_cases():
    cases = []
    for spec in selected_representative_trainer_specs():
        for backend in gpu_wrapper_backends_for_spec(spec):
            cases.append((spec, backend))
    return cases


@pytest.mark.integration
@pytest.mark.composite
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.parametrize(
    ("spec", "attn_implementation"),
    _gpu_composite_cases(),
    ids=[f"{spec.slug}-{backend}" for spec, backend in _gpu_composite_cases()],
)
def test_vlm_sam_runs_single_gpu_train_step(
    spec,
    attn_implementation,
    qwen25_model_id,
    gemma3_model_id,
    sam1_checkpoint,
):
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not sam1_checkpoint:
        pytest.skip("SAM1 checkpoint unavailable")

    model_ref = representative_model_ref_for_spec(spec, qwen25_model_id, gemma3_model_id)
    if not model_ref:
        pytest.skip(f"Representative checkpoint unavailable for {spec.slug}")

    data_module = load_vlm_data_module(
        spec=spec,
        model_ref=model_ref,
        image=make_synthetic_image((256, 256)),
        sam_image_size=1024,
    )
    batch = data_module.Collator([data_module.Dataloader[0]])

    device = torch.device("cuda:0")
    model = build_vlm_sam_model(
        spec=spec,
        model_ref=model_ref,
        data_module=data_module,
        sam_checkpoint=sam1_checkpoint,
        **build_runtime_vlm_args_kwargs(spec, attn_implementation),
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**move_batch_to_device(batch, device))
    outputs["loss"].backward()
    optimizer.step()

    assert float(outputs["loss"].detach().cpu()) > 0
    assert float(outputs["mask_loss"].detach().cpu()) > 0
