from __future__ import annotations

import pytest
import torch

from img_2_svg_pretraining.training.training_core.registry.registry import VLMModelRegistry
from tests.support.builders import build_vlm_args, build_vlm_sam_model, load_vlm_data_module, make_synthetic_image
from tests.support.matrix import MOLMO_1B_SPEC, MOLMO_D_SPEC, MOLMO_O_SPEC, resolve_model_ref, resolve_sam1_checkpoint
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
def test_molmo_d_wrapper_loads(molmo_d_model_id):
    _skip_if_unavailable(molmo_d_model_id, "MOLMO_D")

    model = VLMModelRegistry.get_model(
        MOLMO_D_SPEC.family,
        vlm_args=build_vlm_args(spec=MOLMO_D_SPEC, model_ref=molmo_d_model_id),
        gradient_checkpointing=False,
    ).to(torch.device("cuda:0"))

    assert hasattr(model, "molmo")
    assert model.hidden_size > 0
    assert next(model.parameters()).is_cuda


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
def test_molmo_o_wrapper_loads(molmo_o_model_id):
    _skip_if_unavailable(molmo_o_model_id, "MOLMO_O")

    model = VLMModelRegistry.get_model(
        MOLMO_O_SPEC.family,
        vlm_args=build_vlm_args(spec=MOLMO_O_SPEC, model_ref=molmo_o_model_id),
        gradient_checkpointing=False,
    ).to(torch.device("cuda:0"))

    assert hasattr(model, "molmo")
    assert model.hidden_size > 0
    assert next(model.parameters()).is_cuda


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
def test_molmo_1b_wrapper_loads(molmo_1b_model_id):
    _skip_if_unavailable(molmo_1b_model_id, "MOLMO_1B")

    model = VLMModelRegistry.get_model(
        MOLMO_1B_SPEC.family,
        vlm_args=build_vlm_args(spec=MOLMO_1B_SPEC, model_ref=molmo_1b_model_id),
        gradient_checkpointing=False,
    ).to(torch.device("cuda:0"))

    assert hasattr(model, "molmo")
    assert model.hidden_size > 0
    assert next(model.parameters()).is_cuda


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
def test_molmo_model_forward_loss(molmo_d_model_id):
    _skip_if_unavailable(molmo_d_model_id, "MOLMO_D")
    sam1_checkpoint = resolve_sam1_checkpoint()
    if not sam1_checkpoint:
        pytest.skip("SAM1 checkpoint unavailable")

    device = torch.device("cuda:0")
    data_module = load_vlm_data_module(
        spec=MOLMO_D_SPEC,
        model_ref=molmo_d_model_id,
        image=make_synthetic_image(),
    )
    batch = data_module.Collator([data_module.Dataloader[0]])

    model = VLMModelRegistry.get_model(
        MOLMO_D_SPEC.family,
        vlm_args=build_vlm_args(spec=MOLMO_D_SPEC, model_ref=molmo_d_model_id),
        gradient_checkpointing=False,
    ).to(device)
    model.resize_token_embeddings(data_module.tokenizer_vocab_size)

    molmo_keys = {"input_ids", "attention_mask", "labels", "molmo_images", "image_input_idx", "image_masks"}
    fwd_kwargs = {k: v.to(device) for k, v in batch.items() if k in molmo_keys and torch.is_tensor(v)}
    # Rename back as MolmoModel.forward() expects
    if "molmo_images" in fwd_kwargs:
        fwd_kwargs["images"] = fwd_kwargs.pop("molmo_images")

    output = model(**fwd_kwargs, output_hidden_states=True, return_dict=True)
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
