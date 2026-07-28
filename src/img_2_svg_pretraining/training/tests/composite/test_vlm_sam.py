import os

import pytest
import torch

from img_2_svg_pretraining.training.training_core.registry.utils import DataModule
from tests.support.builders import build_vlm_sam_model, load_vlm_data_module, make_synthetic_image
from tests.support.matrix import GEMMA3_SPEC, MOLMO_D_SPEC, QWEN25_SPEC, resolve_model_ref, resolve_sam1_checkpoint
from tests.support.runtime import build_runtime_vlm_args_kwargs, runtime_enabled


def _large_model_tests_enabled() -> bool:
    return os.environ.get("TRAINING_RUN_LARGE_MODEL_TESTS", "0") == "1"


@pytest.mark.integration
@pytest.mark.composite
@pytest.mark.gpu
@pytest.mark.external
def test_vlm_sam_constructs_with_representative_qwen_and_sam1(sam1_checkpoint):
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    model_ref = resolve_model_ref(QWEN25_SPEC)
    if not model_ref or not sam1_checkpoint or not _large_model_tests_enabled():
        pytest.skip("Representative Qwen2.5 or SAM1 checkpoint unavailable")

    data_module = DataModule(
        model_name=QWEN25_SPEC.model_family_name,
        model_path=model_ref,
        processor=type("Processor", (), {"tokenizer": type("Tokenizer", (), {"eos_token_id": 0})()})(),
        seg_token_idx=0,
        ignore_idx=-100,
        Dataloader=None,
        Collator=None,
        layout_classes=[],
        family_name=QWEN25_SPEC.family,
    )

    model = build_vlm_sam_model(
        spec=QWEN25_SPEC,
        model_ref=model_ref,
        data_module=data_module,
        sam_checkpoint=sam1_checkpoint,
        **build_runtime_vlm_args_kwargs(QWEN25_SPEC, "flash_attention_2"),
    ).to(torch.device("cuda:0"))

    assert model.backbone.hidden_size > 0
    assert model.sam_head.prompt_embed_dim > 0
    assert next(model.parameters()).is_cuda


@pytest.mark.integration
@pytest.mark.composite
@pytest.mark.gpu
@pytest.mark.external
def test_vlm_sam_runs_gemma_backbone_forward_with_sam_batch(gemma3_model_id, sam1_checkpoint):
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not sam1_checkpoint:
        pytest.skip("SAM1 checkpoint unavailable")

    data_module = load_vlm_data_module(
        spec=GEMMA3_SPEC,
        model_ref=gemma3_model_id,
        image=make_synthetic_image(),
    )
    batch = data_module.Collator([data_module.Dataloader[0]])

    device = torch.device("cuda:0")
    model = build_vlm_sam_model(
        spec=GEMMA3_SPEC,
        model_ref=gemma3_model_id,
        data_module=data_module,
        sam_checkpoint=sam1_checkpoint,
    ).to(device)

    outputs = model.backbone(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
        pixel_values=batch["pixel_values"].to(device),
        token_type_ids=batch["token_type_ids"].to(device),
        output_hidden_states=True,
        return_dict=True,
    )

    assert outputs.loss is not None
    assert outputs.hidden_states is not None


@pytest.mark.integration
@pytest.mark.composite
@pytest.mark.gpu
@pytest.mark.external
def test_vlm_sam_constructs_with_molmo_d_and_sam1(molmo_d_model_id, sam1_checkpoint):
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not molmo_d_model_id:
        pytest.skip("MOLMO_D checkpoint not set or path does not exist")
    if not sam1_checkpoint:
        pytest.skip("SAM1 checkpoint unavailable")

    device = torch.device("cuda:0")
    data_module = load_vlm_data_module(
        spec=MOLMO_D_SPEC,
        model_ref=molmo_d_model_id,
        image=make_synthetic_image((256, 256)),
        sam_image_size=1024,
    )
    batch = data_module.Collator([data_module.Dataloader[0]])

    model = build_vlm_sam_model(
        spec=MOLMO_D_SPEC,
        model_ref=molmo_d_model_id,
        data_module=data_module,
        sam_checkpoint=sam1_checkpoint,
    ).to(device)

    batch_device = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }
    outputs = model(**batch_device)

    assert "loss" in outputs
    assert torch.isfinite(outputs["loss"])
    assert model.backbone.hidden_size > 0
    assert model.sam_head.prompt_embed_dim > 0
