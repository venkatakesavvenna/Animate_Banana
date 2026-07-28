from __future__ import annotations

from pathlib import Path

import pytest
import torch

from img_2_svg_pretraining.training.training_core.data_modules.qwen.qwen_data import IGNORE_INDEX
from img_2_svg_pretraining.training.training_core.models.sam.base import SAMModelBase
from img_2_svg_pretraining.training.training_core.models.vlm_sam import VLMSam
from img_2_svg_pretraining.training.training_core.models.vlms.gemma.gemma_model import GemmaModel
from img_2_svg_pretraining.training.training_core.models.vlms.qwen.qwen_model import QwenModel
from img_2_svg_pretraining.training.training_core.registry.utils import ModelConfig, SamModelArguments, VLMArguments
from tests.support.builders import load_vlm_data_module, make_synthetic_image
from tests.support.matrix import GEMMA3_SPEC, QWEN2_SPEC
from tests.support.runtime import move_batch_to_device, runtime_enabled


class FakeGenerateSamHead(SAMModelBase):
    @property
    def prompt_embed_dim(self) -> int:
        return 256

    def forward(
        self,
        images,
        masks,
        vlm_seg_hidden_states,
        resize_list,
        label_list,
        mask_counts=None,
    ):
        outputs = []
        for hidden_states, label_size in zip(vlm_seg_hidden_states, label_list):
            if isinstance(label_size, torch.Tensor):
                height, width = [int(value) for value in label_size.detach().cpu().tolist()]
            else:
                height, width = [int(value) for value in label_size]
            outputs.append(
                torch.zeros(
                    (hidden_states.shape[0], height, width),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            )
        return outputs, None, None


def _truncate_batch_to_prompt(batch: dict) -> dict:
    prompt_lengths = []
    for labels in batch["labels"]:
        prompt_lengths.append(int((labels == IGNORE_INDEX).sum().item()))

    max_prompt_length = max(prompt_lengths)
    truncated = {
        "input_ids": batch["input_ids"][:, :max_prompt_length],
        "attention_mask": batch["attention_mask"][:, :max_prompt_length],
        "images": batch["images"],
        "resize_list": batch["resize_list"],
        "orig_image_size_list": batch["orig_image_size_list"],
    }
    if "pixel_values" in batch:
        truncated["pixel_values"] = batch["pixel_values"]
    if "image_grid_thw" in batch:
        truncated["image_grid_thw"] = batch["image_grid_thw"]
    if "position_ids" in batch:
        truncated["position_ids"] = batch["position_ids"][:, :, :max_prompt_length]
    if "token_type_ids" in batch:
        truncated["token_type_ids"] = batch["token_type_ids"][:, :max_prompt_length]
    return truncated


def test_extract_generated_hidden_states_skips_prefill_and_pads_last_token():
    batch_size = 1
    hidden_size = 4
    target_length = 3
    hidden_states = (
        (torch.zeros((batch_size, 5, hidden_size)),),
        (torch.full((batch_size, 1, hidden_size), 1.0),),
        (torch.full((batch_size, 1, hidden_size), 2.0),),
    )

    extracted = VLMSam._extract_generated_hidden_states(
        hidden_states=hidden_states,
        target_length=target_length,
        batch_size=batch_size,
        hidden_size=hidden_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert extracted.shape == (1, 3, 4)
    assert torch.allclose(extracted[0, 0], torch.ones(hidden_size))
    assert torch.allclose(extracted[0, 1], torch.full((hidden_size,), 2.0))
    assert torch.allclose(extracted[0, 2], torch.zeros(hidden_size))


@pytest.mark.integration
@pytest.mark.composite
@pytest.mark.external
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("spec", "model_ref", "model_cls", "attn_implementation"),
    [
        pytest.param(QWEN2_SPEC, "qwen2_model_id", QwenModel, "eager", id="qwen2vl-generate"),
        pytest.param(GEMMA3_SPEC, "gemma3_model_id", GemmaModel, "eager", id="gemma3-generate"),
    ],
)
def test_vlm_sam_generate_returns_text_and_mask_batch_for_real_vlm(
    spec,
    model_ref,
    model_cls,
    attn_implementation,
    request,
):
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    resolved_model_ref = request.getfixturevalue(model_ref)
    if not resolved_model_ref:
        pytest.skip(f"{spec.slug} checkpoint unavailable")
    if not Path(str(resolved_model_ref)).exists():
        pytest.skip(f"{spec.slug} local checkpoint path unavailable: {resolved_model_ref}")

    data_module = load_vlm_data_module(
        spec=spec,
        model_ref=resolved_model_ref,
        image=make_synthetic_image((64, 64)),
        sam_image_size=128,
    )
    sample = data_module.Dataloader[0]
    batch = data_module.Collator([sample])
    prompt_batch = _truncate_batch_to_prompt(batch)
    device = torch.device("cuda:0")
    prompt_batch = move_batch_to_device(prompt_batch, device)

    backbone = model_cls(
        vlm_args=VLMArguments(
            family=spec.family,
            model_name_or_path=resolved_model_ref,
            model_family_name=spec.model_family_name,
            attn_implementation=attn_implementation,
            tune_mm_llm=False,
            tune_mm_mlp=False,
            tune_mm_vision=False,
            tune_mm_lm_head=False,
        ),
        gradient_checkpointing=False,
        bf16=False,
    ).to(device)
    model = VLMSam(
        config=ModelConfig(
            sam_args=SamModelArguments(
                tune_image_encoder=False,
                tune_prompt_encoder=False,
                tune_mask_decoder=True,
                checkpoint="unused",
                version="sam1",
            ),
            vlm_args=VLMArguments(
                family=spec.family,
                model_name_or_path=resolved_model_ref,
                model_family_name=spec.model_family_name,
                attn_implementation=attn_implementation,
            ),
            vlm_family=spec.family,
            sam_version="sam1",
        ),
        data_module=data_module,
        backbone=backbone,
        sam_head=FakeGenerateSamHead(),
    ).to(device)

    preds, pred_masks = model.generate(**prompt_batch, max_new_tokens=4)

    assert isinstance(preds, list)
    assert len(preds) == 1
    assert isinstance(pred_masks, list)
    assert len(pred_masks) == 1
    assert pred_masks[0].ndim == 3
    assert pred_masks[0].shape[-2:] == torch.Size((64, 64))
