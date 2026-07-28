import os
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import TrainingArguments

from img_2_svg_pretraining.training.training_core.data_modules.qwen.qwen_data import IGNORE_INDEX
from img_2_svg_pretraining.training.training_core.train.custom_trainer import CustomTrainer
from img_2_svg_pretraining.training.training_core.validation.compute_metrics import ComputeMetrics
from tests.support.builders import build_vlm_sam_model, load_vlm_data_module, make_synthetic_image
from tests.support.matrix import selected_representative_trainer_specs
from tests.support.runtime import (
    build_runtime_vlm_args_kwargs,
    representative_model_ref_for_spec,
    runtime_enabled,
    trainer_backends_for_spec,
)


def _eval_cases():
    cases = []
    for spec in selected_representative_trainer_specs():
        backends = trainer_backends_for_spec(spec, "TRAINER")
        if backends:
            cases.append((spec, backends[0]))
    return cases


@pytest.mark.integration
@pytest.mark.trainer
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.slow
@pytest.mark.parametrize(
    ("spec", "attn_implementation"),
    _eval_cases(),
    ids=[f"{spec.slug}-eval-{backend}" for spec, backend in _eval_cases()],
)
def test_vlm_sam_custom_trainer_writes_eval_visualizations(
    spec,
    attn_implementation,
    qwen25_model_id,
    gemma3_model_id,
    molmo_d_model_id,
    sam1_checkpoint,
):
    if not runtime_enabled("gpu"):
        pytest.skip("GPU trainer tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not sam1_checkpoint:
        pytest.skip("SAM1 checkpoint unavailable")

    model_ref = representative_model_ref_for_spec(spec, qwen25_model_id, gemma3_model_id, molmo_d_model_id)
    if not model_ref:
        pytest.skip(f"Representative checkpoint unavailable for {spec.slug}")

    images = [
        make_synthetic_image((256, 256), color="white"),
        make_synthetic_image((256, 256), color="lightgray"),
    ]
    data_module = load_vlm_data_module(
        spec=spec,
        model_ref=model_ref,
        image=images,
        sam_image_size=1024,
    )
    model = build_vlm_sam_model(
        spec=spec,
        model_ref=model_ref,
        data_module=data_module,
        sam_checkpoint=sam1_checkpoint,
        **build_runtime_vlm_args_kwargs(spec, attn_implementation),
    )

    output_dir = tempfile.mkdtemp(prefix=f"{spec.slug}_trainer_eval_{attn_implementation}_", dir="/tmp")
    logging_dir = os.path.join(output_dir, "logs")
    run_name = f"{spec.slug}-trainer-eval-{attn_implementation}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        remove_unused_columns=False,
        max_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        run_name=run_name,
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        disable_tqdm=True,
    )
    training_args._n_gpu = 1

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module.Dataloader,
        eval_dataset=data_module.Dataloader,
        data_collator=data_module.Collator,
        processing_class=data_module.processor,
        compute_metrics=ComputeMetrics(
            data_module.processor.tokenizer,
            IGNORE_INDEX,
            lambda _text: ["title"],
            ["title"],
        ),
    )

    result = trainer.train()

    artifact_root = Path(logging_dir) / "1" / "0"
    saved_pngs = sorted(artifact_root.glob("*.png"))

    assert trainer.state.global_step == 1
    assert result.training_loss > 0
    assert saved_pngs, f"No eval visualizations were written under {artifact_root}"
