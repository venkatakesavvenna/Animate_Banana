import os
import tempfile

import pytest
import torch
from transformers import TrainingArguments

from img_2_svg_pretraining.training.training_core.train.custom_trainer import CustomTrainer
from tests.support.builders import build_vlm_sam_model, load_vlm_data_module, make_synthetic_image
from tests.support.matrix import selected_representative_trainer_specs
from tests.support.runtime import (
    build_runtime_vlm_args_kwargs,
    representative_model_ref_for_spec,
    runtime_enabled,
    trainer_backends_for_spec,
)


def _wandb_logging_enabled() -> bool:
    return os.environ.get("TRAINING_LOG_WANDB", "0") == "1"


def _trainer_cases():
    cases = []
    for spec in selected_representative_trainer_specs():
        for backend in trainer_backends_for_spec(spec, "TRAINER"):
            cases.append((spec, backend))
    return cases


@pytest.mark.integration
@pytest.mark.trainer
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.parametrize(
    ("spec", "attn_implementation"),
    _trainer_cases(),
    ids=[f"{spec.slug}-{backend}" for spec, backend in _trainer_cases()],
)
def test_vlm_sam_custom_trainer_runs_single_gpu_step(
    spec,
    attn_implementation,
    qwen25_model_id,
    gemma3_model_id,
    sam1_checkpoint,
    monkeypatch,
):
    if not runtime_enabled("gpu"):
        pytest.skip("GPU trainer tests disabled")
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
    model = build_vlm_sam_model(
        spec=spec,
        model_ref=model_ref,
        data_module=data_module,
        sam_checkpoint=sam1_checkpoint,
        **build_runtime_vlm_args_kwargs(spec, attn_implementation),
    )

    output_dir = tempfile.mkdtemp(prefix=f"{spec.slug}_trainer_{attn_implementation}_", dir="/tmp")
    logging_dir = os.path.join(output_dir, "logs")

    report_to = "wandb" if _wandb_logging_enabled() else "none"
    if _wandb_logging_enabled():
        os.environ.setdefault("WANDB_PROJECT", "img-2-svg-pretraining-smoke")
        monkeypatch.setenv("WANDB_DIR", output_dir)
        monkeypatch.setenv("WANDB_SILENT", os.environ.get("WANDB_SILENT", "true"))
        monkeypatch.setenv("WANDB_NAME", f"{spec.slug}-trainer-smoke-{attn_implementation}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        remove_unused_columns=False,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
        report_to=report_to,
        run_name=f"{spec.slug}-trainer-smoke-{attn_implementation}",
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
        eval_dataset=None,
        data_collator=data_module.Collator,
        processing_class=data_module.processor,
        compute_metrics=None,
    )

    result = trainer.train()

    assert trainer.state.global_step == 1
    assert result.training_loss > 0
