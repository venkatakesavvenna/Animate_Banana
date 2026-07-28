import os
import tempfile
from pathlib import Path

import sys

import torch
from transformers import TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from img_2_svg_pretraining.training.training_core.train.custom_trainer import CustomTrainer
from tests.support.builders import build_vlm_sam_model, load_vlm_data_module, make_synthetic_image
from tests.support.matrix import VLMCheckpointSpec
from tests.support.runtime import build_runtime_vlm_args_kwargs


def _worker_spec() -> VLMCheckpointSpec:
    return VLMCheckpointSpec(
        slug=os.environ["TRAINING_TEST_MODEL_FAMILY_NAME"],
        family=os.environ["TRAINING_TEST_VLM_FAMILY"],
        model_family_name=os.environ["TRAINING_TEST_MODEL_FAMILY_NAME"],
        env_var="TRAINING_TEST_MODEL_REF",
    )


def main():
    spec = _worker_spec()
    model_ref = os.environ["TRAINING_TEST_MODEL_REF"]
    sam_checkpoint = os.environ["TRAINING_TEST_SAM1_CHECKPOINT"]
    attn_implementation = os.environ.get("TRAINING_TEST_TRAINER_ATTN", "eager")
    mode = os.environ["TRAINING_TEST_TRAINER_MODE"]
    dataset_size = int(os.environ.get("TRAINING_TEST_TRAINER_DATASET_SIZE", "1"))

    images = [make_synthetic_image((256, 256), color="white") for _ in range(dataset_size)]
    if dataset_size > 1:
        images[1] = make_synthetic_image((256, 256), color="lightgray")

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
        sam_checkpoint=sam_checkpoint,
        **build_runtime_vlm_args_kwargs(spec, attn_implementation),
    )

    output_dir = tempfile.mkdtemp(prefix=f"{spec.slug}_{mode}_smoke_", dir="/tmp")
    training_arg_kwargs = {
        "output_dir": output_dir,
        "logging_dir": os.path.join(output_dir, "logs"),
        "remove_unused_columns": False,
        "max_steps": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "eval_strategy": "no",
        "save_strategy": "no",
        "logging_steps": 1,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "disable_tqdm": True,
        "bf16": torch.cuda.is_bf16_supported(),
    }
    if mode == "ddp":
        training_arg_kwargs["ddp_find_unused_parameters"] = False

    training_args = TrainingArguments(
        **training_arg_kwargs,
    )

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

    if mode == "ddp":
        if trainer.is_world_process_zero():
            print("DDP_RESULT=passed")
            print(f"DDP_GLOBAL_STEP={trainer.state.global_step}")
            print(f"DDP_TRAIN_LOSS={result.training_loss}")
    else:
        print("DP_RESULT=passed")
        print(f"DP_N_GPU={trainer.args.n_gpu}")
        print(f"DP_GLOBAL_STEP={trainer.state.global_step}")
        print(f"DP_TRAIN_LOSS={result.training_loss}")


if __name__ == "__main__":
    main()
