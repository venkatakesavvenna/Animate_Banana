import os
import subprocess
import sys

import pytest
import torch

from tests.support.matrix import selected_representative_trainer_specs
from tests.support.runtime import (
    representative_model_ref_for_spec,
    runtime_enabled,
    trainer_backend_env_var,
    trainer_backends_for_spec,
)


def _ddp_cases():
    return [(spec, trainer_backends_for_spec(spec, "TRAINER")[0]) for spec in selected_representative_trainer_specs()]


@pytest.mark.integration
@pytest.mark.trainer
@pytest.mark.gpu
@pytest.mark.ddp
@pytest.mark.external
@pytest.mark.parametrize(
    ("spec", "attn_implementation"),
    _ddp_cases(),
    ids=[f"{spec.slug}-{backend}" for spec, backend in _ddp_cases()],
)
def test_vlm_sam_custom_trainer_runs_two_gpu_ddp_step(
    spec,
    attn_implementation,
    qwen25_model_id,
    gemma3_model_id,
    sam1_checkpoint,
):
    if not runtime_enabled("ddp"):
        pytest.skip("DDP trainer tests disabled")
    if torch.cuda.device_count() < 2:
        pytest.skip("Fewer than 2 CUDA devices available")
    if not sam1_checkpoint:
        pytest.skip("SAM1 checkpoint unavailable")

    model_ref = representative_model_ref_for_spec(spec, qwen25_model_id, gemma3_model_id)
    if not model_ref:
        pytest.skip(f"Representative checkpoint unavailable for {spec.slug}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
    env["HF_HOME"] = os.environ.get("HF_HOME", "/tmp/training_hf_tests")
    env["HUGGINGFACE_HUB_CACHE"] = os.environ.get("HUGGINGFACE_HUB_CACHE", "/tmp/training_hf_tests/hub")
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "/code/src")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TRAINING_TEST_VLM_FAMILY"] = spec.family
    env["TRAINING_TEST_MODEL_FAMILY_NAME"] = spec.model_family_name
    env["TRAINING_TEST_MODEL_REF"] = model_ref
    env["TRAINING_TEST_SAM1_CHECKPOINT"] = sam1_checkpoint
    env["TRAINING_TEST_TRAINER_MODE"] = "ddp"
    env["TRAINING_TEST_TRAINER_DATASET_SIZE"] = "1"
    env["TRAINING_TEST_TRAINER_ATTN"] = attn_implementation
    env[trainer_backend_env_var(spec, "DDP")] = attn_implementation

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "/code/tests/trainer/vlm_trainer_worker.py",
        ],
        cwd="/code",
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "DDP_RESULT=passed" in result.stdout
    assert "DDP_GLOBAL_STEP=1" in result.stdout
