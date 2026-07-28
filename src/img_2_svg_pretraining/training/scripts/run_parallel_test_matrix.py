from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from discover_test_checkpoints import discover_checkpoints


DEFAULT_CONTAINER = "img-2-svg-pretraining-singlenode-venkat.kesav"
DEFAULT_CACHE_ROOT = "/fsxvision_new/anirudh.srinivasan/hf_cache"
ALL_VLM_SPECS = "qwen2vl,qwen2.5vl,qwen3vl,qwen3vl-moe,gemma3,molmo7b-d"
DEFAULT_GPU_IDS = ("0", "1", "2", "3", "4", "5", "6", "7")
DEFAULT_CPU_THREADS_PER_JOB = 4


@dataclass(frozen=True)
class Job:
    name: str
    pytest_targets: tuple[str, ...]
    pytest_args: tuple[str, ...]
    runtime: str
    gpu_slots: int = 0
    priority: int = 0
    env_overrides: tuple[tuple[str, str], ...] = ()


@dataclass
class RunningJob:
    job: Job
    process: subprocess.Popen
    log_path: Path
    gpu_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the training test matrix with dynamic CPU/GPU scheduling.")
    parser.add_argument("--container", default=DEFAULT_CONTAINER, help="Docker container name running the repo.")
    parser.add_argument(
        "--log-dir",
        default="",
        help="Optional host-side log directory. Defaults to logs/test_matrix/<timestamp>.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=",".join(DEFAULT_GPU_IDS),
        help="Comma-separated GPU ids to use for the full-node matrix. Defaults to all 8 local GPUs.",
    )
    parser.add_argument(
        "--cpu-threads-per-job",
        type=int,
        default=DEFAULT_CPU_THREADS_PER_JOB,
        help="CPU thread budget exported into each test job.",
    )
    return parser.parse_args()


def build_common_env(cpu_threads_per_job: int) -> dict[str, str]:
    discovered, missing = discover_checkpoints()
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(f"Missing checkpoint paths for parallel matrix: {missing_str}")

    env = {
        "HF_HOME": DEFAULT_CACHE_ROOT,
        "HUGGINGFACE_HUB_CACHE": f"{DEFAULT_CACHE_ROOT}/hub",
        "HF_ENABLE_PARALLEL_LOADING": "true",
        "HF_PARALLEL_LOADING_WORKERS": str(max(4, cpu_threads_per_job)),
        "TRAINING_TEST_INCLUDE_EXTERNAL": "1",
        "TRAINING_RUN_LARGE_MODEL_TESTS": "1",
        "TRAINING_TEST_SELECTED_VLM_SPECS": ALL_VLM_SPECS,
        "PYTHONPATH": "/code/src",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(cpu_threads_per_job),
        "MKL_NUM_THREADS": str(cpu_threads_per_job),
        "NUMEXPR_NUM_THREADS": str(cpu_threads_per_job),
    }
    env.update(discovered)
    return env


def parse_gpu_ids(raw_value: str) -> tuple[str, ...]:
    gpu_ids = tuple(value.strip() for value in raw_value.split(",") if value.strip())
    if len(gpu_ids) < 8:
        raise SystemExit(f"Expected 8 GPU ids for the full-node matrix, got {len(gpu_ids)} from {raw_value!r}")
    return gpu_ids[:8]


def build_jobs() -> list[Job]:
    return [
        Job(
            name="cpu-contracts",
            pytest_targets=("tests/contracts/test_contracts.py",),
            pytest_args=("-rs", "-vv"),
            runtime="cpu",
            gpu_slots=0,
            priority=0,
        ),
        Job(
            name="cpu-parallel-matrix-contracts",
            pytest_targets=("tests/contracts/test_parallel_test_matrix.py",),
            pytest_args=("-rs", "-vv"),
            runtime="cpu",
            gpu_slots=0,
            priority=0,
        ),
        Job(
            name="cpu-sam-data",
            pytest_targets=("tests/data/test_sam_data.py",),
            pytest_args=("-rs", "-vv"),
            runtime="cpu",
            gpu_slots=0,
            priority=0,
        ),
        Job(
            name="cpu-qwen-data",
            pytest_targets=("tests/data/test_qwen_data.py",),
            pytest_args=("-rs", "-vv"),
            runtime="cpu",
            gpu_slots=0,
            priority=0,
        ),
        Job(
            name="cpu-gemma-data",
            pytest_targets=("tests/data/test_gemma_data.py",),
            pytest_args=("-rs", "-vv"),
            runtime="cpu",
            gpu_slots=0,
            priority=0,
        ),
        Job(
            name="cpu-molmo-data",
            pytest_targets=("tests/data/test_molmo_data.py",),
            pytest_args=("-rs", "-vv"),
            runtime="cpu",
            gpu_slots=0,
            priority=0,
        ),
        Job(
            name="cpu-data-matrix-and-stubs",
            pytest_targets=(
                "tests/data/test_data_matrix.py",
                "tests/composite/test_stubbed_composite.py",
            ),
            pytest_args=("-rs", "-vv"),
            runtime="cpu",
            gpu_slots=0,
            priority=0,
        ),
        Job(
            name="gpu-qwen3-load",
            pytest_targets=("tests/model/test_qwen_model.py::test_qwen_wrapper_loads_actual_model[qwen3vl]",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=0,
        ),
        Job(
            name="gpu-qwen3-moe-load",
            pytest_targets=("tests/model/test_qwen_model.py::test_qwen_wrapper_loads_actual_model[qwen3vl-moe]",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=0,
        ),
        Job(
            name="gpu-qwen-eval-artifacts",
            pytest_targets=("tests/trainer/test_vlm_trainer_eval_artifacts.py",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=0,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "qwen2.5vl"),),
        ),
        Job(
            name="gpu-gemma-eval-artifacts",
            pytest_targets=("tests/trainer/test_vlm_trainer_eval_artifacts.py",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=0,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "gemma3"),),
        ),
        Job(
            name="gpu-molmo-eval-artifacts",
            pytest_targets=("tests/trainer/test_vlm_trainer_eval_artifacts.py",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=0,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "molmo7b-d"),),
        ),
        Job(
            name="gpu-qwen-trainer-step",
            pytest_targets=("tests/trainer/test_vlm_trainer_gpu.py",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=1,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "qwen2.5vl"),),
        ),
        Job(
            name="gpu-gemma-trainer-step",
            pytest_targets=("tests/trainer/test_vlm_trainer_gpu.py",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=1,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "gemma3"),),
        ),
        Job(
            name="gpu-qwen-wrapper-step",
            pytest_targets=("tests/model/test_gpu_vlm_wrappers.py",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=1,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "qwen2.5vl"),),
        ),
        Job(
            name="gpu-gemma-wrapper-step",
            pytest_targets=("tests/model/test_gpu_vlm_wrappers.py",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=1,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "gemma3"),),
        ),
        Job(
            name="gpu-qwen-composite-step",
            pytest_targets=("tests/composite/test_gpu_runtime.py",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=1,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "qwen2.5vl"),),
        ),
        Job(
            name="gpu-gemma-composite-step",
            pytest_targets=("tests/composite/test_gpu_runtime.py",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=1,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "gemma3"),),
        ),
        Job(
            name="gpu-qwen2-load",
            pytest_targets=("tests/model/test_qwen_model.py::test_qwen_wrapper_loads_actual_model[qwen2vl]",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-qwen25-load",
            pytest_targets=("tests/model/test_qwen_model.py::test_qwen_wrapper_loads_actual_model[qwen2.5vl]",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-qwen25-representative-load",
            pytest_targets=("tests/model/test_qwen_model.py::test_qwen_representative_wrapper_loads_cached_model",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-qwen-construct",
            pytest_targets=("tests/composite/test_vlm_sam.py::test_vlm_sam_constructs_with_representative_qwen_and_sam1",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-qwen-generate",
            pytest_targets=("tests/composite/test_vlm_sam_generate.py::test_vlm_sam_generate_returns_text_and_mask_batch_for_real_vlm[qwen2vl-generate]",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-gemma-load",
            pytest_targets=("tests/model/test_gemma_model.py::test_gemma_wrapper_loads_actual_model",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-gemma-composite-contract",
            pytest_targets=("tests/composite/test_vlm_sam.py::test_vlm_sam_runs_gemma_backbone_forward_with_sam_batch",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-sam1-load",
            pytest_targets=("tests/model/test_sam1_model.py::test_sam1_model_constructs_with_checkpoint",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-gemma-generate",
            pytest_targets=("tests/composite/test_vlm_sam_generate.py::test_vlm_sam_generate_returns_text_and_mask_batch_for_real_vlm[gemma3-generate]",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-molmo-load",
            pytest_targets=("tests/model/test_molmo_model.py::test_molmo_d_wrapper_loads",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-molmo-composite-contract",
            pytest_targets=("tests/composite/test_vlm_sam.py::test_vlm_sam_constructs_with_molmo_d_and_sam1",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-ve-clip-load",
            pytest_targets=("tests/model/test_vision_encoders.py::test_clip_encoder[openai/clip-vit-large-patch14-336-1024]",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-ve-siglip-load",
            pytest_targets=("tests/model/test_vision_encoders.py::test_siglip_encoder[google/siglip-so400m-patch14-384-1152]",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-ve-siglip2-load",
            pytest_targets=("tests/model/test_vision_encoders.py::test_siglip2_encoder[google/siglip2-so400m-patch14-384-1152]",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-ve-metaclip-load",
            pytest_targets=("tests/model/test_vision_encoders.py::test_metaclip_encoder[metaclip-facebook/metaclip-l14-fullcc2.5b-1024]",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-ve-openvision-load",
            pytest_targets=("tests/model/test_vision_encoders.py::test_openvision_encoder[openvision-UCSC-VLAA/openvision-vit-large-patch14-224-1024]",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-ve-swap-siglip-qwen",
            pytest_targets=("tests/model/test_vision_encoder_swap.py::test_swap_siglip_into_qwen_vlm_forward_passes",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-ve-swap-clip-qwen-adapter",
            pytest_targets=("tests/model/test_vision_encoder_swap.py::test_swap_clip_into_qwen_vlm_adapter_inserted",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-ve-extract-reload",
            pytest_targets=("tests/model/test_vision_encoder_swap.py::test_extract_qwen_visual_tower_and_reload",),
            pytest_args=("-rs", "-vv"),
            runtime="gpu",
            gpu_slots=1,
            priority=2,
        ),
        Job(
            name="gpu-qwen-ddp",
            pytest_targets=("tests/trainer/test_vlm_trainer_ddp.py",),
            pytest_args=("-rs", "-vv"),
            runtime="ddp",
            gpu_slots=2,
            priority=3,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "qwen2.5vl"),),
        ),
        Job(
            name="gpu-gemma-ddp",
            pytest_targets=("tests/trainer/test_vlm_trainer_ddp.py",),
            pytest_args=("-rs", "-vv"),
            runtime="ddp",
            gpu_slots=2,
            priority=3,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "gemma3"),),
        ),
        Job(
            name="gpu-qwen-dataparallel",
            pytest_targets=("tests/trainer/test_vlm_trainer_dataparallel.py",),
            pytest_args=("-rs", "-vv"),
            runtime="dataparallel",
            gpu_slots=2,
            priority=3,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "qwen2.5vl"),),
        ),
        Job(
            name="gpu-gemma-dataparallel",
            pytest_targets=("tests/trainer/test_vlm_trainer_dataparallel.py",),
            pytest_args=("-rs", "-vv"),
            runtime="dataparallel",
            gpu_slots=2,
            priority=3,
            env_overrides=(("TRAINING_TEST_SELECTED_VLM_SPECS", "gemma3"),),
        ),
    ]


def runtime_env(runtime: str) -> dict[str, str]:
    runtime_names = {
        "cpu": (),
        "gpu": ("gpu",),
        "ddp": ("gpu", "ddp"),
        "dataparallel": ("gpu", "dataparallel"),
    }[runtime]
    env = {
        "TRAINING_TEST_SELECTED_RUNTIMES": ",".join(runtime_names),
        "TRAINING_RUN_GPU_TESTS": "1" if "gpu" in runtime_names else "0",
        "TRAINING_RUN_GPU_TRAINER_TESTS": "1" if runtime == "gpu" else "0",
        "TRAINING_RUN_DDP_TESTS": "1" if runtime == "ddp" else "0",
        "TRAINING_RUN_DATAPARALLEL_TESTS": "1" if runtime == "dataparallel" else "0",
    }
    return env


def shell_assignments(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items() if value)


def build_docker_command(container: str, env: dict[str, str], pytest_targets: Sequence[str], pytest_args: Sequence[str]) -> list[str]:
    pytest_command = [
        "/environments/training_core/bin/python",
        "-m",
        "pytest",
        "-q",
        *pytest_targets,
        *pytest_args,
    ]
    inner = "cd /code/src/img_2_svg_pretraining/training && " + f"{shell_assignments(env)} " + shlex.join(pytest_command)
    return ["docker", "exec", container, "bash", "-lc", inner]


def allocate_gpu_ids(free_gpu_ids: list[str], gpu_slots: int) -> tuple[str, ...]:
    if gpu_slots == 0:
        return ()
    if len(free_gpu_ids) < gpu_slots:
        return ()
    allocated = tuple(free_gpu_ids[:gpu_slots])
    del free_gpu_ids[:gpu_slots]
    return allocated


def release_gpu_ids(free_gpu_ids: list[str], gpu_ids: tuple[str, ...]) -> None:
    free_gpu_ids.extend(gpu_ids)
    free_gpu_ids.sort(key=int)


def launch_job(
    container: str,
    common_env: dict[str, str],
    job: Job,
    log_dir: Path,
    gpu_ids: tuple[str, ...],
) -> RunningJob:
    env = dict(common_env)
    env.update(runtime_env(job.runtime))
    env.update(dict(job.env_overrides))
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    command = build_docker_command(container, env, job.pytest_targets, job.pytest_args)
    log_path = log_dir / f"{job.name}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    process._training_log_handle = log_handle  # type: ignore[attr-defined]
    return RunningJob(job=job, process=process, log_path=log_path, gpu_ids=gpu_ids)


def close_process_log(process: subprocess.Popen) -> None:
    log_handle = getattr(process, "_training_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def pop_next_schedulable_job(pending_jobs: list[Job], free_gpu_ids: list[str]) -> Job | None:
    for index, job in enumerate(pending_jobs):
        if job.gpu_slots <= len(free_gpu_ids):
            return pending_jobs.pop(index)
    return None


def format_gpu_ids(gpu_ids: tuple[str, ...]) -> str:
    return ",".join(gpu_ids) if gpu_ids else "-"


def wait_for_any_job(running_jobs: dict[str, RunningJob]) -> RunningJob:
    while True:
        for name, running in list(running_jobs.items()):
            return_code = running.process.poll()
            if return_code is not None:
                return running_jobs.pop(name)
        time.sleep(1)


def main() -> int:
    args = parse_args()
    common_env = build_common_env(args.cpu_threads_per_job)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    jobs = build_jobs()

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_dir) if args.log_dir else REPO_ROOT / "logs" / "test_matrix" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    free_gpu_ids = list(gpu_ids)
    pending_jobs = sorted(
        jobs,
        key=lambda job: (job.gpu_slots, job.priority, job.name),
    )
    running_jobs: dict[str, RunningJob] = {}
    failed = False

    print(f"Launching {len(jobs)} jobs across GPUs {','.join(gpu_ids)}. Logs: {log_dir}", flush=True)

    while pending_jobs or running_jobs:
        launched = False
        while True:
            next_job = pop_next_schedulable_job(pending_jobs, free_gpu_ids)
            if next_job is None:
                break
            assigned_gpu_ids = allocate_gpu_ids(free_gpu_ids, next_job.gpu_slots)
            running = launch_job(args.container, common_env, next_job, log_dir, assigned_gpu_ids)
            running_jobs[next_job.name] = running
            print(
                f"starting {next_job.name} | runtime={next_job.runtime} | gpu_slots={next_job.gpu_slots} | gpus={format_gpu_ids(assigned_gpu_ids)}",
                flush=True,
            )
            launched = True

        if launched:
            continue

        if not running_jobs:
            break

        completed = wait_for_any_job(running_jobs)
        return_code = completed.process.wait()
        close_process_log(completed.process)
        release_gpu_ids(free_gpu_ids, completed.gpu_ids)
        status = "PASSED" if return_code == 0 else f"FAILED ({return_code})"
        print(f"{completed.job.name}: {status} -> {completed.log_path}", flush=True)
        if return_code != 0:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
