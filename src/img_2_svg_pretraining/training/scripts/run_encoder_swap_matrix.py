from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTER_SRC = REPO_ROOT.parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(OUTER_SRC) not in sys.path:
    sys.path.insert(0, str(OUTER_SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from discover_test_checkpoints import discover_checkpoints
from img_2_svg_pretraining.training.training_core.builders.extract_vision_encoder import extract_vision_encoder
from img_2_svg_pretraining.training.training_core.matrix.encoder_swap_matrix import (
    DECODER_SPECS,
    ENCODER_SPECS,
    MATRIX_RUN_SPECS,
    MATRIX_TRAINING_DEFAULTS,
    MatrixRunSpec,
)


DEFAULT_GPU_IDS = ("0", "1", "2", "3", "4", "5", "6", "7")
DEFAULT_OUTPUT_BASE = REPO_ROOT / "outputs" / "encoder_swap_matrix"
DEFAULT_LOG_BASE = REPO_ROOT / "logs" / "encoder_swap_matrix"
DEFAULT_WANDB_KEY_FILE = REPO_ROOT / "api_keys" / "wandb_key"
DEFAULT_HF_TOKEN_FILE = REPO_ROOT / "api_keys" / "hf_token"
DEFAULT_HF_HOME = "/fsxvision_new/venkat.kesav/backup/hf_cache"
DEFAULT_HF_HUB_CACHE = f"{DEFAULT_HF_HOME}/hub"
DEFAULT_HF_MODULES_CACHE = "/tmp/training_hf_modules"


@dataclass(frozen=True)
class ScheduledRun:
    spec: MatrixRunSpec
    config_path: Path
    output_dir: Path
    logging_dir: Path


@dataclass
class RunningProcess:
    scheduled: ScheduledRun
    process: subprocess.Popen
    log_path: Path
    gpu_id: str
    log_handle: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full cross-decoder encoder-swap matrix on the local 8-GPU node.")
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--log-base", default=str(DEFAULT_LOG_BASE))
    parser.add_argument("--gpu-ids", default=",".join(DEFAULT_GPU_IDS))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--report-to", choices=("auto", "none", "wandb"), default="auto")
    parser.add_argument("--wandb-project", default="img-2-svg-pretraining-encoder-swap-matrix")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-key-file", default=str(DEFAULT_WANDB_KEY_FILE))
    parser.add_argument("--hf-token-file", default=str(DEFAULT_HF_TOKEN_FILE))
    parser.add_argument("--decoders", default="", help="Optional comma-separated decoder subset.")
    parser.add_argument("--encoders", default="", help="Optional comma-separated encoder subset.")
    parser.add_argument("--modes", default="", help="Optional comma-separated mode subset.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on total scheduled runs for debugging.")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _split_csv(raw_value: str) -> set[str]:
    return {value.strip() for value in raw_value.split(",") if value.strip()}


def should_enable_wandb(args: argparse.Namespace) -> bool:
    if args.report_to == "none":
        return False
    if args.report_to == "wandb":
        return True
    return bool(os.environ.get("WANDB_API_KEY") or Path(args.wandb_key_file).exists())


def build_common_env() -> dict[str, str]:
    env = os.environ.copy()
    discovered, missing = discover_checkpoints()
    env.update(discovered)
    required = {
        "TRAINING_TEST_QWEN25_MODEL",
        "TRAINING_TEST_GEMMA3_MODEL",
        "TRAINING_TEST_MOLMO_O_MODEL",
        "TRAINING_TEST_MOLMO_D_MODEL",
        "TRAINING_TEST_SAM1_CHECKPOINT",
    }
    missing_required = sorted(name for name in required if not env.get(name))
    if missing_required:
        raise SystemExit(
            "Preflight failed. Missing required checkpoint env vars or discovered paths: "
            + ", ".join(missing_required)
        )
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HOME", DEFAULT_HF_HOME)
    env.setdefault("HUGGINGFACE_HUB_CACHE", DEFAULT_HF_HUB_CACHE)
    env.setdefault("HF_MODULES_CACHE", DEFAULT_HF_MODULES_CACHE)
    env.setdefault("TRANSFORMERS_CACHE", DEFAULT_HF_HUB_CACHE)
    env.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
    env.setdefault("HF_PARALLEL_LOADING_WORKERS", "8")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("NUMEXPR_NUM_THREADS", "4")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def resolve_selected_runs(args: argparse.Namespace) -> list[MatrixRunSpec]:
    selected_decoders = _split_csv(args.decoders)
    selected_encoders = _split_csv(args.encoders)
    selected_modes = _split_csv(args.modes)

    runs = []
    for spec in MATRIX_RUN_SPECS:
        if selected_decoders and spec.decoder_slug not in selected_decoders:
            continue
        if selected_encoders and spec.encoder_slug not in selected_encoders:
            continue
        if selected_modes and spec.mode not in selected_modes:
            continue
        runs.append(spec)

    # Front-load the heaviest or most failure-prone rows to reduce tail risk.
    decoder_rank = {
        "molmo7b-d": 0,
        "molmo7b-o": 1,
        "qwen25": 2,
        "gemma3": 3,
    }
    encoder_rank = {
        "openvision": 0,
        "metaclip2": 1,
        "metaclip": 2,
        "siglip2": 3,
        "siglip": 4,
        "clip": 5,
        "extracted-qwen25": 6,
        "extracted-gemma3": 7,
        "extracted-molmo7bo": 8,
        "extracted-molmo7bd": 9,
        "native": 10,
    }
    runs.sort(key=lambda spec: (decoder_rank.get(spec.decoder_slug, 99), encoder_rank.get(spec.encoder_slug, 99), spec.mode))
    if args.limit > 0:
        runs = runs[: args.limit]
    return runs


def _extracted_encoder_slug_by_source() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for encoder in ENCODER_SPECS:
        if encoder.kind == "extracted" and encoder.source_decoder_slug:
            mapping[encoder.source_decoder_slug] = encoder.slug
    return mapping


def find_existing_extracted_encoder(output_base: Path, encoder_slug: str) -> Path | None:
    if not output_base.exists():
        return None

    candidates = []
    for run_dir in sorted(output_base.iterdir(), reverse=True):
        encoder_dir = run_dir / "extracted_encoders" / encoder_slug
        module_path = encoder_dir / "module.pt"
        preprocessor_path = encoder_dir / "preprocessor.json"
        if module_path.exists() and preprocessor_path.exists():
            candidates.append(encoder_dir)
    return candidates[0] if candidates else None


def prepare_extracted_encoders(
    env: dict[str, str],
    extracted_root: Path,
    selected_runs: list[MatrixRunSpec],
    force_reextract: bool,
) -> dict[str, Path]:
    extracted_root.mkdir(parents=True, exist_ok=True)
    slug_by_source = _extracted_encoder_slug_by_source()
    paths: dict[str, Path] = {}
    required_sources = {
        spec.encoder_source_decoder_slug
        for spec in selected_runs
        if spec.encoder_kind == "extracted" and spec.encoder_source_decoder_slug
    }

    for decoder in DECODER_SPECS:
        if decoder.slug not in required_sources:
            continue
        encoder_slug = slug_by_source[decoder.slug]
        encoder_dir = extracted_root / encoder_slug
        module_path = encoder_dir / "module.pt"
        if module_path.exists() and not force_reextract:
            paths[decoder.slug] = encoder_dir
            continue
        if not force_reextract:
            existing = find_existing_extracted_encoder(DEFAULT_OUTPUT_BASE, encoder_slug)
            if existing is not None:
                paths[decoder.slug] = existing
                continue

        model_ref = env[decoder.checkpoint_env_var]
        extract_vision_encoder(
            vlm_family=decoder.vlm_family,
            vlm_checkpoint=model_ref,
            model_family_name=decoder.model_family_name,
            encoder_name=encoder_slug,
            output_dir=str(extracted_root),
        )
        paths[decoder.slug] = encoder_dir

    return paths


def build_run_config_dict(
    spec: MatrixRunSpec,
    env: dict[str, str],
    scheduled_output_dir: Path,
    scheduled_logging_dir: Path,
    extracted_paths: dict[str, Path],
    wandb_enabled: bool,
    args: argparse.Namespace,
) -> dict:
    trainer_cfg = dict(MATRIX_TRAINING_DEFAULTS["trainer"])
    trainer_cfg["gradient_checkpointing"] = spec.gradient_checkpointing
    trainer_cfg["run_name"] = spec.run_name
    trainer_cfg["report_to"] = "wandb" if wandb_enabled else "none"

    cfg = {
        "vlm_family": spec.vlm_family,
        "base_model": env[spec.checkpoint_env_var],
        "model_family_name": spec.model_family_name,
        "attn_implementation": spec.attn_implementation,
        # The matrix is a compatibility / short-run benchmark. Keep the expensive
        # decoder and vision tower frozen so each row focuses on whether the
        # composition works end-to-end on a shared 8-GPU node.
        "tune_mm_llm": False,
        "tune_mm_vision": False,
        "tune_mm_mlp": True,
        "tune_mm_lm_head": True,
        "sam_version": spec.sam_version,
        "sam_checkpoint": env[spec.sam_checkpoint_env_var],
        "out_dim": 256,
        "sam_image_size": 1024,
        "run_name": spec.run_name,
        "split_ratio": 0.5,
        "seed": 42,
        "skip_final_model_save": True,
        "logging_dir": str(scheduled_logging_dir),
        "output_dir": str(scheduled_output_dir),
        "checkpoint": {
            "resume": False,
            "path": None,
            "mode": "weights_only",
        },
        "trainer": trainer_cfg,
        "loss": dict(MATRIX_TRAINING_DEFAULTS["loss_by_mode"][spec.mode]),
        "datasets": {
            "train": [
                {
                    "name": MATRIX_TRAINING_DEFAULTS["dataset_name"],
                    "dataset_kwargs": dict(MATRIX_TRAINING_DEFAULTS["dataset_kwargs"]),
                }
            ],
            "val": [{"name": MATRIX_TRAINING_DEFAULTS["dataset_name"]}],
            "eval": [{"name": MATRIX_TRAINING_DEFAULTS["dataset_name"]}],
        },
    }

    if wandb_enabled:
        cfg["wandb_project"] = args.wandb_project
        if args.wandb_entity:
            cfg["wandb_entity"] = args.wandb_entity
        cfg["wandb_key_file"] = args.wandb_key_file

    if spec.encoder_kind == "standalone":
        cfg["vision_encoder"] = spec.vision_encoder
        cfg["vision_encoder_checkpoint"] = spec.vision_encoder_checkpoint
        if spec.vision_encoder_arch:
            cfg["vision_encoder_arch"] = spec.vision_encoder_arch
    elif spec.encoder_kind == "extracted":
        cfg["vision_encoder"] = "extracted"
        cfg["vision_encoder_checkpoint"] = str(extracted_paths[spec.encoder_source_decoder_slug])

    return cfg


def materialize_runs(
    selected_runs: list[MatrixRunSpec],
    env: dict[str, str],
    run_root: Path,
    extracted_paths: dict[str, Path],
    wandb_enabled: bool,
    args: argparse.Namespace,
) -> list[ScheduledRun]:
    configs_dir = run_root / "configs"
    jobs_dir = run_root / "jobs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    scheduled_runs: list[ScheduledRun] = []
    for spec in selected_runs:
        job_root = jobs_dir / spec.run_name
        output_dir = job_root / "checkpoints"
        logging_dir = job_root / "validation_steps"
        config_path = configs_dir / f"{spec.run_name}.yaml"
        cfg = build_run_config_dict(
            spec=spec,
            env=env,
            scheduled_output_dir=output_dir,
            scheduled_logging_dir=logging_dir,
            extracted_paths=extracted_paths,
            wandb_enabled=wandb_enabled,
            args=args,
        )
        OmegaConf.save(config=OmegaConf.create(cfg), f=str(config_path))
        scheduled_runs.append(
            ScheduledRun(
                spec=spec,
                config_path=config_path,
                output_dir=output_dir,
                logging_dir=logging_dir,
            )
        )
    return scheduled_runs


def prewarm_doclaynet_metadata(common_env: dict[str, str], python_bin: str) -> None:
    data_path = MATRIX_TRAINING_DEFAULTS["dataset_kwargs"]["data_path"]
    command = [
        python_bin,
        "-c",
        (
            "from datasets import load_dataset; "
            f"ds = load_dataset('{data_path}', split='train', streaming=True); "
            "next(iter(ds)); "
            "print('doclaynet-prewarm-ok')"
        ),
    ]
    subprocess.run(command, cwd=str(REPO_ROOT), env=common_env, check=True)


def parse_gpu_ids(raw_value: str) -> list[str]:
    gpu_ids = [value.strip() for value in raw_value.split(",") if value.strip()]
    if len(gpu_ids) != 8:
        raise SystemExit(f"Expected exactly 8 GPU ids for this matrix, got {len(gpu_ids)} from {raw_value!r}")
    return gpu_ids


def launch_job(
    scheduled: ScheduledRun,
    gpu_id: str,
    common_env: dict[str, str],
    python_bin: str,
    log_root: Path,
    wandb_enabled: bool,
) -> RunningProcess:
    log_path = log_root / f"{scheduled.spec.run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = common_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["TRAINING_CONFIG_PATH"] = str(scheduled.config_path)
    env["WANDB_NAME"] = scheduled.spec.run_name
    # Prewarm already cached the dataset metadata; skip Hub API calls in training
    # jobs so 8 concurrent workers don't hit the 429 rate limit (1000 req/5 min).
    env["HUGGINGFACE_HUB_OFFLINE"] = "1"
    if wandb_enabled:
        env["WANDB_GROUP"] = f"{scheduled.spec.decoder_slug}__{scheduled.spec.encoder_slug}"
        env["WANDB_JOB_TYPE"] = scheduled.spec.mode

    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        [python_bin, "-m", "img_2_svg_pretraining.training.training_core.train.train"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return RunningProcess(
        scheduled=scheduled,
        process=process,
        log_path=log_path,
        gpu_id=gpu_id,
        log_handle=log_handle,
    )


def find_first_validation_png(logging_dir: Path) -> str | None:
    if not logging_dir.exists():
        return None
    pngs = sorted(logging_dir.rglob("*.png"))
    return str(pngs[0]) if pngs else None


def load_train_loss(output_dir: Path) -> float | None:
    trainer_state = output_dir / "trainer_state.json"
    if not trainer_state.exists():
        return None
    data = json.loads(trainer_state.read_text())
    for entry in reversed(data.get("log_history", [])):
        if "train_loss" in entry:
            return float(entry["train_loss"])
        if "loss" in entry:
            return float(entry["loss"])
    return None


def build_result_record(running: RunningProcess, returncode: int) -> dict:
    output_dir = running.scheduled.output_dir
    logging_dir = running.scheduled.logging_dir
    validation_png = find_first_validation_png(logging_dir)
    status = "passed" if returncode == 0 else "failed"
    if running.scheduled.spec.mode == "sam1" and not validation_png:
        status = "failed"
    return {
        "decoder": running.scheduled.spec.decoder_slug,
        "encoder": running.scheduled.spec.encoder_slug,
        "mode": running.scheduled.spec.mode,
        "attention_backend": running.scheduled.spec.attn_implementation,
        "status": status,
        "returncode": returncode,
        "train_loss": load_train_loss(output_dir),
        "output_path": str(output_dir),
        "validation_artifact_path": validation_png,
        "config_path": str(running.scheduled.config_path),
        "log_path": str(running.log_path),
        "gpu_id": running.gpu_id,
    }


def write_result_summaries(run_root: Path, records: list[dict]) -> None:
    manifest_path = run_root / "manifest.json"
    csv_path = run_root / "summary.csv"
    markdown_path = run_root / "SUMMARY.md"

    manifest_path.write_text(json.dumps(records, indent=2))

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)

    lines = [
        "# Encoder Swap Matrix",
        "",
        "| decoder | encoder | mode | status | train_loss | validation_artifact |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for record in records:
        train_loss = "" if record["train_loss"] is None else f"{record['train_loss']:.4f}"
        validation_artifact = record["validation_artifact_path"] or ""
        lines.append(
            f"| {record['decoder']} | {record['encoder']} | {record['mode']} | "
            f"{record['status']} | {train_loss} | {validation_artifact} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n")


def run_matrix(args: argparse.Namespace) -> int:
    common_env = build_common_env()
    wandb_enabled = should_enable_wandb(args)
    if not common_env.get("HF_TOKEN") and Path(args.hf_token_file).exists():
        hf_token = Path(args.hf_token_file).read_text(encoding="utf-8").strip()
        common_env["HF_TOKEN"] = hf_token
        common_env.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
    if wandb_enabled and not common_env.get("WANDB_API_KEY") and Path(args.wandb_key_file).exists():
        common_env["WANDB_API_KEY"] = Path(args.wandb_key_file).read_text(encoding="utf-8").strip()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(args.output_base) / timestamp
    log_root = Path(args.log_base) / timestamp
    log_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    selected_runs = resolve_selected_runs(args)
    prewarm_doclaynet_metadata(common_env, args.python_bin)
    extracted_paths = prepare_extracted_encoders(
        env=common_env,
        extracted_root=run_root / "extracted_encoders",
        selected_runs=selected_runs,
        force_reextract=args.force_reextract,
    )
    scheduled_runs = materialize_runs(
        selected_runs=selected_runs,
        env=common_env,
        run_root=run_root,
        extracted_paths=extracted_paths,
        wandb_enabled=wandb_enabled,
        args=args,
    )
    (run_root / "selected_runs.json").write_text(
        json.dumps([asdict(run.spec) for run in scheduled_runs], indent=2)
    )

    free_gpus = parse_gpu_ids(args.gpu_ids)
    pending = list(scheduled_runs)
    running: list[RunningProcess] = []
    finished_records: list[dict] = []

    while pending or running:
        while pending and free_gpus:
            gpu_id = free_gpus.pop(0)
            running.append(
                launch_job(
                    scheduled=pending.pop(0),
                    gpu_id=gpu_id,
                    common_env=common_env,
                    python_bin=args.python_bin,
                    log_root=log_root,
                    wandb_enabled=wandb_enabled,
                )
            )

        next_running: list[RunningProcess] = []
        for job in running:
            returncode = job.process.poll()
            if returncode is None:
                next_running.append(job)
                continue
            job.log_handle.close()
            free_gpus.append(job.gpu_id)
            record = build_result_record(job, returncode)
            finished_records.append(record)
            write_result_summaries(run_root, finished_records)
            if args.fail_fast and record["status"] != "passed":
                for other in next_running:
                    other.process.terminate()
                return 1
        running = next_running
        free_gpus.sort(key=int)
        if pending or running:
            time.sleep(5)

    write_result_summaries(run_root, finished_records)
    failed = [record for record in finished_records if record["status"] != "passed"]
    print(f"Run root: {run_root}")
    print(f"Completed {len(finished_records)} runs; failures={len(failed)}")
    return 1 if failed else 0


def main() -> None:
    raise SystemExit(run_matrix(parse_args()))


if __name__ == "__main__":
    main()
