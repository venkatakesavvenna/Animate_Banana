from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTER_SRC = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.support.matrix import get_spec, resolve_model_ref, resolve_sam1_checkpoint


LAYER_PATHS = {
    "contracts": "tests/contracts",
    "data": "tests/data",
    "model": "tests/model",
    "composite": "tests/composite",
    "trainer": "tests/trainer",
}

PROFILE_CONFIG = {
    "cpu-pr": {
        "layers": ("contracts", "data", "model", "composite"),
        "marker": "not external and not gpu and not ddp and not dataparallel and not slow",
        "runtimes": ("cpu",),
        "specs": (),
    },
    "gpu-representative": {
        "layers": ("model", "composite", "trainer"),
        "marker": "external and (gpu or ddp or dataparallel)",
        "runtimes": ("gpu", "ddp", "dataparallel"),
        "specs": ("qwen2.5vl", "gemma3", "sam1"),
    },
    "gpu-nightly": {
        "layers": ("data", "model", "composite", "trainer"),
        "marker": "external",
        "runtimes": ("gpu", "ddp", "dataparallel"),
        "specs": ("qwen2vl", "qwen2.5vl", "qwen3vl", "qwen3vl-moe", "gemma3", "sam1"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run structured training test suites.")
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIG), help="Named test-suite profile to run.")
    parser.add_argument("--layers", default="", help="Comma-separated layer list overriding the profile.")
    parser.add_argument("--marker", default=None, help="Optional pytest marker expression override.")
    parser.add_argument("--specs", default="", help="Comma-separated external model specs required for preflight.")
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER, default=[], help="Extra pytest arguments.")
    return parser.parse_args()


def legacy_runtimes_from_env() -> tuple[str, ...]:
    runtimes = []
    if os.environ.get("TRAINING_RUN_GPU_TESTS", "0") == "1" or os.environ.get("TRAINING_RUN_GPU_TRAINER_TESTS", "0") == "1":
        runtimes.append("gpu")
    if os.environ.get("TRAINING_RUN_DDP_TESTS", "0") == "1":
        runtimes.append("ddp")
    if os.environ.get("TRAINING_RUN_DATAPARALLEL_TESTS", "0") == "1":
        runtimes.append("dataparallel")
    return tuple(runtimes or ["cpu"])


def resolve_missing_specs(spec_slugs: tuple[str, ...]) -> list[str]:
    missing = []
    for slug in spec_slugs:
        if slug == "sam1":
            if not resolve_sam1_checkpoint():
                missing.append("TRAINING_TEST_SAM1_CHECKPOINT")
            continue
        spec = get_spec(slug)
        if not resolve_model_ref(spec):
            missing.append(spec.env_var)
    return missing


def main():
    args = parse_args()
    profile = PROFILE_CONFIG.get(args.profile or "", {})

    layers = tuple(value.strip() for value in (args.layers or "").split(",") if value.strip()) or profile.get("layers") or tuple(LAYER_PATHS)
    marker = args.marker if args.marker is not None else profile.get("marker")
    runtimes = profile.get("runtimes") or legacy_runtimes_from_env()
    spec_slugs = tuple(value.strip() for value in (args.specs or "").split(",") if value.strip()) or profile.get("specs") or ()

    missing = resolve_missing_specs(spec_slugs)
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(f"Preflight failed. Missing required checkpoint env vars or paths: {missing_str}")

    env = os.environ.copy()
    env["TRAINING_TEST_SELECTED_LAYERS"] = ",".join(layers)
    env["TRAINING_TEST_SELECTED_RUNTIMES"] = ",".join(runtimes)
    env["TRAINING_TEST_SELECTED_VLM_SPECS"] = ",".join(slug for slug in spec_slugs if slug != "sam1")
    env.setdefault("PYTHONPATH", os.pathsep.join([str(OUTER_SRC), str(REPO_ROOT)]))

    if any(slug != "sam1" for slug in spec_slugs):
        env["TRAINING_TEST_INCLUDE_EXTERNAL"] = "1"
        env["TRAINING_RUN_LARGE_MODEL_TESTS"] = "1"
    if "gpu" in runtimes:
        env["TRAINING_RUN_GPU_TESTS"] = "1"
        env["TRAINING_RUN_GPU_TRAINER_TESTS"] = "1"
    if "ddp" in runtimes:
        env["TRAINING_RUN_DDP_TESTS"] = "1"
    if "dataparallel" in runtimes:
        env["TRAINING_RUN_DATAPARALLEL_TESTS"] = "1"

    command = [sys.executable, "-m", "pytest", "-q"]
    if marker:
        command.extend(["-m", marker])
    command.extend(LAYER_PATHS[layer] for layer in layers)
    command.extend(args.pytest_args)

    completed = subprocess.run(command, env=env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
