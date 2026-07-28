from __future__ import annotations

from pathlib import Path


CACHE_ROOT_GLOBS = (
    "/fsxvision_new/*/hf_cache/hub",
    "/fsxvision_new/*/hf_cache",
    "/fsxvision_new/*/backup/hf_cache/hub",
    "/fsxvision_new/*/backup/hf_cache",
    "/fsxvision_new/*/*/hf_cache/hub",
    "/fsxvision_new/*/*/hf_cache",
)

MODEL_CANDIDATES = {
    "TRAINING_TEST_QWEN2_MODEL": (
        "models--Qwen--Qwen2-VL-2B-Instruct",
        "models--Qwen--Qwen2-VL-7B-Instruct",
    ),
    "TRAINING_TEST_QWEN25_MODEL": (
        "models--Qwen--Qwen2.5-VL-7B-Instruct",
    ),
    "TRAINING_TEST_QWEN3_MODEL": (
        "models--Qwen--Qwen3-VL-4B-Instruct",
        "models--Qwen--Qwen3-VL-8B-Instruct",
        "models--Qwen--Qwen3-VL-32B-Instruct",
    ),
    "TRAINING_TEST_QWEN3_MOE_MODEL": (
        "models--Qwen--Qwen3-VL-30B-A3B-Instruct",
    ),
    "TRAINING_TEST_GEMMA3_MODEL": (
        "models--google--gemma-3-4b-it",
        "models--tetf--gemma-3-4b-it",
    ),
    "TRAINING_TEST_MOLMO_D_MODEL": (
        "models--allenai--Molmo-7B-D-0924",
    ),
    "TRAINING_TEST_MOLMO_O_MODEL": (
        "models--allenai--Molmo-7B-O-0924",
    ),
    "TRAINING_TEST_MOLMO_1B_MODEL": (
        "models--allenai--MolmoE-1B-0924",
    ),
}

SAM1_CANDIDATES = (
    Path("/fsxvision_new/srihari.bandarupalli/DocGrounding/checkpoints/sam_vit_h_4b8939.pth"),
)


def iter_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for pattern in CACHE_ROOT_GLOBS:
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            if path.is_dir() and path not in roots:
                roots.append(path)
    return roots


def first_snapshot(model_dir: Path) -> Path | None:
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    return snapshots[0] if snapshots else None


def discover_checkpoints() -> tuple[dict[str, str], list[str]]:
    discovered_env: dict[str, str] = {}
    missing: list[str] = []
    roots = iter_cache_roots()

    for env_var, model_dirs in MODEL_CANDIDATES.items():
        discovered = None
        for root in roots:
            for model_dir_name in model_dirs:
                snapshot = first_snapshot(root / model_dir_name)
                if snapshot:
                    discovered = snapshot
                    break
            if discovered:
                break

        if discovered:
            discovered_env[env_var] = str(discovered)
        else:
            missing.append(env_var)

    for candidate in SAM1_CANDIDATES:
        if candidate.exists():
            discovered_env["TRAINING_TEST_SAM1_CHECKPOINT"] = str(candidate)
            break
    else:
        missing.append("TRAINING_TEST_SAM1_CHECKPOINT")

    return discovered_env, missing


def discover_model_exports() -> tuple[list[str], list[str]]:
    discovered_env, missing = discover_checkpoints()
    exports = [f"export {env_var}={value}" for env_var, value in discovered_env.items()]
    return exports, missing


def main() -> None:
    exports, missing = discover_model_exports()
    print("# Suggested checkpoint exports")
    for line in exports:
        print(line)

    if missing:
        print()
        print("# Missing in current local cache")
        for env_var in missing:
            print(env_var)


if __name__ == "__main__":
    main()
