#!/bin/bash
# Submit smoke test jobs (max_steps=200) for all PT configs, skipping 72B.
#
# Usage:
#   bash scripts/submit_smoke_tests.sh [--dry-run]
#
# Smoke configs are pre-generated in src/configs/smoke/encoder_swap/.
# Native smoke configs are in the same dir (mn_*_pt_smoke.yaml).
# W&B project: img-2-svg-pretraining-smoke

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_DIR="${REPO_ROOT}/configs/smoke/encoder_swap"
SWAP_SCRIPTS_DIR="${REPO_ROOT}/scripts/encoder_swap"
NATIVE_SCRIPTS_DIR="${REPO_ROOT}/scripts"
DRY_RUN=false

[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "${REPO_ROOT}/logs"

submitted=0
failed=0

submit_one() {
    local smoke_cfg="$1" script="$2" name="$3"
    if [[ ! -f "$script" ]]; then
        echo "[WARN] No script: $script"
        failed=$(( failed + 1 ))
        return
    fi
    if $DRY_RUN; then
        echo "[DRY] sbatch --export=ALL,TRAINING_CONFIG=/code/src/img_2_svg_pretraining/training/configs/smoke/encoder_swap/$(basename "$smoke_cfg") $script"
        submitted=$(( submitted + 1 ))
    else
        if sbatch \
            --export="ALL,TRAINING_CONFIG=/code/src/img_2_svg_pretraining/training/configs/smoke/encoder_swap/$(basename "$smoke_cfg")" \
            "$script"; then
            echo "  Submitted: $name"
            submitted=$(( submitted + 1 ))
        else
            echo "  [WARN] sbatch failed for: $name"
            failed=$(( failed + 1 ))
        fi
    fi
}

echo "=== Smoke test submission (200 steps) ==="
echo ""

for smoke_cfg in "${SMOKE_DIR}"/*.yaml; do
    name=$(basename "$smoke_cfg" .yaml)
    # Strip _smoke suffix to get the production config name
    prod_name="${name%_smoke}"

    # Determine which script to use (same as production script)
    if [[ -f "${SWAP_SCRIPTS_DIR}/${prod_name}.sh" ]]; then
        script="${SWAP_SCRIPTS_DIR}/${prod_name}.sh"
    elif [[ -f "${NATIVE_SCRIPTS_DIR}/${prod_name}.sh" ]]; then
        script="${NATIVE_SCRIPTS_DIR}/${prod_name}.sh"
    else
        echo "[WARN] No script found for: $prod_name"
        failed=$(( failed + 1 ))
        continue
    fi

    submit_one "$smoke_cfg" "$script" "$name"
done

echo ""
echo "=== Summary ==="
echo "  Submitted : ${submitted}"
echo "  Failed    : ${failed}"
$DRY_RUN && echo "  [DRY RUN — no jobs submitted]"
echo ""
echo "Monitor progress: squeue -u \$(whoami)"
echo "W&B dashboard:    https://wandb.ai/img-2-svg-pretraining-smoke"
