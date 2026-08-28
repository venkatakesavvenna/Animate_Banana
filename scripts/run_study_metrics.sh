#!/usr/bin/env bash
# Score the five study cells with the REVISED animation metrics.
#
# Two judge passes, because the two groups need different judges and different
# modalities:
#
#   vfs_band + ascs_video   Gemini 3.7 Flash, native `gemini` backend, video.
#                           Free-tier Google keys (75 x 20/day), so $0.
#   sss + gps               Qwen3.8-Flash via OpenRouter, frame pairs + XML.
#                           --rubric letters selects the JSON A-E prompts.
#
# `omission` runs here with its ORIGINAL prompt but a DIFFERENT judge. The one
# that produced every stored omission score -- Qwen3-VL-235B on the local vLLM
# at :8010 -- is down, so these numbers are not comparable with previously
# stored omission scores and should not be pooled with them. The viewer renders
# a missing value as 0, which also cannot be distinguished from a real zero.
#
# A SEPARATE EVALS ROOT. `run_eval` stamps `provenance` from whichever judge it
# was invoked with, so writing into the live tree would relabel every existing
# Qwen-produced score on each record it touched. Into a fresh root, rollback is
# `rm -rf` and nothing published is at risk.
set -uo pipefail

REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
C=img-2-svg-pretraining-singlenode-venkat.kesav
PY=/environments/img_2_svg_pretraining/bin/python
VROOT=data/animatebench_v3_cache/animatebench_v3/evals_study
KEY=$(grep -E "^OPEN_ROUTER_KEY(_[0-9]+)?=" "$REPO/.env" | cut -d= -f2 | paste -sd,)

CELLS=(
  "progressive_reveal:CVPR_2025_arch00012"
  "colour_pop:CVPR_2025_arch00389"
  "hopping_bounding_box:CVPR_2025_arch00554"
  "alpha_masking:CVPR_2025_arch00855"
  "progressive_reveal:CVPR_2025_arch01215"
  "sliding_bounding_box:CVPR_2025_arch01224"
  "progressive_reveal:CVPR_2025_arch01225"
  "sliding_bounding_box:CVPR_2025_arch01311"
  "alpha_masking:CVPR_2025_arch01541"
  "hopping_bounding_box:CVPR_2025_arch01707"
  "sliding_bounding_box:CVPR_2025_arch01709"
  "progressive_reveal:CVPR_2025_arch01871"
  "progressive_reveal:Paper2Fig_pipe_diff_000000536"
  "hopping_bounding_box:Paper2Fig_pipe_diff_000000542"
  "progressive_reveal:Paper2Fig_pipe_diff_000000673"
  "progressive_reveal:Paper2Fig_pipe_diff_000000677"
  "progressive_reveal:Paper2Fig_pipe_diff_000000690"
  "hopping_bounding_box:Paper2Fig_pipe_diff_000000714"
  "sliding_bounding_box:arch_arxiv_ai_2025_000000126"
  "colour_pop:arch_arxiv_ai_2025_000000127"
  "hopping_bounding_box:arch_arxiv_ai_2025_000000151"
  "hopping_bounding_box:arch_arxiv_cv_2023_000000298"
  "hopping_bounding_box:arch_arxiv_cv_2023_000000303"
  "progressive_reveal:arch_arxiv_db_ir_2024_000000546"
  "progressive_reveal:arch_arxiv_db_ir_2025_000000580"
  "progressive_reveal:arch_arxiv_db_ir_2025_000000587"
  "hopping_bounding_box:arch_arxiv_db_ir_2025_000000589"
  "progressive_reveal:arch_arxiv_db_ir_2025_000000611"
  "colour_pop:arch_arxiv_db_ir_2025_000000618"
  "colour_pop:arch_arxiv_db_ir_2026_000000714"
  "sliding_bounding_box:arch_arxiv_db_ir_2026_000000715"
  "progressive_reveal:arch_arxiv_db_ir_2026_000000718"
  "progressive_reveal:CVPR_2025_pipe00004"
  "progressive_reveal:CVPR_2025_pipe00010"
  "progressive_reveal:CVPR_2025_pipe00011"
  "progressive_reveal:CVPR_2025_pipe00041"
  "progressive_reveal:CVPR_2025_pipe00045"
  "progressive_reveal:CVPR_2025_pipe00137"
)
CONFIGS=(bench_v3_or_svg.yaml bench_v3_or_svg_qwen38_27b.yaml \
         bench_v3_or_svg_gemma4_26b.yaml bench_v3_zeroshot_gemini31.yaml)

score() {   # score <config> <style> <sample> <backend> <rubric> <stages...>
  local cfg="$1" style="$2" sample="$3" backend="$4" rubric="$5"; shift 5
  docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$KEY" \
    -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
    "cd /code && PYTHONPATH=src PIPELINE_KEY_FILE=api_keys.csv $PY -u \
       -m img_2_svg_pretraining.animatebench.run_eval animation \
       --config src/img_2_svg_pretraining/pipeline/configs/$cfg \
       --style $style --only $sample --stages $* \
       --rubric $rubric --judge-backend $backend --evals-root $VROOT --force" \
    >> "$REPO/logs/qwen38/metrics__${cfg%.yaml}__${style}__${sample}.log" 2>&1
  echo "    $cfg $style/$sample [$*] exit=$?"
}

# CELLS RUN CONCURRENTLY, PASSES DO NOT.
#
# `_banded_json` judges one timestep per call, sequentially, so a 21-frame cell
# is ~42 round trips and the sweep is entirely latency-bound. Running cells in
# parallel turns 9 x that into ceil(9/JOBS) x that.
#
# The two passes within a cell stay ordered on purpose: both write the SAME
# animation.json, and `--force` makes each one read-modify-write it. Running
# them concurrently is a lost-update race -- whichever finished second would
# overwrite the other's stages with a copy of the record it read before they
# were added, and the loss would be invisible because both passes report
# success.
JOBS="${JOBS:-4}"

run_cell() {   # run_cell <config> <style> <sample>
  score "$1" "$2" "$3" gemini_video       headers vfs_band ascs_video
  # Zeroshot ships a finished animation with no parser XML and no sequence.
  # `run_eval` builds `step_frames_` only when BOTH exist, and sss/gps/omission
  # are gated on it -- they would return nothing at all. Skipping is honest and
  # saves ~40 pointless judge calls per cell.
  case "$1" in *zeroshot*) return ;; esac
  score "$1" "$2" "$3" qwen38_flash_judge letters sss gps omission
}

for cfg in "${CONFIGS[@]}"; do
  for cell in "${CELLS[@]}"; do
    style="${cell%%:*}"; sample="${cell##*:}"
    # Only score a cell that actually produced a video -- FOR THIS CONFIG.
    # A bare exports/*/<sample>/ glob matches the other model's lineage too, so
    # it would report "have mp4" for a Qwen cell still mid-generation purely
    # because Gemini had already exported that sample, and the judge would then
    # score whichever deck it found.
    case "$cfg" in
      *qwen38_27b*)  prefix="qwen-qwen3.8-27b" ;;
      *gemma4_26b*)  prefix="google-gemma-4-26b-a4b-it" ;;
      *zeroshot*)    prefix="google-gemini-3.1-zeroshot" ;;
      *or_svg.yaml)  prefix="google-gemini-3.7-flash" ;;
      *)
        # NO SILENT DEFAULT. Falling through to the Gemini prefix would make the
        # runner score GEMINI's mp4 while writing the record under another
        # model's config name -- silent, plausible, and it invalidates the whole
        # comparison. Fail loudly instead.
        echo "    ABORT: no lineage prefix for config '$cfg'" >&2; exit 2 ;;
    esac
    have=$(docker exec "$C" bash -lc "ls /code/data/animatebench_v3_cache/animatebench_v3/exports/${prefix}*__${style}__*/${sample}/animation.mp4 2>/dev/null | wc -l")
    if [ "${have:-0}" -eq 0 ]; then echo "    skip $cfg $style/$sample (no mp4 yet)"; continue; fi
    run_cell "$cfg" "$style" "$sample" &
    # Cap in-flight cells so the judge endpoints are not swamped.
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  done
done
wait
echo "=== done; records under $VROOT"
