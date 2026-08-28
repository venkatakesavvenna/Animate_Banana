#!/usr/bin/env bash
# Run the five-cell metric-soundness study on Qwen3.8-27B (SVG target).
#
# The five cells are the ones with a GT SVG reference video, chosen so the
# three-way comparison (reference / Gemini 3.7 Flash / Qwen3.8-27B) is possible.
# Style is per-cell, so this is three invocations, not one.
#
# Runs inside the container: playwright + chromium (SVG frame export) and the
# imageio-ffmpeg binary (mp4) live there, not on the host.
#
# PLAYWRIGHT_BROWSERS_PATH IS NOT OPTIONAL. The container does not set it, and
# playwright then looks in ~/.cache/ms-playwright -- which does not exist, so
# `export` dies with "Executable doesn't exist" after every model call has
# already been paid for. It bites ONLY at export: SVG stage-1 rendering uses a
# static rasteriser, and the animator critic's `compile_check` is tikz-only
# (animator/critic.py:102), so nothing earlier reveals the problem.
#
# COST. Qwen3.8-27B bills $0.425/M in, $2.55/M out, and the OpenRouter key has a
# hard $3 cap. Output dominates and the critics re-emit the whole SVG each
# round, so check `scripts/or_credit.sh` between waves rather than after.
set -uo pipefail

REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
C=img-2-svg-pretraining-singlenode-venkat.kesav
PY=/environments/img_2_svg_pretraining/bin/python
CFG=src/img_2_svg_pretraining/pipeline/configs/bench_v3_or_svg_qwen38_27b.yaml
# BOTH keys, comma-joined. `resolve_keys` splits on commas and hands the
# pool to KeyRing, so a key that hits its hard credit cap is marked
# exhausted mid-run and the next one takes over instead of the run dying.
KEY=$(grep -E "^OPEN_ROUTER_KEY(_[0-9]+)?=" "$REPO/.env" | cut -d= -f2 | paste -sd,)

run_cell() {                      # run_cell <style> <sample>...
  local style="$1"; shift
  local log="$REPO/logs/qwen38/${style}__$1.log"
  echo "=== $style : $* -> $log"
  docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$KEY" \
    -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
    "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline \
       all --config $CFG --style $style --only $*" > "$log" 2>&1
  echo "    exit=$? $(date +%H:%M:%S)"
}

case "${1:-all}" in
  one)  run_cell progressive_reveal CVPR_2025_arch01225 ;;
  rest) run_cell colour_pop           CVPR_2025_arch00389
        run_cell hopping_bounding_box CVPR_2025_arch00554
        run_cell hopping_bounding_box Paper2Fig_pipe_diff_000000714
        run_cell progressive_reveal   Paper2Fig_pipe_diff_000000677 ;;
  all)  run_cell progressive_reveal   CVPR_2025_arch01225
        run_cell colour_pop           CVPR_2025_arch00389
        run_cell hopping_bounding_box CVPR_2025_arch00554
        run_cell hopping_bounding_box Paper2Fig_pipe_diff_000000714
        run_cell progressive_reveal   Paper2Fig_pipe_diff_000000677 ;;
  *) echo "usage: $0 [one|rest|all]"; exit 2 ;;
esac
