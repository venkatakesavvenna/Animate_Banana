#!/usr/bin/env bash
# Re-export every generated cell with the style-aware frame sampling.
#
# FREE: `export` makes no model calls. It re-rasterises the animation SVG that
# already exists on disk, so this costs only wall clock.
#
# WHY RE-EXPORT AT ALL. The old uniform grid sampled ON state boundaries, which
# is where one element has faded out and the next has not yet faded in. Two
# consequences, both measured:
#   * narration subtitles were absent from 21 of 22 frames on a correctly
#     authored animation;
#   * a sliding box exported as rest positions only -- 26 states, identical in
#     structure to a hop -- so the two styles were indistinguishable.
# `svg_style_frames.sample_times_for` now samples the MIDDLE of each state, and
# adds in-transit frames for sliding.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
C=img-2-svg-pretraining-singlenode-venkat.kesav
PY=/environments/img_2_svg_pretraining/bin/python
JOBS="${JOBS:-4}"
mkdir -p logs/reexport

V2="CVPR_2025_pipe00004 CVPR_2025_pipe00010 CVPR_2025_pipe00011 CVPR_2025_pipe00041 CVPR_2025_pipe00045 CVPR_2025_pipe00137"

one() {  # one <config-stem> <style> <sample>
  local cfg="$1" style="$2" sample="$3"
  case " $V2 " in *" $sample "*) cfg="${cfg/bench_v3_/bench_v2_}" ;; esac
  [ -f "src/img_2_svg_pretraining/pipeline/configs/$cfg.yaml" ] || return
  docker exec -u "$(id -u):$(id -g)" \
    -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
    "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline export \
       --config src/img_2_svg_pretraining/pipeline/configs/$cfg.yaml \
       --style $style --only $sample --force" \
    > "logs/reexport/${cfg}__${style}__${sample}.log" 2>&1
  local n
  n=$(grep -oE "\([0-9]+ frames" "logs/reexport/${cfg}__${style}__${sample}.log" | head -1 | tr -dc 0-9)
  echo "  ${cfg%%.yaml} $style/$sample -> ${n:-FAIL} frames"
}

for cfg in bench_v3_or_svg bench_v3_or_svg_qwen38_27b bench_v3_or_svg_gemma4_26b bench_v3_zeroshot_gemini31; do
  case "$cfg" in
    *qwen38_27b*) prefix="qwen-qwen3.8-27b" ;;
    *gemma4_26b*) prefix="google-gemma-4-26b-a4b-it" ;;
    *zeroshot*)   prefix="google-gemini-3.1-zeroshot" ;;
    *or_svg)      prefix="google-gemini-3.7-flash" ;;
    *) echo "ABORT: no prefix for $cfg" >&2; exit 2 ;;
  esac
  while read -r line; do
    cell=$(echo "$line" | tr -d ' "'); [ -z "$cell" ] && continue
    style="${cell%%:*}"; sample="${cell##*:}"
    case "$cfg" in *zeroshot*) [ "$style" = "progressive_reveal" ] || continue ;; esac
    ls data/animatebench_v3_cache/*/exports/${prefix}*__svg__*__${style}__*/${sample}/animation.mp4 >/dev/null 2>&1 || continue
    one "$cfg" "$style" "$sample" &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  done < /fsxvision_new/venkat.kesav/img_2_svg_pretraining/data/cells38.txt
done
wait
echo "=== re-export complete"
