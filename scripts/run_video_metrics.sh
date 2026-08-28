#!/usr/bin/env bash
# Video metrics (vfs_band + ascs_video) for every model, over the study cells.
#
# Judged by `gemini_video_paid` -- gemini-3.7-flash reached through OpenRouter,
# which DOES accept video as a `video_url` part carrying a base64 data URL
# (verified live: the usage breakdown returns video_tokens). The native Google
# route works too but its free keys are 20 requests/DAY and were exhausted.
#
# THE KEY MUST BE PASSED INTO THE CONTAINER. The earlier drip omitted
# `-e OPEN_ROUTER_KEY` and every call died with `401 Missing Authentication
# header` -- which surfaced only as an empty record, and looked exactly like a
# quota problem. That is why this script exists rather than a bespoke loop.
#
# ONLY vfs_band + ascs_video are requested here. `run_eval --force` rewrites the
# whole record, so a pass that also asked for sss/gps would wipe whatever the
# frame pass wrote for this cell, and vice versa. Requesting disjoint stages is
# what lets the two passes run concurrently on the same cell safely.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
C=img-2-svg-pretraining-singlenode-venkat.kesav
PY=/environments/img_2_svg_pretraining/bin/python
VROOT=data/animatebench_v3_cache/animatebench_v3/evals_study
KEY=$(grep -E "^OPEN_ROUTER_KEY=" .env | cut -d= -f2- | tr -d '\r\n ')
JOBS="${JOBS:-4}"
mkdir -p logs/scale

have_video() {  # <config-stem> <style> <sample>
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys, pathlib
cfg, style, sample = sys.argv[1:4]
for base in ("animatebench_v3","animatebench_v2"):
    p = pathlib.Path(f"data/animatebench_v3_cache/{base}/evals_study/{cfg}/{style}/{sample}/animation.json")
    if p.exists():
        try:
            if json.loads(p.read_text()).get("vfs_band") is not None: sys.exit(0)
        except Exception: pass
sys.exit(1)
PY
}

for cfg in bench_v3_or_svg bench_v3_or_svg_qwen38_27b bench_v3_or_svg_gemma4_26b bench_v3_zeroshot_gemini31; do
  case "$cfg" in
    *qwen38_27b*) prefix="qwen-qwen3.8-27b" ;;
    *gemma4_26b*) prefix="google-gemma-4-26b-a4b-it" ;;
    *zeroshot*)   prefix="google-gemini-3.1-zeroshot" ;;
    *or_svg)      prefix="google-gemini-3.7-flash" ;;
    *) echo "ABORT: no lineage prefix for $cfg" >&2; exit 2 ;;
  esac
  while read -r line; do
    cell=$(echo "$line" | tr -d ' "'); [ -z "$cell" ] && continue
    style="${cell%%:*}"; sample="${cell##*:}"
    use="$cfg"; case "$sample" in CVPR_2025_pipe000*) use="${cfg/bench_v3_/bench_v2_}" ;; esac
    [ -f "src/img_2_svg_pretraining/pipeline/configs/$use.yaml" ] || continue
    # Zeroshot implements progressive_reveal in every shipped SVG (measured:
    # opacity 0->1, fadeIn, forwards). Judging it against a sample's
    # sliding/alpha GT style would score the style mismatch, not the model.
    case "$cfg" in *zeroshot*) [ "$style" = "progressive_reveal" ] || continue ;; esac
    ls data/animatebench_v3_cache/*/exports/${prefix}*__svg__*__${style}__*/${sample}/animation.mp4 >/dev/null 2>&1 || continue
    have_video "$use" "$style" "$sample" && { echo "  have $use $style/$sample"; continue; }
    {
      docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$KEY" \
        -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
        "cd /code && PYTHONPATH=src PIPELINE_KEY_FILE=api_keys.csv $PY -u \
           -m img_2_svg_pretraining.animatebench.run_eval animation \
           --config src/img_2_svg_pretraining/pipeline/configs/$use.yaml \
           --style $style --only $sample --stages vfs_band ascs_video \
           --judge-backend gemini_video_paid --evals-root $VROOT --force" \
        >> "logs/scale/vid__${use}__${style}__${sample}.log" 2>&1
      if have_video "$use" "$style" "$sample"; then echo "  OK   $use $style/$sample"
      else echo "  FAIL $use $style/$sample"; fi
    } &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  done < /fsxvision_new/venkat.kesav/img_2_svg_pretraining/data/cells38.txt
done
wait
echo "=== video metrics pass complete"
