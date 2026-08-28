#!/usr/bin/env bash
# Frame metrics (sss + gps + omission) across the study.
#
# INDEPENDENT OF THE VIDEO PASS. These are judged by `qwen38_flash_judge`
# (qwen/qwen3.8-flash on OpenRouter), not by `gemini_video` on Google keys, so
# the free-tier quota exhaustion that throttles video judging does not touch
# them. They can and should run concurrently with scripts/run_video_drip.sh.
#
# MEASURED COLLISION, now guarded against. Both passes write the SAME
# animation.json under --force. Observed: this pass rewrote a cell the video
# drip had just scored, leaving vfs_band=None with an error string from the
# frame judge -- which cannot send video at all. The record then looked like a
# video-judging failure when it was really a lost update.
#
# `run_eval` merges a partial --stages run into the existing record, so the two
# passes CAN share a cell as long as neither requests the other's stages. This
# pass therefore requests only sss/gps/omission and never touches vfs_band or
# ascs_video.
#
# Zeroshot is excluded: it ships a finished animation with no parser XML and no
# sequence, and `run_eval` builds `step_frames_` only when both exist, so sss,
# gps and the omission checklist would all return nothing.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
C=img-2-svg-pretraining-singlenode-venkat.kesav
PY=/environments/img_2_svg_pretraining/bin/python
VROOT=data/animatebench_v3_cache/animatebench_v3/evals_study
KEY=$(grep -E "^OPEN_ROUTER_KEY=" .env | cut -d= -f2- | tr -d '\r\n ')
JOBS="${JOBS:-4}"
mkdir -p logs/scale

done_already() {  # <config-stem> <style> <sample>
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys, pathlib
cfg, style, sample = sys.argv[1:4]
for base in ("animatebench_v3","animatebench_v2"):
    p = pathlib.Path(f"data/animatebench_v3_cache/{base}/evals_study/{cfg}/{style}/{sample}/animation.json")
    if p.exists():
        try:
            if json.loads(p.read_text()).get("sss") is not None: sys.exit(0)
        except Exception: pass
sys.exit(1)
PY
}

run_one() {  # <config-stem> <style> <sample>
  docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$KEY" \
    -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
    "cd /code && PYTHONPATH=src PIPELINE_KEY_FILE=api_keys.csv $PY -u \
       -m img_2_svg_pretraining.animatebench.run_eval animation \
       --config src/img_2_svg_pretraining/pipeline/configs/$1.yaml \
       --style $2 --only $3 --stages sss gps omission --rubric letters \
       --judge-backend qwen38_flash_judge --evals-root $VROOT --force" \
    >> "logs/scale/frame__$1__$2__$3.log" 2>&1
  echo "  done $1 $2/$3 exit=$?"
}

# CONFIGS INTERLEAVED, NOT DRAINED IN ORDER. Run sequentially and the whole of
# Gemini finishes before Gemma starts, so any mid-run read of the metric spread
# is really a read of one model. Round-robin keeps all three arms filling
# together, which is what makes a partial table interpretable.
for cfg in bench_v3_or_svg_gemma4_26b bench_v3_or_svg_qwen38_27b bench_v3_or_svg; do
  case "$cfg" in
    *qwen38_27b*) prefix="qwen-qwen3.8-27b" ;;
    *gemma4_26b*) prefix="google-gemma-4-26b-a4b-it" ;;
    *or_svg)      prefix="google-gemini-3.7-flash" ;;
    *) echo "ABORT: no prefix for $cfg" >&2; exit 2 ;;
  esac
  while read -r line; do
    cell=$(echo "$line" | tr -d ' "'); [ -z "$cell" ] && continue
    style="${cell%%:*}"; sample="${cell##*:}"
    use="$cfg"; case "$sample" in CVPR_2025_pipe000*) use="${cfg/bench_v3_/bench_v2_}" ;; esac
    [ -f "src/img_2_svg_pretraining/pipeline/configs/$use.yaml" ] || continue
    ls data/animatebench_v3_cache/*/exports/${prefix}*__svg__*__${style}__*/${sample}/animation.mp4 >/dev/null 2>&1 || continue
    done_already "$use" "$style" "$sample" && { echo "  have $use $style/$sample"; continue; }
    run_one "$use" "$style" "$sample" &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  done < /fsxvision_new/venkat.kesav/img_2_svg_pretraining/data/cells38.txt
done
wait
echo "=== frame metrics complete"
