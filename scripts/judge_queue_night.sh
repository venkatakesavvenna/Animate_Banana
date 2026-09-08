#!/usr/bin/env bash
# Judge everything pending, in the user's stated order, against local Kimi K2.6.
#
#   1. ablation gap        108 cells (04_Narration 18, 07_Stage3 90)
#   2. DeepSeek V4 outputs 441 cells (abl 85, zs 166, v6 190)
#   3. Gemini 3.1 zero-shot  -- see NOTE below
#
# PER-CELL HEALTH GATE. The 2-node server dies when either node loses a process
# (twice on the previous pair). run_eval does NOT fail fast on that: it retries,
# then writes a record whose *_errors are all "Connection error" -- and because
# run_eval skips any cell that HAS a record, that stub masks the cell from every
# future resume, while GPS can still emit a computed score from zero judge
# input. Waiting for a healthy judge costs minutes; a poisoned record costs the
# cell permanently.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"; export PYTHONPATH=src
PORT=${PORT:-8011}
JOBS=${JOBS:-4}
LOG=$REPO/logs/judge_queue; mkdir -p "$LOG/cells"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/driver.log"; }

judge_up(){ curl -s -m 8 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q Kimi; }
wait_judge(){
  for _ in $(seq 1 120); do judge_up && return 0; sleep 30; done
  return 1
}

one(){   # cfg style sample
  local cfg="$1" style="$2" sid="$3" tag out
  tag="${cfg}__${style}__${sid}"; out="$LOG/cells/$tag.log"
  for _ in $(seq 1 120); do
    curl -s -m 6 "http://127.0.0.1:8011/v1/models" 2>/dev/null | grep -q Kimi && break
    sleep 30
  done
  curl -s -m 6 "http://127.0.0.1:8011/v1/models" 2>/dev/null | grep -q Kimi \
    || { echo "JUDGE-DOWN $tag"; return; }
  timeout -k 30 3000 python3 -u -m img_2_svg_pretraining.animatebench.run_eval \
      animation --config "src/img_2_svg_pretraining/pipeline/configs/${cfg}.yaml" \
      --style "$style" --only "$sid" --stages sss gps nas \
      --rubric letters --judge-backend kimi_judge --force > "$out" 2>&1
      # --force IS REQUIRED, not optional. Purging the poisoned sss/gps/nas
      # fields left each record in place (its vfs_band/ascs_video were real and
      # worth keeping), and run_eval skips ANY cell that has a record -- so
      # without --force those cells report "animation: cached" forever and are
      # never re-judged. A partial --stages run MERGES into the existing record
      # (run_eval.py), so the video metrics survive this.
  if ! grep -q "judge: moonshotai/Kimi-K2.6" "$out"; then echo "WRONG-JUDGE $tag"
  elif grep -qE "animation: .*(sss=|gps=|nas=)|animation: written" "$out"; then echo "ok $tag"
  elif grep -q "animation: cached" "$out"; then echo "CACHED $tag"
  else echo "FAIL $tag"; fi
}
export -f one; export LOG

wait_judge || { say "FATAL: judge never came up"; exit 1; }
say "judge reachable"

# --- 1. ablation gap -------------------------------------------------------
say "=== [1/3] ablation gap"
python3 -c "
import json
for c in json.load(open('data/ablation_gap_cells.json')):
    print(c['config'],c['style'],c['sample'])" \
 | xargs -P "$JOBS" -L1 bash -c 'one "$@"' _ \
 | tee -a "$LOG/driver.log" | grep -E "FAIL|JUDGE-DOWN|WRONG-JUDGE" || true
say "=== [1/3] done"

# --- 2. DeepSeek V4 --------------------------------------------------------
say "=== [2/3] DeepSeek V4 outputs"
for SET in abl zs v6; do
  case "$SET" in
    zs)  MAP=data/zs_style_map.json ;;
    v6)  MAP=data/v6_style_map.json ;;
    abl) MAP=data/abl_style_map.json ;;
  esac
  say "  -- dsv4fv/$SET"
  python3 -c "
import json,sys
sys.path.insert(0,'src')
from img_2_svg_pretraining.pipeline.config import load_config
from img_2_svg_pretraining.pipeline.cache import CachePaths
cfg=load_config('src/img_2_svg_pretraining/pipeline/configs/gen_dsv4fv_$SET.yaml')
for sid,st in sorted(json.load(open('$MAP')).items()):
    cfg.style=st; cfg.raw['animation_style']=st
    if (CachePaths.from_config(cfg).exports(sid)/'animation.mp4').exists():
        print('gen_dsv4fv_$SET',st,sid)" \
   | xargs -P "$JOBS" -L1 bash -c 'one "$@"' _ \
   | tee -a "$LOG/driver.log" | grep -E "FAIL|JUDGE-DOWN|WRONG-JUDGE" || true
done
say "=== [2/3] done"

# --- 3. Gemini 3.1 zero-shot ----------------------------------------------
# NOTE: only 8 exports exist (5 in animatebench_v2, 3 in v3) and they carry NO
# xml and NO sequence, so SSS/GPS/NAS cannot gate on them as they stand. Left
# out of this queue deliberately rather than emitting empty records.
say "=== [3/3] gemini-3.1 zero-shot SKIPPED -- no xml/sequence; see driver notes"

say "=== QUEUE COMPLETE"
python3 scripts/collect_gen_scores.py --model dsv4fv --set abl || true
python3 scripts/collect_gen_scores.py --model dsv4fv --set zs  || true
python3 scripts/collect_gen_scores.py --model dsv4fv --set v6  || true
