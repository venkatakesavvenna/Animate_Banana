#!/usr/bin/env bash
# SSS/GPS/NAS over the generated cells, once their generation run has finished.
#
#   MODEL=dsv4fv SET=abl bash scripts/gen_metrics_night.sh
#
# WAITS FOR THE GENERATOR. A flat export count is NOT a finished run while the
# generator is alive: batches take ~an hour and publish exports only at the end,
# so a plateau check that ignores the generator declares victory early and
# scores a third of the set. The generator's pidfile is the authority.
#
# JUDGE. `kimi_judge` in these configs points at a LOCAL server (port 8011). If
# that is not up the judge calls fail, and a record written from failed calls is
# worse than no record: GPS has a computed component that still emits a score
# when every judge call errored, so a cell can read gps=1.0 built from zero
# judge input, and run_eval will then SKIP that cell forever as "done". This
# script therefore refuses to start without a reachable judge.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
export PYTHONPATH=src

MODEL=${MODEL:?set MODEL}
SET=${SET:?set SET}
STAGES=${STAGES:-"sss gps nas"}
JUDGE=${JUDGE:-kimi_judge}
WORKERS=${WORKERS:-6}
CFG=src/img_2_svg_pretraining/pipeline/configs/gen_${MODEL}_${SET}.yaml
CFGNAME=gen_${MODEL}_${SET}
case "$SET" in
  zs)  MAP=data/zs_style_map.json ;;
  v6)  MAP=data/v6_style_map.json ;;
  abl) MAP=data/abl_style_map.json ;;
esac
GENPID=$REPO/logs/gen_${MODEL}_${SET}/run.pid
LOG=$REPO/logs/metrics_${MODEL}_${SET}; mkdir -p "$LOG/cells"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/driver.log"; }

judge_up(){
  local url
  url=$(python3 -c "
import yaml;d=yaml.safe_load(open('$CFG'))
print(d['backends']['$JUDGE'].get('base_url',''))")
  [ -n "$url" ] && curl -s -m 8 "${url%/}/models" >/dev/null 2>&1
}

say "=== metrics $MODEL/$SET | stages='$STAGES' judge=$JUDGE"

# 1. Wait for generation to finish.
while [ -f "$GENPID" ] && kill -0 "$(cat "$GENPID")" 2>/dev/null; do
  say "generator still alive (pid $(cat "$GENPID")) -- waiting 10min"
  sleep 600
done
say "generator finished"

# 2. Wait for a judge.
until judge_up; do
  say "judge $JUDGE not reachable -- retrying in 15min"
  sleep 900
done
say "judge reachable"

# 3. Score every cell that has an export.
mapfile -t CELLS < <(python3 -c "
import json,glob,os,sys
sys.path.insert(0,'src')
from img_2_svg_pretraining.pipeline.config import load_config
from img_2_svg_pretraining.pipeline.cache import CachePaths
sm=json.load(open('$MAP')); cfg=load_config('$CFG')
for sid,st in sorted(sm.items()):
    cfg.style=st; cfg.raw['animation_style']=st
    p=CachePaths.from_config(cfg)
    if (p.exports(sid)/'animation.mp4').exists(): print(f'{st}:{sid}')
")
say "${#CELLS[@]} cell(s) have exports"

judge_one(){
  local style="${1%%:*}" sid="${1##*:}"
  timeout -k 30 2400 python3 -u -m img_2_svg_pretraining.animatebench.run_eval \
     animation --config "$CFG" --style "$style" --only "$sid" \
     --stages $STAGES --judge-backend "$JUDGE" --rubric letters \
     > "$LOG/cells/${style}__${sid}.log" 2>&1
  echo "$? $style:$sid"
}
export -f judge_one; export CFG STAGES JUDGE LOG

printf '%s\n' "${CELLS[@]}" | xargs -P "$WORKERS" -I{} bash -c 'judge_one "$@"' _ {} \
  | tee -a "$LOG/driver.log" | awk '$1!=0{print "  FAILED "$2}'

say "=== metrics done"
python3 scripts/collect_gen_scores.py --model "$MODEL" --set "$SET" || true
