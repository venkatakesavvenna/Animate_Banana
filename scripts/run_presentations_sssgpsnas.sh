#!/usr/bin/env bash
# SSS/GPS/NAS over the 35 Original_Presentations talks, judged by Kimi K2.6.
#
# JUDGE MUST BE NAMED IN THE CONFIG. `--judge-backend kimi_judge` does NOT fail
# when that backend is absent -- it falls back to a Gemini judge and says so in
# one line of stdout, producing scores that look fine and are not comparable
# with the ablations or the DeepSeek sets. kimi_judge was added to
# presentations_video.yaml for exactly this reason; the guard below refuses to
# start if it ever goes missing again.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"; export PYTHONPATH=src
CFG=src/img_2_svg_pretraining/pipeline/configs/presentations_video.yaml
LOG=$REPO/logs/pres_sssgpsnas; mkdir -p "$LOG/cells"
JOBS=${JOBS:-3}

python3 -c "
import yaml,sys
d=yaml.safe_load(open('$CFG'))
sys.exit(0 if 'kimi_judge' in d['backends'] else 1)" \
  || { echo "FATAL: kimi_judge missing from $CFG"; exit 1; }
curl -s -m 8 http://127.0.0.1:8011/v1/models | grep -q Kimi \
  || { echo "FATAL: Kimi not reachable on 8011"; exit 1; }

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/driver.log"; }
mapfile -t CELLS < <(python3 -c "
import json
for k,v in sorted(json.load(open('data/presentations_style_map.json')).items()): print(f'{v}:{k}')")
say "=== ${#CELLS[@]} cells, JOBS=$JOBS, judge=kimi_judge"

one(){
  local style="${1%%:*}" sid="${1##*:}"
  # HEALTH GATE PER CELL. When the 2-node server dies (a worker node dropping
  # takes the whole engine with it), run_eval does NOT fail fast -- it retries,
  # then WRITES a record whose *_errors are all "Connection error". run_eval
  # skips any cell that has a record, so that stub masks the cell from every
  # future resume, and GPS can still emit a computed score built from zero judge
  # input. One node dropping at 08:14 poisoned 19 cells exactly this way.
  # Waiting here costs minutes; a poisoned record costs the cell permanently.
  for _ in $(seq 1 60); do
    curl -s -m 6 http://127.0.0.1:8011/v1/models 2>/dev/null | grep -q Kimi && break
    sleep 30
  done
  curl -s -m 6 http://127.0.0.1:8011/v1/models 2>/dev/null | grep -q Kimi \
    || { echo "JUDGE-DOWN $sid"; return; }
  timeout -k 30 3000 python3 -u -m img_2_svg_pretraining.animatebench.run_eval \
    animation --config "$CFG" --style "$style" --only "$sid" \
    --stages sss gps nas --rubric letters --judge-backend kimi_judge --force \
    > "$LOG/cells/${style}__${sid}.log" 2>&1
  # A record whose judge was NOT Kimi is a silent miscomparison; flag it loudly.
  if ! grep -q "judge: moonshotai/Kimi-K2.6" "$LOG/cells/${style}__${sid}.log"; then
    echo "WRONG-JUDGE $sid"
  # "animation: cached" ALSO matches a bare `animation:` grep, so a run that
  # skipped every cell reports 35 x ok in seconds. Require a real score line.
  elif grep -qE "animation: .*(sss=|gps=|nas=)" "$LOG/cells/${style}__${sid}.log"; then echo "ok $sid"
  elif grep -q "animation: cached" "$LOG/cells/${style}__${sid}.log"; then echo "CACHED-SKIP $sid"
  else echo "FAIL $sid"; fi
}
export -f one; export CFG LOG
printf '%s\n' "${CELLS[@]}" | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {} \
  | tee -a "$LOG/driver.log" | grep -E "FAIL|WRONG-JUDGE|CACHED-SKIP|JUDGE-DOWN" || true
say "=== done"
python3 scripts/collect_presentation_scores.py || true
