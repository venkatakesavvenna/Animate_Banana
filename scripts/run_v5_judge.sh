#!/usr/bin/env bash
# SSS + GPS + NAS over every judgeable cell, judged by Kimi K2.6 on 2 nodes.
#
# PARALLEL ACROSS CELLS -- this is the whole point.
#
# docs/ANIMATION_TREE_TELEMETRY.md records the v3 animation tree scoring 45
# cells in 3.8h while the server sat at "Running: 1 reqs" against
# --max-num-seqs 16: ~94% of serving capacity idle, because the driver ran ONE
# run_eval subprocess per cell, sequentially. The judging work is embarrassingly
# parallel -- every (config, style, sample) is independent -- so the only thing
# that ever limited it was requests in flight. The prescribed fix in that
# document ("a process pool over cells, 3.8h -> ~25-30 min") is exactly this.
#
# Each worker still gets the judge's own intra-cell fan-out, so resident
# requests are roughly WORKERS x a few. WORKERS=12 lands near the Running: 32-48
# this server sustains; raising it further just queues.
#
# RESUMABLE: run_eval skips a cell whose record exists unless --force, so
# re-running is a resume. That matters for an overnight run -- a transient
# failure must not mean starting again.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
C=animatebanana-v5
PY=/environments/img_2_svg_pretraining/bin/python
LOG=$REPO/logs/bench_v5/judge; mkdir -p "$LOG"
WORKERS=${WORKERS:-12}
STAGES=${STAGES:-"sss gps nas"}
CELLS=${CELLS:-$REPO/data/v5_judge_cells.json}

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/driver.log"; }

# One cell. Prints a single status line so the log stays readable at 446 rows.
judge_one() {
  local cfg="$1" style="$2" sample="$3"
  local tag="${cfg}__${style}__${sample}"
  local out="$LOG/cells/$tag.log"
  mkdir -p "$LOG/cells"
  # timeout: a cell is ~50 judge calls at seconds each; 40 min means the cell is
  # wedged (engine deadlock leaves the client blocked in a read forever). -k
  # covers a SIGTERM the docker exec ignores.
  timeout -k 30 2400 \
  docker exec -u "$(id -u):$(id -g)" -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
    "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.animatebench.run_eval \
       animation --config src/img_2_svg_pretraining/pipeline/configs/${cfg}.yaml \
       --style $style --only $sample --stages $STAGES --judge-backend kimi_judge" \
    >"$out" 2>&1
  echo "$? $tag"
}
export -f judge_one
export C PY LOG STAGES

# PURGE ERROR STUBS FIRST. run_eval skips any cell whose record exists, so a
# record written while the server was down ("Connection error" in *_errors)
# permanently masks that cell from every future resume. And a PARTIAL such
# record is worse than an empty one: GPS has a computed component that still
# emits a score when every judge call failed, so a cell can read gps=1.0
# built from zero judge input. Any record carrying connection failures dies,
# partial scores included -- re-running is cheap because successful judge
# calls replay from the response cache.
python3 - <<'PURGE'
import json, glob, os
purged = 0
for f in glob.glob("data/animatebench_v5*cache/animatebench_v5/evals/*/*/*/animation.json"):
    try: d = json.load(open(f))
    except Exception: os.unlink(f); purged += 1; continue
    errs = str(d.get("sss_errors", "")) + str(d.get("gps_errors", "")) + str(d.get("nas_errors", ""))
    if "Connection error" in errs or "judge failed" in errs or "timed out" in errs:
        os.unlink(f); purged += 1
print(f"purged {purged} error-stub record(s)")
PURGE

say "=== v5 judging | stages: $STAGES | workers: $WORKERS"
mapfile -t ROWS < <(python3 -c "
import json;
[print(f\"{r['config']} {r['style']} {r['sample']}\") for r in json.load(open('$CELLS'))]")
say "  cells: ${#ROWS[@]}"

printf '%s\n' "${ROWS[@]}" \
  | xargs -P "$WORKERS" -I{} bash -c 'judge_one $0' {} \
  | while read -r rc tag; do
      [ "$rc" = 0 ] && say "  ok   $tag" || say "  FAIL $tag (rc=$rc)"
    done

say "=== judging pass complete"
