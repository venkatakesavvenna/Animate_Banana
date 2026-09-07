#!/usr/bin/env bash
# Wait for the zs Gemini generation to finish, then run SSS/GPS/NAS over it
# with the local Kimi K2.6 judge. Costs nothing on OpenRouter -- the judge is
# the 2-node local server, so this is free regardless of how long it runs.
#
# PHASE 1 waits for generation. "Complete" is EITHER exports == TARGET, or the
# count sitting unchanged for PLATEAU_POLLS consecutive polls. The plateau arm
# matters: a run that ends with some cells failed never reaches TARGET, and
# waiting for a number that can no longer arrive would hang until morning.
#
# PHASE 2 judges, supervised. Three real failure modes on this stack:
#   * server gone         -> port 8011 stops listening
#   * ENGINE DEADLOCK     -> port answers, requests "running", but
#     vllm:generation_tokens_total does not move. Port-up is NOT engine-alive.
#   * driver exited early -> relaunch; run_eval skips finished cells, so a
#     restart costs only the unfinished ones.
#
# Restarts go through serve_kimi_lustre.sh with HUB set. serve_kimi_mn.sh and
# serve_kimi_stable.sh hardcode an NVMe path that is empty or another user's on
# these nodes, and wedge with a 0-byte log rather than failing loudly.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
# NODES ARE CHOSEN AT JUDGING TIME, NOT BAKED IN. Both nodes that served Kimi
# earlier were taken over by other users while it was down (see
# scripts/pick_free_nodes.sh). HEAD/WORKER are discovered below and re-discovered
# on every restart; set them explicitly only to override.
HEAD=${HEAD:-}; WORKER=${WORKER:-}
C=${C:-animatebanana-mn}
HUB=${HUB:-/opt/dlami/nvme/vkkimi/hf_cache}
CFG=${CFG:-}
CACHE=${CACHE:-$REPO/data/animatebench_v6_cache}
SUITE=${SUITE:-animatebench_v6_ds}
# no fixed STYLE: it is per sample (zs_style_map.json / lineage)
TARGET=${TARGET:-216}
PLATEAU_POLLS=${PLATEAU_POLLS:-6}
# A plateau below MIN_PLATEAU is NOT "generation finished". This cache already
# held 1 export from an earlier partial run, and a flat count of 1 would
# otherwise satisfy the plateau arm within 30 minutes -- declaring a run
# complete before it had started, then judging one cell and exiting "done".
MIN_PLATEAU=${MIN_PLATEAU:-20}
WORKERS=${WORKERS:-6}
CELLS=$REPO/data/v6_judge_cells.json
LOG=$REPO/logs/v6_metrics; mkdir -p "$LOG"
SSH="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no"

PID=$LOG/night.pid
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "already running as pid $(cat "$PID")"; exit 0
fi
echo $$ > "$PID"; trap 'rm -f "$PID"' EXIT

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/night.log"; }
exports(){ find "$CACHE" -path '*exports*' -name 'animation.mp4' 2>/dev/null | wc -l; }
records(){ find "$CACHE" -path '*evals*' -name 'animation.json' 2>/dev/null | wc -l; }

# ---------- phase 1: wait for generation ----------
say "phase 1: waiting for generation (target=$TARGET)"
LAST=-1; FLAT=0
while :; do
  N=$(exports)
  if [ "$N" -ge "$TARGET" ]; then say "generation complete: $N/$TARGET"; break; fi
  # A FLAT COUNT IS NOT A FINISHED RUN WHILE THE GENERATOR IS STILL ALIVE.
  # Batches take ~60 min and publish their exports only at the end, so exports
  # sit unchanged for far longer than the plateau window during normal work.
  # On 2026-09-04 this fired mid-run, sent the supervisor into bring-up, and it
  # exhausted its retries and exited claiming "generation is done" while 106
  # cells were still to come. The runner's own pidfile is the authority.
  GEN_ALIVE=0
  if [ -f "$REPO/logs/v6/run.pid" ] && kill -0 "$(cat "$REPO/logs/v6/run.pid" 2>/dev/null)" 2>/dev/null; then GEN_ALIVE=1; fi
  if [ "$GEN_ALIVE" = 1 ]; then
    say "exports=$N/$TARGET (generator alive -- not a plateau)"
    LAST=$N; FLAT=0; sleep 300; continue
  fi
  if [ "$N" = "$LAST" ] && [ "$N" -ge "$MIN_PLATEAU" ]; then
    FLAT=$((FLAT+1))
    say "exports=$N/$TARGET (flat $FLAT/$PLATEAU_POLLS)"
    [ "$FLAT" -ge "$PLATEAU_POLLS" ] && { say "plateau -- treating $N as final"; break; }
  else
    FLAT=0; say "exports=$N/$TARGET$([ "$N" -lt "$MIN_PLATEAU" ] && echo " (below plateau floor $MIN_PLATEAU)")"
  fi
  LAST=$N
  sleep 300
done

# ---------- build cells ----------
CACHE="$CACHE" SUITE="$SUITE" CELLS="$CELLS" python3 - <<'PY'
import glob,os,json,collections
cache,suite,out=(os.environ[k] for k in ("CACHE","SUITE","CELLS"))
# v6 runs FOUR models into ONE cache_root, so cells must be enumerated per
# config. The exports tree is keyed by lineage, and the lineage encodes both
# the model and the style -- so it, not any config default, is the authority
# on what was actually produced.
STYLES=("progressive_reveal","hopping_bounding_box","sliding_bounding_box","colour_pop","alpha_masking")
MODEL_LINEAGE={
 "bench_v6_svg_qwen38_27b":"Qwen-Qwen3.8-27B",
 "bench_v6_svg_glm46v":"zai-org-GLM-4.6V",
 "bench_v6_svg_gemma4_31b":"google-gemma-4-31B-it",
 "bench_v6_svg_qwen3vl235b":"Qwen-Qwen3-VL-235B-A22B-Instruct",
}
smap=json.load(open('data/v6_style_map.json'))
cells=[]; per=collections.Counter()
for mp4 in glob.glob(f"{cache}/{suite}/exports/*/*/animation.mp4"):
    parts=mp4.split(os.sep); sample=parts[-2]; lineage=parts[-3]
    style=next((t for t in STYLES if f"__{t}__" in lineage), None) or smap.get(sample)
    cfg=next((c for c,tag in MODEL_LINEAGE.items() if tag in lineage), None)
    if not style or not cfg: continue
    rec=f"{cache}/{suite}/evals/{cfg}/{style}/{sample}/animation.json"
    if os.path.exists(rec): continue
    cells.append({"config":cfg,"style":style,"sample":sample}); per[cfg]+=1
cells.sort(key=lambda c:(c["config"],c["style"],c["sample"]))
json.dump(cells,open(out,'w'),indent=1)
print(f"cells to judge: {len(cells)}")
for k,v in per.most_common(): print(f"  {k:<34} {v}")
PY
say "cells file: $CELLS ($(python3 -c "import json;print(len(json.load(open('$CELLS'))))") cells)"

# ---------- phase 2: judge, supervised ----------
server_up(){ local p; p=$($SSH "$HEAD" 'ss -lnt 2>/dev/null | grep -c ":8011"' 2>/dev/null|tail -1); [ "${p:-0}" -gt 0 ] 2>/dev/null; }
snap(){ $SSH "$HEAD" "docker exec $C bash -lc 'curl -s -m 10 http://127.0.0.1:8011/metrics'" 2>/dev/null | awk '
  index($1,"vllm:num_requests_running")==1{r=$2} index($1,"vllm:generation_tokens_total")==1{g=$2}
  END{if(r!=""||g!="")printf "%.0f %.0f\n",r,g}'; }
driver_up(){ local n; n=$($SSH "$HEAD" 'pgrep -fc "run_v5_[j]udge.sh"' 2>/dev/null|tail -1); [ "${n:-0}" -gt 0 ] 2>/dev/null; }
# Find two usable nodes and stand Kimi up on them from scratch: containers,
# Ray, then the server. Weights come from shared Lustre (~50-60 min load) rather
# than node-local NVMe, because a freshly chosen node has no staged copy and an
# empty NVMe cache does not raise -- it hangs with a 0-byte log.
bring_up(){
  local pick n1 n2 img1 img2
  pick=$(NEED=2 MAXMEM=60000 bash "$REPO/scripts/pick_free_nodes.sh" 2>/dev/null)
  n1=$(echo "$pick"|sed -n 1p|awk '{print $1}'); img1=$(echo "$pick"|sed -n 1p|awk '{print $2}')
  n2=$(echo "$pick"|sed -n 2p|awk '{print $1}'); img2=$(echo "$pick"|sed -n 2p|awk '{print $2}')
  if [ -z "$n1" ] || [ -z "$n2" ]; then
    say "no two free nodes right now (found: ${n1:-none} ${n2:-none}) -- will retry"
    return 1
  fi
  HEAD=$n1; WORKER=$n2
  say "bringing kimi up on $HEAD (head) + $WORKER (worker)"
  $SSH "$HEAD"   "C=$C IMAGE=$img1 bash $REPO/scripts/remote/init_mn.sh" >/dev/null 2>&1
  $SSH "$WORKER" "C=$C IMAGE=$img2 bash $REPO/scripts/remote/init_mn.sh" >/dev/null 2>&1
  HEAD="$HEAD" WORKER="$WORKER" C="$C" bash "$REPO/scripts/remote/serve_kimi_lustre.sh" >/dev/null 2>&1
  for i in $(seq 1 140); do
    sleep 30
    $SSH "$HEAD" "docker exec $C bash -lc 'curl -s -m 5 http://127.0.0.1:8011/v1/models'" 2>/dev/null | grep -q Kimi \
      && { say "kimi serving on $HEAD"; return 0; }
    if $SSH "$HEAD" "docker exec $C bash -lc 'grep -qE \"ActorHandleNotFound|Engine core initialization failed|No available memory\" /tmp/vllm_kimi_mn.log'" 2>/dev/null; then
      say "kimi failed to init on $HEAD (node likely taken) -- will re-pick"; return 1
    fi
  done
  say "kimi did not come up in 70min"; return 1
}

restart_server(){
  say "restarting kimi (re-picking nodes)"
  for a in 1 2 3 4 5 6 7 8; do
    bring_up && return 0
    say "bring-up attempt $a failed; waiting 15min for nodes"
    sleep 900
  done
  return 1
}
start_driver(){
  say "starting judge driver"
  $SSH "$HEAD" "cd $REPO && setsid nohup env C=$C WORKERS=$WORKERS CELLS=$CELLS \
     EVALS_GLOB='$CACHE' bash $REPO/scripts/run_v5_judge.sh \
     > $REPO/logs/v6_metrics/driver.log 2>&1 < /dev/null &" >/dev/null 2>&1; }

TOT=$(python3 -c "import json;print(len(json.load(open('$CELLS'))))")
[ "$TOT" -eq 0 ] && { say "nothing to judge"; exit 0; }
if [ -z "$HEAD" ] || ! server_up; then
  # DO NOT EXIT ON MISSING CAPACITY. Judging costs nothing and the cluster
  # turns over; exiting just means nobody is watching when nodes free up.
  until restart_server; do say "still no capacity -- retrying in 30min"; sleep 1800; done
fi
start_driver
FROZEN=0; LASTG=-1
while :; do
  R=$(records); say "records=$R (target $TOT new)"
  [ "$(python3 -c "import json,os,glob;c=json.load(open('$CELLS'));print(sum(1 for x in c if not os.path.exists('$CACHE/$SUITE/evals/'+x['config']+'/'+x['style']+'/'+x['sample']+'/animation.json')))")" = "0" ] && { say "ALL ZS CELLS JUDGED"; break; }
  if ! server_up; then say "server down"; restart_server && start_driver; sleep 300; continue; fi
  S=$(snap); RR=$(echo "$S"|awk '{print $1}'); G=$(echo "$S"|awk '{print $2}')
  if [ -n "${G:-}" ] && [ "${RR:-0}" -gt 0 ] 2>/dev/null; then
    if [ "$G" = "$LASTG" ]; then FROZEN=$((FROZEN+1)); say "tokens frozen ($FROZEN) running=$RR"
      if [ "$FROZEN" -ge 5 ]; then say "ENGINE DEADLOCK -- restarting"
        $SSH "$HEAD" "pkill -f 'run_v5_[j]udge.sh'" >/dev/null 2>&1
        restart_server && start_driver; FROZEN=0; fi
    else FROZEN=0; fi
    LASTG=$G
  fi
  driver_up || { say "driver gone; relaunching"; start_driver; }
  sleep 300
done
