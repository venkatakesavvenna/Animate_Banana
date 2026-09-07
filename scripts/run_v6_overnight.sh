#!/usr/bin/env bash
# v6 (216 samples) x open-weights models: serve, generate, then evals.
#
# SERVER AND CLIENT BOTH RUN ON THE CHOSEN NODE. This box (ip-10-20-238-191)
# has no free GPUs, and serve_v5.sh serves on 127.0.0.1 with the config's
# base_url pointing there -- so serving remotely while running the client here
# would need base_url surgery on every config. Running both inside
# animatebanana-mn on the target node keeps base_url correct and needs no
# networking: that container mounts /code and /environments from Lustre.
#
# STYLE IS PER SAMPLE (data/v6_style_map.json, built from buffer_items.csv).
# The configs carry only a default animation_style; generating everything at
# the default would put 143 of 216 samples in a style they were never meant to
# have, and every style-sensitive metric downstream would compare unlike things.
#
# NODES ARE RE-PICKED, never hardcoded: nodes were taken over by other users
# repeatedly on 09-03/04, and vLLM asking for 90% of an occupied GPU dies as a
# confusing ActorHandleNotFoundError minutes later.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
C=${C:-animatebanana-mn}
PY=/environments/img_2_svg_pretraining/bin/python
VENV=${VENV:-/environments/v5serve}
MODELS=${MODELS:-"qwen38_27b glm46v gemma4_31b qwen3vl235b"}
BATCH=${BATCH:-8}
JOBS=${JOBS:-2}
PORT=8011
# EXPORT IS DELIBERATELY EXCLUDED (--to critique-animation). The exporter
# rasterises frames through headless Playwright, which cannot launch in the
# vlm-ingest image used on the serving nodes:
#   TargetClosedError: BrowserType.launch: Target page, context or browser has been closed
# Every model stage succeeds there; only the browser step fails. Export is
# CPU-only and works in this box's own container, so scripts/v6_export_daemon.sh
# does it locally while generation continues on the GPU node.
# CTX MUST EXCEED THE CONFIGS' max_tokens (65536). Serving at 32768 made vLLM
# reject EVERY request with
#   400 max_tokens=65536 cannot be greater than max_model_len=32768
# and the runner walked all 216 samples in 40 seconds logging one "nonzero" per
# batch -- a total no-op that reads like progress in the log. serve_v5.sh
# defaults to 131072 for exactly this reason.
CTX=${CTX:-131072}
LOG=$REPO/logs/v6; mkdir -p "$LOG"
SSH="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no"
PID=$LOG/run.pid
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > "$PID"; trap 'rm -f "$PID"' EXIT
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/run.log"; }

pick_node(){ NEED=1 MAXMEM=60000 bash "$REPO/scripts/pick_free_nodes.sh" 2>/dev/null | head -1; }

NODELINE=$(pick_node)
NODE=$(echo "$NODELINE" | awk '{print $1}'); IMG=$(echo "$NODELINE" | awk '{print $2}')
[ -n "$NODE" ] || { say "FATAL: no free node"; exit 1; }
say "=== v6 overnight | node=$NODE image=$IMG | models: $MODELS"
$SSH "$NODE" "C=$C IMAGE=$IMG bash $REPO/scripts/remote/init_mn.sh" >/dev/null 2>&1
say "container ready on $NODE"

dex(){ $SSH "$NODE" "docker exec -u \$(id -u):\$(id -g) -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers $C bash -lc \"cd /code && PYTHONPATH=src $1\""; }

# TEARDOWN MUST WAIT FOR THE GPUs TO ACTUALLY DRAIN, not just signal the
# process. vLLM workers hold device memory well after SIGKILL; `pkill` + a flat
# sleep looks like it worked and isn't. Measured 2026-09-05: qwen38_27b still
# held 74GB on all 8 GPUs ten seconds after teardown, so glm46v died with
#   ValueError: Free memory on device cuda:1 (4.37/79.18 GiB) on startup is
#   less than desired GPU memory utilization
# surfacing only as the generic "Engine core initialization failed" -- the real
# error lives in worker stderr, which the API-server wrapper never echoes.
serve_stop(){
  # RESTART THE CONTAINER; do not try to kill the workers. Processes inside
  # animatebanana-mn run as the IMAGE's user (raghuveer.r), so from the host we
  # lack permission to signal them -- `kill -9` silently no-ops -- and an
  # in-container pkill did not reach them either. Measured 2026-09-05: 595GB
  # stayed pinned across five minutes of both. `docker restart` frees all 8
  # GPUs in ~25s and is the only teardown that works here.
  $SSH "$NODE" "docker restart $C" >/dev/null 2>&1
  sleep 20
  for i in $(seq 1 40); do
    sleep 15
    m=$($SSH "$NODE" 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|paste -sd+|bc' 2>/dev/null|tail -1)
    if [ "${m:-999999}" -lt 8000 ] 2>/dev/null; then say "gpus drained (${m}MiB) after $((i*15))s"; return 0; fi
  done
  say "WARNING: gpus did not drain (${m:-?}MiB) -- next serve will likely fail"
}

serve_start(){  # $1 = model stem
  local cfg=src/img_2_svg_pretraining/pipeline/configs/bench_v6_svg_$1.yaml
  local rid tp
  rid=$(python3 -c "import yaml;print(yaml.safe_load(open('$cfg'))['backends']['served']['model'])")
  tp=8; [ "$1" = glm46v ] && tp=4      # 12 attention heads: 12 % 8 != 0, TP must divide it
  say "serving $1 ($rid) TP=$tp on $NODE"
  $SSH "$NODE" "docker exec -d -e HF_HUB_OFFLINE=0 $C bash -lc '$VENV/bin/python -m vllm.entrypoints.openai.api_server \
      --model \"$rid\" --served-model-name \"$rid\" --trust-remote-code \
      --tensor-parallel-size $tp --gpu-memory-utilization 0.90 \
      --max-model-len ${CTX:-131072} --limit-mm-per-prompt.image 2 --max-num-seqs 16 \
      --port $PORT --host 127.0.0.1 > /tmp/vllm_$1.log 2>&1'"
  # LOAD BUDGET SCALES WITH CHECKPOINT SIZE. 240x15s = 60 min is ample for the
  # 54-212GB models but NOT for Qwen3-VL-235B (495GB on Lustre): on 2026-09-05
  # it was still streaming weights at 60 min, the wait expired, teardown ran,
  # and the interrupted load surfaced as "RuntimeError: cancelled" -- which
  # reads like a model fault and was really our own timeout.
  local waits=240; case "$1" in qwen3vl235b) waits=600;; esac
  for i in $(seq 1 $waits); do
    sleep 15
    $SSH "$NODE" "docker exec $C bash -lc 'curl -s -m 5 http://127.0.0.1:$PORT/v1/models'" 2>/dev/null | grep -q "$rid" && { say "$1 serving"; return 0; }
    $SSH "$NODE" "docker exec $C bash -lc 'grep -qE \"Error|Traceback|assert\" /tmp/vllm_$1.log'" 2>/dev/null && {
      say "$1 FAILED to serve:"; $SSH "$NODE" "docker exec $C bash -lc 'tail -6 /tmp/vllm_$1.log'" 2>/dev/null | tee -a "$LOG/run.log"; return 1; }
  done
  say "$1 serve timeout after $((waits*15/60))min"; return 1
}

for M in $MODELS; do
  [ -f "$LOG/$M.done" ] && { say "== $M already done"; continue; }
  CFG=src/img_2_svg_pretraining/pipeline/configs/bench_v6_svg_$M.yaml
  serve_stop
  serve_start "$M" || { say "== $M SKIPPED (serve failed)"; touch "$LOG/$M.failed"; continue; }
  for STYLE in progressive_reveal hopping_bounding_box alpha_masking sliding_bounding_box colour_pop; do
    mapfile -t ARR < <(STYLE="$STYLE" python3 -c "
import json,os
sm=json.load(open('data/v6_style_map.json'))
print('\n'.join(sorted(s for s,v in sm.items() if v==os.environ['STYLE'])))")
    [ "${#ARR[@]}" -eq 0 ] && continue
    say "== $M / $STYLE : ${#ARR[@]} sample(s)"
    i=0
    while [ $i -lt ${#ARR[@]} ]; do
      running=0
      while [ $i -lt ${#ARR[@]} ] && [ $running -lt $JOBS ]; do
        b=$(printf "%s " "${ARR[@]:$i:$BATCH}")
        say "   $M $STYLE batch $((i/BATCH)) ($(echo $b|wc -w))"
        ( if dex "$PY -u -m img_2_svg_pretraining.pipeline.run_pipeline all --to critique-animation --config $CFG --style $STYLE --only $b" \
              >>"$LOG/pipe_${M}_${STYLE}.log" 2>&1; then :; else
             say "   $M $STYLE batch $((i/BATCH)) nonzero"; echo x >> "$LOG/.fails_$M"; fi ) &
        i=$((i+BATCH)); running=$((running+1))
      done
      wait
      # ABORT A DEAD RUN rather than walking the corpus. Every batch failing
      # instantly is a configuration fault (wrong ctx, bad key, absent model),
      # never bad luck -- and without this the runner "completes" in under a
      # minute having produced nothing, which is what both the Gemini 401 and
      # the 32768-ctx bug looked like in the log.
      F=$(wc -l < "$LOG/.fails_$M" 2>/dev/null | tr -d " ")
      A=$(find "$REPO/data/animatebench_v6_cache" -name '*.svg' 2>/dev/null | wc -l)
      if [ "${F:-0}" -ge 4 ] && [ "${A:-0}" -eq 0 ]; then
        say "   ABORT $M: ${F} batches failed, 0 artifacts -- config fault"
        tail -4 "$LOG/pipe_${M}_${STYLE}.log" 2>/dev/null | tee -a "$LOG/run.log"
        touch "$LOG/$M.failed"; break 2
      fi
    done
  done
  touch "$LOG/$M.done"; say "== $M generation complete"
done
serve_stop
say "=== ALL GENERATION DONE"
