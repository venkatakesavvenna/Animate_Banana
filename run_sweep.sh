#!/usr/bin/env bash
# Unattended full-pipeline sweep across local HF models.
#
#   ./run_sweep.sh
#   tail -f /tmp/sweep/LOG                                   # watch progress
#   cat /tmp/sweep/SUMMARY.txt                               # read when done
#
# Each model is pinned to its own GPU and runs concurrently, so the sweep costs
# roughly the slowest model rather than the sum. Every stage except raster
# integration (1b) runs: convert-code -> strategize -> parse -> sequence ->
# critique-sequence -> design -> critique-animation -> export.
#
# /tmp/sweep is a bind-mounted path visible to both host and container, so logs
# written here are readable from either side.
set -u

CONTAINER=img-2-svg-pretraining-singlenode-venkat.kesav
OUT=/tmp/sweep
DATASET=/code/data/test_benchmark
DATASET_NAME=test_benchmark
LIMIT=${LIMIT:-11}
BATCH=${BATCH:-2}
# Per-model ceiling. Generation is ~50 tok/s with no continuous batching and
# eight stages each emit thousands of tokens, so this is generous on purpose --
# a model that trips it was never going to finish.
TIMEOUT=${TIMEOUT:-10800}

MAIN=/environments/img_2_svg_pretraining/bin/python
GEMMA=/environments/gemma4/bin/python

# name|repo|attn|venv|gpu
#
# Attention differs by venv, not by taste:
#   main venv    NGC torch 2.7.0, flash_attn 2.7.3 works
#   gemma4 venv  torch 2.11.0, flash_attn ABI-broken, but has the transformers
#                the gemma4 architecture needs -> eager
# `sdpa` is used nowhere: it dispatches into cuDNN, which raises "No valid
# execution plans built" in this container, for both Gemma and Qwen.
#
# Ordered by confidence. gemma4-12b and qwen3-vl-8b are verified generating;
# qwen2.5-vl-7b shares an architecture family with a verified model; the 27B
# and 31B have never been run end-to-end here, so they are last -- with a GPU
# each they cannot delay the others.
MODELS=(
  "gemma4-12b|google/gemma-4-12B-it|eager|$GEMMA|1"
  "qwen3-vl-8b|Qwen/Qwen3-VL-8B-Instruct|flash_attention_2|$MAIN|2"
  "qwen2.5-vl-7b|Qwen/Qwen2.5-VL-7B-Instruct|flash_attention_2|$MAIN|3"
  "qwen3.6-27b|Qwen/Qwen3.6-27B|flash_attention_2|$MAIN|4"
  "gemma4-31b|google/gemma-4-31B-it|eager|$GEMMA|5"
)

STAGES=(convert-code strategize parse sequence critique-sequence
        design critique-animation export)

dexec() { docker exec "$CONTAINER" bash -c "$1"; }

mkdir -p "$OUT"
dexec "mkdir -p $OUT" || { echo "container unreachable"; exit 1; }
: > "$OUT/LOG"
: > "$OUT/STAGES.txt"

log() { echo "[$(date +%H:%M:%S)] $*" >> "$OUT/LOG"; }

log "sweep starting: ${#MODELS[@]} models x $LIMIT samples, batch=$BATCH"
dexec "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader" >> "$OUT/LOG" 2>&1
log "---"

run_model() {
  local spec="$1"
  IFS='|' read -r NAME REPO ATTN VENV GPU <<< "$spec"
  local dir="$OUT/$NAME"
  mkdir -p "$dir"

  # Config generation is a real script, not a nested heredoc -- quoting inside
  # `docker exec bash -c "..."` swallowed the heredoc and wrote nothing.
  if ! dexec "cd /code && $MAIN sweep_config.py '$NAME' '$REPO' '$ATTN' $BATCH $LIMIT '$DATASET' '$dir'" >> "$dir/setup.log" 2>&1; then
    log "$NAME: CONFIG FAILED (see $dir/setup.log)"
    return
  fi

  local started=$(date +%s)
  log "$NAME starting on gpu $GPU ($REPO, attn=$ATTN)"

  for stage in "${STAGES[@]}"; do
    local s0=$(date +%s)
    dexec "cd /code && timeout $TIMEOUT $VENV -m img_2_svg_pretraining.pipeline.run_pipeline \
             $stage --config $dir/config.yaml --gpu $GPU --force \
             >> $dir/run.log 2>&1"
    local rc=$?
    local secs=$(( $(date +%s) - s0 ))
    echo "$NAME $stage rc=$rc ${secs}s" >> "$OUT/STAGES.txt"
    log "  $NAME/$stage -> rc=$rc (${secs}s)"
    # convert-code feeds everything downstream; without it the rest is noise.
    if [ "$stage" = "convert-code" ] && [ $rc -ne 0 ]; then
      log "  $NAME: convert-code failed, skipping remaining stages"
      break
    fi
  done

  log "$NAME finished in $(( $(date +%s) - started ))s"
}

for spec in "${MODELS[@]}"; do
  run_model "$spec" &
  sleep 20   # stagger so five weight loads don't hammer the filesystem at once
done
wait

log "--- all models done, scoring ---"
dexec "cd /code && $MAIN sweep_score.py $OUT $DATASET_NAME $LIMIT" > "$OUT/SUMMARY.txt" 2>&1

{
  echo
  echo "================= SUMMARY ================="
  cat "$OUT/SUMMARY.txt"
  echo
  echo "per-stage timings: cat $OUT/STAGES.txt"
  echo "a model's output:  cat $OUT/<model>/run.log"
} >> "$OUT/LOG"

cat "$OUT/SUMMARY.txt"
log "sweep complete"
