#!/usr/bin/env bash
# Progress + health line for the Kimi generation run, every 10 minutes.
#
# Watches the two things that distinguish a live run from a dead one:
#   * generation_tokens_total  -- flat means every cell is being SKIPPED as
#     already-recorded, which looks exactly like success in the runner log.
#   * finished_reason=length   -- silent truncation at max_tokens; artifacts can
#     still land, so this is the only place it shows up.
set -u
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
HEAD=${HEAD:-10.20.188.73}
LOG=$REPO/logs/kimi_night/watch.log
SSH="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no"
prev=0
while :; do
  m=$($SSH "$HEAD" 'docker exec kimi-mn bash -lc "curl -s -m 5 http://127.0.0.1:8011/metrics"' 2>/dev/null)
  # MATCH `_total` EXPLICITLY. Prometheus also exposes `..._created` for each
  # counter, whose value is a unix timestamp; a pattern matching only
  # `finished_reason="stop"` hits BOTH lines and awk concatenates them, printing
  # e.g. 1181788753579 for a true count of 118 -- a health readout that looks
  # like a huge number rather than an obvious parse error.
  val(){ echo "$m" | grep -E "^vllm:$1" | grep -v "_created" | head -1 | awk '{printf "%.0f",$NF}'; }
  tok=$(val 'generation_tokens_total')
  run=$(val 'num_requests_running')
  stop=$(echo "$m" | grep '^vllm:request_success_total' | grep 'finished_reason="stop"'   | head -1 | awk '{printf "%.0f",$NF}')
  len=$(echo  "$m" | grep '^vllm:request_success_total' | grep 'finished_reason="length"' | head -1 | awk '{printf "%.0f",$NF}')
  err=$(echo  "$m" | grep '^vllm:request_success_total' | grep 'finished_reason="error"'  | head -1 | awk '{printf "%.0f",$NF}')
  line=""
  for s in abl zs v6; do
    e=$(find data/gen_cache/kimi_k26_$s -path '*/exports/*' -name animation.mp4 2>/dev/null|wc -l)
    a=$(find data/gen_cache/kimi_k26_$s -path '*/animation/*' -name '*.svg' 2>/dev/null|wc -l)
    line="$line $s:a=$a/e=$e"
  done
  d=$(( ${tok:-0} - prev )); prev=${tok:-0}
  flag=""; [ "$d" -eq 0 ] && flag="  <-- TOKENS FLAT, run may be idle/finished"
  echo "[$(date '+%m-%d %H:%M')] run=$run tok=${tok:-0} (+$d) stop=${stop:-0} len=${len:-0} err=${err:-0} |$line$flag" >> "$LOG"
  sleep 600
done
