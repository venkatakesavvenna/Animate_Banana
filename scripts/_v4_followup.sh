#!/usr/bin/env bash
# Wait for the running sweep to finish, then run it again.
#
# The driver visits each model once per invocation, so a model that failed
# early (gemma4_31b, which needed a different serving venv) is not revisited
# within that same run. A second invocation skips whatever is marked .done and
# retries the rest -- and because every stage skips work whose artifact already
# exists, the retry costs only the missing cells.
cd /fsxvision_new/venkat.kesav/img_2_svg_pretraining
while pgrep -f "run_bench_v4_oss.sh" >/dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] first pass finished; starting follow-up" >> logs/bench_v4/driver.log
bash scripts/run_bench_v4_oss.sh
