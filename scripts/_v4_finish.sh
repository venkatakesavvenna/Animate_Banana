#!/usr/bin/env bash
# Finish the sweep unattended: stragglers -> judge them -> final table.
#
# Waits for the running judge to release the GPUs, then serves the three models
# that need the CUDA-13 container, judges everything that is still unscored, and
# regenerates the aggregate. Each step is individually resumable, so re-running
# this after any interruption picks up where it stopped.
cd /fsxvision_new/venkat.kesav/img_2_svg_pretraining
say() { echo "[$(date +%H:%M:%S)] [finish] $*" | tee -a logs/bench_v4/driver.log; }

while pgrep -f "run_bench_v4_judge.sh" >/dev/null 2>&1; do sleep 60; done
say "judge idle; generating the three stragglers"
rm -f logs/bench_v4/state/gemma4_31b.failed logs/bench_v4/state/glm46v.failed \
      logs/bench_v4/state/internvl35_241b.failed
MODELS="gemma4_31b glm46v internvl35_241b" bash scripts/run_bench_v4_oss.sh

say "generation done; judging every model"
bash scripts/run_bench_v4_judge.sh

say "building the final aggregate"
docker exec -u "$(id -u):$(id -g)" animatebanana-v4 bash -lc \
 'cd /code && PYTHONPATH=src /environments/img_2_svg_pretraining/bin/python -u \
    -m img_2_svg_pretraining.animatebench.aggregate \
    --evals-root data/animatebench_v4_cache/animatebench_v4/evals \
    --dataset data/animatebench_v4 --target svg --scope both --format md csv json' \
  > logs/bench_v4/final_aggregate.txt 2>&1
say "COMPLETE -- table at data/animatebench_v4_cache/animatebench_v4/evals/aggregate.md"
