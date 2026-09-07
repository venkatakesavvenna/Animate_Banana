#!/usr/bin/env bash
# Serve Kimi K2.6 across 2 nodes reading weights from SHARED LUSTRE.
#
# WHY THIS EXISTS ALONGSIDE serve_kimi_mn.sh / serve_kimi_stable.sh
# -----------------------------------------------------------------
# Those two both export HF_HOME=/opt/dlami/nvme/venkat.kesav/hf_cache, i.e.
# NODE-LOCAL NVMe, because the original pair of nodes had the checkpoint staged
# there by stage_nvme.sh. That staging is per node and does NOT follow you: on a
# freshly chosen node the NVMe path is empty.
#
# With HF_HUB_OFFLINE=1 an empty cache does not raise. vLLM resolves the model
# id, finds nothing, and the run WEDGES with a 0-byte log while Ray holds a
# 16-GPU placement group -- indistinguishable from a slow load. Measured
# 2026-09-02 on 10.20.213.4: eight minutes of flat GPU memory and an empty log
# before the cause was found.
#
# This profile simply does not override HF_*: init_mn.sh already points the
# container at /fsxvision_new/venkat.kesav/hf_cache_shared, which every
# Lustre-mounted node can read. Slower to load (~40-60 min vs ~10 off NVMe,
# per the note in serve_kimi_mn.sh) but correct anywhere Lustre is mounted.
#
# NVMe STAGING PATH IS NOT SAFE TO ASSUME EITHER. On 10.20.214.12 the path
# /opt/dlami/nvme/venkat.kesav is owned by *aryanjain.intern*, mode 755 --
# another user had created a directory named after us. `mkdir -p` succeeds on
# the existing parents and only the leaf create fails, so this surfaces late as
# a bare "Permission denied" from cp. Stage under a path you certainly own
# (HUB=/opt/dlami/nvme/vkkimi/hf_cache) and verify with `ls -ld`, per node.
#
# PLACEMENT GROUPS OUTLIVE THE API SERVER. `pkill -f api_server` leaves
# VLLM::EngineCore alive still holding all 16 GPUs, so the next launch waits
# forever for a placement group it can never get. Always tear the engine down
# and restart Ray -- which is what this script does before launching.
set -u
C=${C:-animatebanana-mn}
HEAD=${HEAD:-10.20.213.4}
WORKER=${WORKER:-10.20.214.12}
PY=/environments/v5serve/bin
# HUB unset -> weights come from the container's default (shared Lustre).
# HUB set   -> node-local NVMe hub, e.g. /opt/dlami/nvme/vkkimi/hf_cache.
HUB=${HUB:-}
if [ -n "$HUB" ]; then
  # HF_MODULES_CACHE: --trust-remote-code writes Kimi's modelling code into
  # $HF_HOME/modules. The container user (raghuveer.r) is not the host owner of
  # the staged dir, so that write is EPERM and the server dies at import time
  # with a bare PermissionError. Point it at /tmp, which is writable for all.
  HFENV="-e HF_HOME=$HUB -e HF_HUB_CACHE=$HUB/hub -e HUGGINGFACE_HUB_CACHE=$HUB/hub -e HF_HUB_OFFLINE=1 -e HF_MODULES_CACHE=/tmp/hf_modules"
else
  HFENV=""
fi
SSH="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no"

# TEARDOWN MUST DRAIN THE GPUs BEFORE RAY RESTARTS, not merely signal the
# processes. Ray workers hold device memory for a while after SIGKILL, and
# starting a new Ray session while the old one's workers are still resident
# produces, minutes into the load,
#     ActorHandleNotFoundError: ActorHandle objects are not valid across Ray sessions
# -- seen twice on 2026-09-02/03, both times after a restart launched while the
# nodes still showed 100-500GB resident. Waiting for the drain is the fix; the
# error appears long after launch, so a restart that "looked fine" is not.
for N in "$HEAD" "$WORKER"; do
  $SSH "$N" "docker exec -u 0 $C bash -lc '
      pkill -9 -f vllm.entrypoints; pkill -9 -f EngineCore; pkill -9 -f VLLM::
      pkill -9 -f RayWorkerProc; pkill -9 -f raylet; pkill -9 -f gcs_server
      $PY/ray stop --force >/dev/null 2>&1; true'" >/dev/null 2>&1
done
for i in $(seq 1 40); do
  sleep 15; drained=1
  for N in "$HEAD" "$WORKER"; do
    m=$($SSH "$N" 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|paste -sd+|bc' 2>/dev/null|tail -1)
    [ "${m:-999999}" -lt 2000 ] 2>/dev/null || drained=0
  done
  [ "$drained" = 1 ] && { echo "gpus drained after $((i*15))s"; break; }
  [ "$i" = 40 ] && echo "WARNING: gpus did not drain in 600s; launching anyway"
done
$SSH "$HEAD"   "docker exec $C bash -lc '$PY/ray start --head --node-ip-address=$HEAD --port=6379 --num-gpus=8 --disable-usage-stats'" >/dev/null 2>&1
sleep 8
$SSH "$WORKER" "docker exec $C bash -lc '$PY/ray start --address=$HEAD:6379 --num-gpus=8 --disable-usage-stats'" >/dev/null 2>&1
sleep 5

$SSH "$HEAD" "docker exec -d $HFENV $C bash -lc '
  $PY/python -u -m vllm.entrypoints.openai.api_server \
    --model moonshotai/Kimi-K2.6 --served-model-name moonshotai/Kimi-K2.6 \
    --trust-remote-code \
    --tensor-parallel-size 8 --pipeline-parallel-size 2 \
    --distributed-executor-backend ray --disable-custom-all-reduce \
    --gpu-memory-utilization 0.90 \
    --max-model-len 131072 --limit-mm-per-prompt.image 2 --max-num-seqs 16 \
    --port 8011 --host 0.0.0.0 > /tmp/vllm_kimi_mn.log 2>&1'"
echo "launched from lustre on $HEAD (+$WORKER)"
