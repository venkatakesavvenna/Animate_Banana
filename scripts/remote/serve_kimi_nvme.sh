#!/usr/bin/env bash
# Serve Kimi-K2.6 across TWO nodes from NVMe, using vllm/vllm-openai:nightly.
#
#   HEAD=10.20.188.73 WORKER=10.20.197.208 bash scripts/remote/serve_kimi_nvme.sh
#
# TWO NODES ARE MANDATORY. The checkpoint is 595GB; one node at
# --gpu-memory-utilization 0.90 offers 8 x 80 x 0.90 = 576GB, which is LESS than
# the weights. TP=8 x PP=2 across two nodes gives 1152GB.
#
# WHY THIS IMAGE AND NOT THE v5serve VENV: the venv's python3.12 symlinks to
# /usr/bin/python3.12, absent from most images here, and fails as "No such file
# or directory" on the venv's own python. The vllm image is self-contained.
#
# WEIGHTS FROM NVMe, PER NODE. Lustre is 100% full, and both nodes must see the
# checkpoint, so each downloads its own copy to /opt/dlami/nvme/vkkimi.
#
# RESTART THE CONTAINER RATHER THAN KILLING WORKERS: processes inside a
# pre-existing container may run as the image's user, and a host-side kill -9
# silently no-ops while 595GB stays pinned. Here we own the container, so
# `docker rm -f` is both sufficient and immediate.
set -u
HEAD=${HEAD:?set HEAD ip}
WORKER=${WORKER:?set WORKER ip}
PORT=${PORT:-8011}
NAME=${NAME:-kimi-mn}
SNAP_GLOB='/opt/dlami/nvme/vkkimi/hf/hub/models--moonshotai--Kimi-K2.6/snapshots/*'
SSH="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no"

run(){ $SSH "$1" "$2"; }

# 1. Clean slate on both nodes.
for N in "$HEAD" "$WORKER"; do
  run "$N" "docker rm -f $NAME >/dev/null 2>&1; true"
done

# 2. Start a long-lived container on each node, sharing the host network so Ray
#    can address peers directly.
for N in "$HEAD" "$WORKER"; do
  # THESE FLAGS CAN ONLY BE SET AT CREATION -- an existing container cannot be
  # adapted, which is why the container is always recreated rather than reused.
  #   --privileged + /dev/infiniband + memlock  : EFA needs all three
  #   --shm-size=512g                           : NCCL buffers; 32g wedges TP=8
  #   NCCL_NVLS_ENABLE=0                        : NVLink SHARP needs Fabric
  #     Manager, which is absent inside Docker here
  #   NCCL_SOCKET_IFNAME=^docker,lo,veth        : else NCCL binds the docker
  #     bridge and the two nodes never find each other
  run "$N" "docker run -d --name $NAME --gpus all --ipc=host --network host \
     --privileged --shm-size=512g --ulimit memlock=-1 \
     -v /dev/infiniband:/dev/infiniband \
     -v /opt/dlami/nvme/vkkimi:/opt/dlami/nvme/vkkimi \
     -e HF_HOME=/opt/dlami/nvme/vkkimi/hf \
     -e HF_MODULES_CACHE=/tmp/hf_modules \
     -e VLLM_HOST_IP=$N \
     -e FI_PROVIDER=efa \
     -e NCCL_SOCKET_IFNAME=^docker,lo,veth \
     -e NCCL_NVLS_ENABLE=0 \
     --entrypoint sleep vllm/vllm-openai:nightly infinity" >/dev/null
done
sleep 5

# 2b. RAY IS NOT IN THIS IMAGE. `vllm/vllm-openai:nightly` ships no `ray` module
#     at all -- `ray start` fails as "command not found" and, without this, the
#     GPU pre-check below reports 0 and nothing launches. Multi-node vLLM has no
#     Ray-free path, so it is installed into each fresh container.
for N in "$HEAD" "$WORKER"; do
  run "$N" "docker exec $NAME bash -lc 'pip install -q ray[default]'" >/dev/null 2>&1 &
done
wait
for N in "$HEAD" "$WORKER"; do
  run "$N" "docker exec $NAME python3 -c 'import ray' " >/dev/null 2>&1 \
    || { echo "FATAL: ray missing on $N after install"; exit 1; }
done

# 3. Ray cluster: head first, then the worker joins.
run "$HEAD"   "docker exec $NAME bash -lc 'python3 -m ray.scripts.scripts start --head --node-ip-address=$HEAD --port=6379 --num-gpus=8 --disable-usage-stats'" >/dev/null 2>&1
sleep 8
run "$WORKER" "docker exec $NAME bash -lc 'python3 -m ray.scripts.scripts start --address=$HEAD:6379 --num-gpus=8 --disable-usage-stats'" >/dev/null 2>&1
sleep 5

# 3b. PROVE THE CLUSTER BEFORE LOADING. A 595GB load is ~10 minutes; finding out
#     afterwards that the second node never joined wastes it every attempt.
gpus=$(run "$HEAD" "docker exec $NAME bash -lc 'python3 -c \"import ray;ray.init(address=\\\"auto\\\");print(int(ray.cluster_resources().get(\\\"GPU\\\",0)))\"' 2>/dev/null" | tail -1)
echo "ray cluster GPUs: ${gpus:-0}"
if [ "${gpus:-0}" != "16" ]; then
  echo "FATAL: expected 16 GPUs across both nodes, got ${gpus:-0} -- not launching"
  exit 1
fi

# 4. Launch the API server on the head.
run "$HEAD" "docker exec -d $NAME bash -lc '
  SNAP=\$(ls -d $SNAP_GLOB | head -1)
  python3 -u -m vllm.entrypoints.openai.api_server \
    --model \"\$SNAP\" --served-model-name moonshotai/Kimi-K2.6 \
    --trust-remote-code \
    --tensor-parallel-size 8 --pipeline-parallel-size 2 \
    --distributed-executor-backend ray --disable-custom-all-reduce \
    --gpu-memory-utilization 0.90 \
    --max-model-len 131072 --limit-mm-per-prompt.image 2 --max-num-seqs 16 \
    --port $PORT --host 0.0.0.0 > /tmp/vllm_kimi_mn.log 2>&1'"
echo "launched Kimi on $HEAD (+$WORKER) port $PORT"
