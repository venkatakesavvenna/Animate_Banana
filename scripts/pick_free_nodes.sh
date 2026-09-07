#!/usr/bin/env bash
# Print IPs of N nodes that can actually host a vLLM shard right now.
#   NEED=2 bash scripts/pick_free_nodes.sh
#
# A node qualifies only if ALL THREE hold:
#   * GPUs genuinely free      -- `sinfo` says "idle" for nodes whose 8 H100s are
#     fully pinned, because almost everyone here runs in bare docker outside
#     Slurm. Only nvidia-smi tells the truth.
#   * /fsxvision_new mounted   -- several free-GPU nodes have no Lustre, so they
#     cannot see the weights, the venv or the repo.
#   * docker usable AND a CUDA-capable image present. /environments (the venv)
#     is bind-mounted from Lustre, so the image only has to supply CUDA + bash --
#     any vlm-ingest-*/visualgrounding image does. A node whose only images are
#     small app images (e.g. better-scraper, 896MB) cannot serve.
#
# NODE OCCUPANCY IS NOT STABLE. On 2026-09-03 two nodes that had served Kimi
# hours earlier were taken over by other users (sreevatsa.s, aryanjain.intern)
# while it was down. vLLM asks for 90% of each GPU, gets refused, and the
# workers die -- surfacing minutes later as
#     ActorHandleNotFoundError: ActorHandle objects are not valid across Ray sessions
# which reads like a Ray bug and is actually "someone else is on your node".
# ALWAYS re-pick before serving; never reuse a node list across a restart.
set -u
NEED=${NEED:-2}
MAXMEM=${MAXMEM:-5000}          # MiB already in use that still counts as free
NODES=${NODES:-"135-108 144-213 155-194 173-10 176-23 179-184 181-135 183-3 188-73 193-129 196-218 197-208 211-172 213-4 214-12 224-174 225-177 227-19 233-75 238-24 239-162 239-233"}
TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT
for n in $NODES; do
  ip=$(echo "$n" | sed 's/^/10.20./; s/-/./')
  (
    r=$(timeout 30 ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ip" \
      'm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|paste -sd+|bc)
       l=$([ -d /fsxvision_new/venkat.kesav ] && echo Y || echo N)
       d=$(docker ps >/dev/null 2>&1 && echo Y || echo N)
       img=$(docker images --format "{{.Repository}}:{{.Tag}}" 2>/dev/null \
             | grep -E "vlm-ingest|visualgrounding" | head -1)
       echo "${m:-999999} $l $d ${img:-NONE}"' 2>/dev/null | tail -1)
    set -- $r
    [ "${1:-999999}" -le "$MAXMEM" ] 2>/dev/null && [ "${2:-N}" = Y ] \
      && [ "${3:-N}" = Y ] && [ "${4:-NONE}" != NONE ] && echo "$ip ${4}"
  ) >> "$TMP" &
done
wait
head -n "$NEED" "$TMP"
