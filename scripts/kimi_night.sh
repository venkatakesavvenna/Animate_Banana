#!/usr/bin/env bash
# Wait for the Kimi weights on both nodes, serve, then generate all three sets.
#
# WHY LOCAL AND NOT OpenRouter: Kimi via API is ~$101 for the three sets
# (measured 6 calls/cell, 0.95/4.00 per M) against a ~$30 balance. Served
# locally it costs nothing, which is the only way all 500 cells fit -- and the
# DeepSeek runs need what balance there is.
#
# The generator itself is scripts/run_gen_night.sh with MODEL=kimi_k26, whose
# config points `served` at http://127.0.0.1:8011/v1 -- so this script must
# also open an SSH TUNNEL from this host to the head node's port 8011.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
HEAD=${HEAD:-10.20.188.73}
WORKER=${WORKER:-10.20.197.208}
PORT=${PORT:-8011}
NEED_GB=${NEED_GB:-590}
LOG=$REPO/logs/kimi_night; mkdir -p "$LOG"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/run.log"; }
SSH="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no"

# COMPLETENESS BY SHARD, NOT BY SIZE.
#
# The size gate this replaces compared a GiB measurement against a DECIMAL-GB
# constant: the repo is 595.2 GB = 554.3 GiB, `du -sb`/2^30 reports 555, and
# `555 -ge 590` is false forever. Both nodes finished at ~20:30 and the
# supervisor still said "waiting" seven hours later -- a COMPLETE download that
# reads exactly like a stalled one. Never gate a download on a size constant.
#
# Counting shards against the index is also the check that actually matters: a
# truncated shard hides inside size noise (kimi-serving-portability trap 4),
# while a missing weight_map entry cannot.
ready(){ $SSH "$1" 'S=$(ls -d /opt/dlami/nvme/vkkimi/hf/hub/models--moonshotai--Kimi-K2.6/snapshots/* 2>/dev/null|head -1)
  [ -n "$S" ] || exit 1
  [ "$(find /opt/dlami/nvme/vkkimi -name "*.incomplete" 2>/dev/null|wc -l)" = 0 ] || exit 1
  S="$S" python3 -c "
import json,glob,os,sys
S=os.environ[\"S\"]
idx=json.load(open(os.path.join(S,\"model.safetensors.index.json\")))
need=set(idx[\"weight_map\"].values())
have={os.path.basename(p) for p in glob.glob(S+\"/*.safetensors\")}
sys.exit(0 if not (need-have) else 1)
"' >/dev/null 2>&1; }

say "=== kimi night | head=$HEAD worker=$WORKER"

# 1. Both nodes must hold the full checkpoint. A node that is still downloading
#    will load a TRUNCATED shard and die deep inside weight loading, which reads
#    as a model bug rather than a missing file.
while :; do
  ready "$HEAD" && ready "$WORKER" && break
  say "weights not yet complete on both nodes -- waiting 5min"
  sleep 300
done
say "weights present on both nodes"

# 2. Serve, and wait for the API to answer. A 595GB two-node load is slow;
#    90 minutes is generous rather than optimistic.
HEAD=$HEAD WORKER=$WORKER PORT=$PORT bash scripts/remote/serve_kimi_nvme.sh 2>&1 | tee -a "$LOG/run.log"
for i in $(seq 1 180); do
  if $SSH "$HEAD" "curl -s -m 5 http://127.0.0.1:$PORT/v1/models" | grep -q Kimi; then
    say "server up after $((i*30))s"; break
  fi
  [ "$i" = 180 ] && { say "FATAL: server did not come up in 90min"; $SSH "$HEAD" "tail -40 /tmp/vllm_kimi_mn.log" | tee -a "$LOG/run.log"; exit 1; }
  sleep 30
done

# 3. Tunnel 8011 here, because the configs address 127.0.0.1.
pkill -f "ssh.*-L $PORT:127.0.0.1:$PORT" 2>/dev/null
ssh -f -N -o BatchMode=yes -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
    -L "$PORT:127.0.0.1:$PORT" "$HEAD"
sleep 3
curl -s -m 8 "http://127.0.0.1:$PORT/v1/models" | grep -q Kimi \
  || { say "FATAL: tunnel not serving"; exit 1; }
say "tunnel up on 127.0.0.1:$PORT"

# 4. Generate. Sequential across sets: they share ONE server, and three
#    concurrent runners at JOBS=3 would put 36 requests against --max-num-seqs 16.
for SET in abl zs v6; do
  say "--- generating kimi_k26/$SET"
  MODEL=kimi_k26 SET=$SET JOBS=2 bash scripts/run_gen_night.sh 2>&1 | tail -3 | tee -a "$LOG/run.log"
  say "--- $SET generation returned $?"
done
say "=== kimi generation done"
