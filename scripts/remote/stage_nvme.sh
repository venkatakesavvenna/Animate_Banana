#!/usr/bin/env bash
# Push the Kimi checkpoint from THIS node's NVMe to a peer node's NVMe.
#
# WHY: 16 ranks all reading one 555GB checkpoint off shared Lustre took ~35-40s
# per shard (~60 min total). Node-local NVMe read the same shards at ~7s. With a
# copy on each node's own disk, every rank reads locally and the load drops to
# roughly 10 minutes.
#
# Goes NVMe -> NVMe over the network on purpose, never via Lustre: a Lustre-
# sourced copy would compete with the very reads it is meant to replace.
# aes128-gcm is chosen because a bulk transfer this size is bounded by SSH
# cipher throughput, not the link.
set -u
PEER=${PEER:?set PEER}
HUB=/opt/dlami/nvme/venkat.kesav/hf_cache/hub
K=models--moonshotai--Kimi-K2.6
# -n is REQUIRED on the probe ssh below: without it ssh inherits and drains the
# `ls` pipeline's stdin, the loop sees an empty list, and the copy reports
# success having transferred nothing (observed: "STAGE DONE 23M").
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -c aes128-gcm@openssh.com"
SSHN="$SSH -n"

$SSH "$PEER" "mkdir -p $HUB/$K/blobs $HUB/$K/snapshots $HUB/$K/refs"
tar -C "$HUB/$K" -cf - refs snapshots | $SSH "$PEER" "tar -C $HUB/$K -xf -"

# Blobs in parallel; skip any already present at full size so a re-run resumes.
cd "$HUB/$K/blobs" || exit 1
ls | while read -r b; do
  s=$(stat -c %s "$b")
  d=$($SSHN "$PEER" "stat -c %s $HUB/$K/blobs/$b 2>/dev/null||echo 0")
  [ "$s" = "$d" ] || echo "$b"
done | xargs -P 6 -I{} sh -c \
  "cat {} | $SSH $PEER 'cat > $HUB/$K/blobs/{}.part && mv $HUB/$K/blobs/{}.part $HUB/$K/blobs/{}'"

echo "STAGE DONE -> $PEER $($SSH "$PEER" "du -sh $HUB/$K 2>/dev/null|cut -f1")"
