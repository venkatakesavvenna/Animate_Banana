#!/bin/bash
# Parallel copy of the Kimi HF cache from node-local NVMe to shared Lustre.
# Parallel because a single cp stream does not saturate Lustre; 8 writers do.
# Blobs are the real bytes; snapshots are symlinks into blobs, so blobs must
# land first or the snapshot links dangle.
set -u
SRC=/opt/dlami/nvme/venkat.kesav/hf_cache/hub/models--moonshotai--Kimi-K2.6
DST=/fsxvision_new/venkat.kesav/hf_cache_shared/hub/models--moonshotai--Kimi-K2.6
mkdir -p "$DST/blobs" "$DST/snapshots" "$DST/refs"
cp -a "$SRC/refs/." "$DST/refs/" 2>/dev/null
# Skip blobs already copied at full size: makes a re-run a resume.
ls "$SRC/blobs" | while read -r b; do
  s=$(stat -c %s "$SRC/blobs/$b" 2>/dev/null)
  d=$(stat -c %s "$DST/blobs/$b" 2>/dev/null || echo 0)
  [ "$s" = "$d" ] || echo "$b"
done | xargs -P 8 -I{} cp -a "$SRC/blobs/{}" "$DST/blobs/{}"
# Snapshots last: they are symlinks and are meaningless without the blobs.
for rev in "$SRC"/snapshots/*/; do
  r=$(basename "$rev"); mkdir -p "$DST/snapshots/$r"
  cp -a "$rev." "$DST/snapshots/$r/" 2>/dev/null
done
echo "COPY DONE $(du -sh $DST 2>/dev/null|cut -f1)"
