#!/usr/bin/env bash
# Rebuild the score files from the eval records and push them to Drive, every
# SYNC_MIN minutes, while judging is still running.
#
# WHY INCREMENTAL: judging 549 cells takes many hours, and a single upload at
# the end means nothing is visible until then -- and any interruption loses the
# whole window. rclone copy only sends what changed, so a repeat pass is cheap.
#
# The collectors are re-run each pass rather than appending, so a cell that was
# re-judged (e.g. after an error stub was purged) overwrites its old score
# instead of leaving a stale one behind.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
SYNC_MIN=${SYNC_MIN:-20}
LOG=$REPO/logs/drive; mkdir -p "$LOG"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG/sync.log"; }

push(){  # local_dir  remote
  ~/bin/rclone copy "$1" "$2" --transfers 8 --checkers 8 \
     --drive-chunk-size 32M --stats-one-line --log-level ERROR 2>>"$LOG/sync.log"
}

while :; do
  # --- ablations (now 728/728) -------------------------------------------
  python3 scripts/collect_ablation_scores.py >/dev/null 2>&1 || true
  n_abl=$(find data/ablation_scores -name '*.json' 2>/dev/null | wc -l)
  [ "$n_abl" -gt 0 ] && push data/ablation_scores "gdrive:AnimateBanana_Ablations_20260904/Scores"

  # --- DeepSeek V4 --------------------------------------------------------
  for s in abl zs v6; do
    python3 scripts/collect_gen_scores.py --model dsv4fv --set "$s" >/dev/null 2>&1 || true
  done
  n_ds=$(find data/gen_scores -name '*.json' 2>/dev/null | wc -l)
  [ "$n_ds" -gt 0 ] && push data/gen_scores "gdrive:AnimateBanana_DeepSeekV4_20260908/Scores"

  say "synced: ablation=$n_abl deepseek=$n_ds json files"
  sleep $((SYNC_MIN*60))
done
