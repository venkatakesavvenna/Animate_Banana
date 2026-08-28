#!/usr/bin/env bash
# Second pass over the animation tree, run after the main sweep finishes.
#
# The main driver walks the cell list once and skips anything already scored,
# so a cell whose record was deleted AFTER the driver walked past it is never
# revisited. That is exactly what happened to the three hopping_bounding_box
# cells scored before the schema gained its missing
# `mobile_boxes_static_elements` field: their records were removed so they
# would be re-judged, but the driver was already past them.
#
# Re-running the same driver is safe and cheap: it re-scores only cells with
# no record, and the backend response cache means unchanged prompts cost
# nothing to re-ask.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"

# Wait for the primary sweep to finish before starting, so the two never
# judge the same cell concurrently and race on its record.
while pgrep -f "scripts/animation_sweep.py" >/dev/null; do sleep 60; done

echo "[$(date +%H:%M:%S)] primary sweep done; starting follow-up pass"
PYTHONPATH=src python3 scripts/animation_sweep.py

# The sweeps import animation_quality once at start, so any record written
# after UNENFORCEABLE_RULES changed carries the stale annotation. Rewrite it
# from the live table now that nothing is still writing records.
echo "[$(date +%H:%M:%S)] correcting ascs_unenforced_rules metadata"
PYTHONPATH=src python3 scripts/fix_unenforced_rules.py
echo "[$(date +%H:%M:%S)] follow-up complete"
