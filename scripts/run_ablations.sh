#!/usr/bin/env bash
# Run the AnimateBanana ablation suite with response-cache stage reuse.
#
# Usage:
#   run_ablations.sh --dry-run            # show what would run, spend nothing
#   run_ablations.sh                      # everything: full first, then the 7
#   run_ablations.sh stage2_no_image ...  # just the named config(s)
#
# ORDER MATTERS AND THE SCRIPT ENFORCES IT. `full` (the ANIMATEBANANA baseline)
# must complete before any ablation starts, because every ablation's savings
# come from hardlink-seeding its response cache from full's: a call whose input
# is identical to the baseline's hits the cache and costs $0; only the ablated
# stage and its downstream pay. Seeding from a half-finished baseline would
# make the late samples pay full price -- silently.
#
# `full` itself is seeded from the round-one v5 cache, whose prompts are the
# ones in the tree today (every prompt edit predates the 29 Aug run), so the
# expensive stage-1 converter calls are cache hits there too.
#
# WHY NOT --force ANYWHERE: force recomputes artifacts that exist, re-spending
# for identical results. A cell is complete when its export exists; everything
# else follows from stage-output existence plus the response cache.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
C=img-2-svg-pretraining-singlenode-venkat.kesav
PY=/environments/img_2_svg_pretraining/bin/python
KEY=$(grep -E "^OPEN_ROUTER_KEY=" .env | cut -d= -f2- | tr -d '\r\n ')
JOBS="${JOBS:-6}"
# ONLY=<sample_id> restricts the whole suite to one sample -- a cheap end-to-end
# demo of every ablation before committing to 91. The baseline guard then only
# requires THAT cell to exist in full's cache, not all 91.
ONLY="${ONLY:-}"
CELLS="$REPO/data/ablation_cells.txt"
R1_CACHE="$REPO/data/animatebench_v5_or_cache"
ABL_CACHE="$REPO/data/ablation_cache"
DRY=0

ORDER=(full stage1_no_critic stage2_no_image sequencer_no_xml \
       stage2_no_critic narration_no_context designer_no_image stage3_no_critic)

args=()
for a in "$@"; do
  [ "$a" = "--dry-run" ] && DRY=1 || args+=("$a")
done
[ ${#args[@]} -gt 0 ] && ORDER=("${args[@]}")

mkdir -p logs/ablation

done_cell() {  # done_cell <name> <style> <sample>
  ls "$ABL_CACHE/$1"/animatebench_v5/exports/*__"$2"__*/"$3"/animation.mp4 >/dev/null 2>&1
}

run_cell() {   # run_cell <name> <style> <sample>
  local name="$1" style="$2" sample="$3"
  docker exec -u "$(id -u):$(id -g)" -e OPEN_ROUTER_KEY="$KEY" \
    -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers "$C" bash -lc \
    "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline all \
       --config src/img_2_svg_pretraining/pipeline/configs/gemini_ablation_${name}.yaml \
       --style $style --only $sample" \
    > "logs/ablation/${name}__${style}__${sample}.log" 2>&1
  local rc=$?
  if done_cell "$name" "$style" "$sample"; then echo "  ok   $name $style/$sample"
  else echo "  FAIL $name $style/$sample (exit=$rc, log logs/ablation/${name}__${style}__${sample}.log)"; fi
}

for name in "${ORDER[@]}"; do
  cfg="src/img_2_svg_pretraining/pipeline/configs/gemini_ablation_${name}.yaml"
  [ -f "$cfg" ] || { echo "ABORT: no config for '$name'" >&2; exit 2; }

  # --- seed the response cache BEFORE any cell of this config runs ---
  if [ "$name" = "full" ]; then
    src="$R1_CACHE"
  else
    # Refuse to start an ablation before the baseline is complete: its cache
    # would be missing the very entries the ablation is meant to reuse.
    if [ -n "$ONLY" ]; then
      total=1
      have=$(ls "$ABL_CACHE"/full/animatebench_v5/exports/*/"$ONLY"/animation.mp4 2>/dev/null | wc -l)
    else
      # BASELINE_MIN lets a repair pass that hit a genuine model wall (the
      # animator critic refusing SVG on a couple of cells) proceed on the
      # cells that DO exist, rather than blocking the whole suite forever.
      total="${BASELINE_MIN:-$(wc -l < "$CELLS")}"
      have=$(ls "$ABL_CACHE"/full/animatebench_v5/exports/*/*/animation.mp4 2>/dev/null | wc -l)
    fi
    if [ "$have" -lt "$total" ]; then
      if [ "$DRY" -eq 1 ]; then
        echo "DRY: $name  BLOCKED until baseline completes ($have/$total exports)"
        continue
      fi
      echo "ABORT: baseline 'full' has $have/$total exports; ablation '$name' would seed from an incomplete cache" >&2
      exit 3
    fi
    src="$ABL_CACHE/full"
  fi
  if [ "$DRY" -eq 1 ]; then
    todo=0
    while IFS=: read -r style sample; do
      [ -n "$ONLY" ] && [ "$sample" != "$ONLY" ] && continue
      done_cell "$name" "$style" "$sample" || todo=$((todo+1))
    done < "$CELLS"
    echo "DRY: $name  seed<-$(basename "$src")  cells-to-run=$todo/$(wc -l < "$CELLS")"
    continue
  fi

  echo "=== $name: seeding responses from $(basename "$src")"
  python3 scripts/seed_ablation_responses.py "$src" "$ABL_CACHE/$name" || exit 4
  # Stage-1 artifacts too -- the integrator embeds ABSOLUTE cache paths in the
  # SVG, so without byte-identical code every downstream request hashes
  # differently and misses the response cache (measured: ablations diverging at
  # narration still paid from parse onward). stage1_no_critic must not receive
  # code_reviewed: resolve_code would serve the critic's output inside the very
  # ablation that removes the critic. `full` seeds from round one, which never
  # ran this baseline's critic, so code_reviewed is excluded there as well --
  # its stale 28-Aug leftovers would otherwise pre-empt the critic stage.
  skipfam=""
  case "$name" in full|stage1_no_critic) skipfam="--skip code_reviewed" ;; esac
  echo "=== $name: seeding stage-1 artifacts from $(basename "$src")"
  python3 scripts/seed_ablation_artifacts.py "$src" "$ABL_CACHE/$name" $skipfam || exit 4

  echo "=== $name: running cells (JOBS=$JOBS)"
  while IFS=: read -r style sample; do
    [ -z "$sample" ] && continue
    [ -n "$ONLY" ] && [ "$sample" != "$ONLY" ] && continue
    done_cell "$name" "$style" "$sample" && { echo "  have $name $style/$sample"; continue; }
    run_cell "$name" "$style" "$sample" &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  done < "$CELLS"
  wait
  n=$(ls "$ABL_CACHE/$name"/animatebench_v5/exports/*/*/animation.mp4 2>/dev/null | wc -l)
  echo "=== $name complete: $n/$(wc -l < "$CELLS") exports"
done
echo "=== ablation suite done"
