#!/usr/bin/env bash
# Export v6 animations locally, continuously, while generation runs remotely.
#
# WHY SEPARATE: the exporter needs headless Playwright, which fails to launch in
# the vlm-ingest image on the serving nodes. It is CPU-only, so it runs in this
# box's own container instead. Generation stops at critique-animation; this
# picks up every animation that has no export yet.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
C=img-2-svg-pretraining-singlenode-venkat.kesav
PY=/environments/img_2_svg_pretraining/bin/python
LOG=$REPO/logs/v6_export; mkdir -p "$LOG"
PID=$LOG/run.pid
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then echo "already running"; exit 0; fi
echo $$ > "$PID"; trap 'rm -f "$PID"' EXIT
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/run.log"; }
say "=== v6 export daemon started"
IDLE=0
while :; do
  mapfile -t TODO < <(python3 - <<'PY'
import glob,os,json,re
cache='data/animatebench_v6_cache/animatebench_v6_ds'
STYLES=("progressive_reveal","hopping_bounding_box","sliding_bounding_box","colour_pop","alpha_masking")
MODELS={"Qwen-Qwen3.8-27B":"bench_v6_svg_qwen38_27b","zai-org-GLM-4.6V":"bench_v6_svg_glm46v",
        "google-gemma-4-31B-it":"bench_v6_svg_gemma4_31b","Qwen-Qwen3-VL-235B-A22B-Instruct":"bench_v6_svg_qwen3vl235b"}
smap=json.load(open('data/v6_style_map.json'))
out=[]
for pat in ('animation_final','animation'):
    for svg in glob.glob(f'{cache}/{pat}/*/*.svg'):
        lin=svg.split(os.sep)[-2]; name=os.path.basename(svg)
        # STRIP THE EXTENSION FIRST, THEN THE ROUND SUFFIX. Doing it the other
        # way ("SAMPLE.round0.svg".split('.round')[0][:-4]) yields "SAMP" -- four
        # characters eaten off a name that no longer had an extension. The
        # truncated id then misses the style map, style resolves empty, and the
        # export dies with "--style: expected one argument".
        samp=re.sub(r'\.svg$','',name)
        samp=re.sub(r'\.round\d+$','',samp)
        style=next((t for t in STYLES if f'__{t}__' in lin), None) or smap.get(samp)
        cfg=next((c for tag,c in MODELS.items() if tag in lin), None)
        if not style or not cfg: continue
        # DEDUP MUST MATCH THE MODEL, NOT JUST THE STYLE. Globbing
        # exports/*{style}*/{samp}/ matches ANY model's export of that sample,
        # so once qwen38 had exported a cell, every other model's copy of it was
        # judged "already done" and silently skipped -- glm46v sat at 2 exports
        # against 72 finished animations. The lineage dir identifies the model,
        # so dedup against that exact lineage.
        if os.path.exists(f'{cache}/exports/{lin}/{samp}/animation.mp4'): continue
        out.append(f"{cfg}|{style}|{samp}")
print('\n'.join(sorted(set(out))))
PY
)
  if [ "${#TODO[@]}" -eq 0 ]; then
    IDLE=$((IDLE+1)); [ $((IDLE % 12)) -eq 1 ] && say "nothing to export (idle)"
    sleep 300; continue
  fi
  IDLE=0
  say "exporting ${#TODO[@]} pending"
  for row in "${TODO[@]}"; do
    IFS='|' read -r cfg style samp <<<"$row"
    [ -z "$style" ] || [ -z "$samp" ] || [ -z "$cfg" ] && { say "  skip malformed row: $row"; continue; }
    timeout 600 docker exec -u "$(id -u):$(id -g)" -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers \
      "$C" bash -lc "cd /code && PYTHONPATH=src $PY -u -m img_2_svg_pretraining.pipeline.run_pipeline \
        export --config src/img_2_svg_pretraining/pipeline/configs/$cfg.yaml --style $style --only $samp" \
      >>"$LOG/export.log" 2>&1 || say "  export failed: $samp"
  done
  say "wave done"
done
