#!/usr/bin/env bash
# 1) wait for the in-flight upload of the corrected Gemini set to land
# 2) wait for the remaining cells to generate
# 3) re-stage and COPY (not purge) the stragglers into the same folder
# Names are stable (<sample>.mp4), so a plain copy fills gaps without a second
# destructive purge of a folder that is already correct.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"; LOG=$REPO/logs/drive
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/finish_gemini.log"; }
while pgrep -f "rclone copy data/drive_stage_zs_gemini" >/dev/null 2>&1; do sleep 30; done
n=$(~/bin/rclone ls "gdrive:Zero_Shot_193/AnimateBanana" 2>/dev/null | wc -l)
say "first upload landed: $n files in Drive"
while [ -f logs/zs_gemini/run.pid ] && kill -0 "$(cat logs/zs_gemini/run.pid 2>/dev/null)" 2>/dev/null; do sleep 120; done
say "generation finished; re-staging"
python3 - <<'PY' | tee -a "$LOG/finish_gemini.log"
import glob,os,shutil,json,collections
from pathlib import Path
base=Path('data/animatebench_zs_or_cache/animatebench_zs'); DST=Path('data/drive_stage_zs_gemini')
SD={"progressive_reveal":"Progressive_Reveal","alpha_masking":"Alpha_Masking","colour_pop":"Colour_Pop",
    "hopping_bounding_box":"Hopping_Bbox","sliding_bounding_box":"Sliding_Bbox"}
smap=json.load(open('data/zs_style_map.json'))
if DST.exists(): shutil.rmtree(DST)
per=collections.Counter()
for mp4 in sorted(glob.glob(str(base/'exports'/'*'/'*'/'animation.mp4'))):
    p=mp4.split(os.sep); lin,samp=p[-3],p[-2]
    st=next((s for s in SD if f"__{s}__" in lin), None) or smap.get(samp)
    if not st: continue
    out=DST/SD[st]; out.mkdir(parents=True,exist_ok=True); t=out/f"{samp}.mp4"
    if not t.exists():
        try: os.link(mp4,t)
        except OSError: shutil.copy2(mp4,t)
    per[SD[st]]+=1
print(f"  staged TOTAL {sum(per.values())}")
PY
~/bin/rclone copy data/drive_stage_zs_gemini "gdrive:Zero_Shot_193/AnimateBanana" \
  --transfers 8 --checkers 8 --stats 30s --stats-one-line \
  --log-file "$LOG/upload_gemini_final.log" --log-level INFO
n=$(~/bin/rclone ls "gdrive:Zero_Shot_193/AnimateBanana" 2>/dev/null | wc -l)
say "FINAL: AnimateBanana holds $n files"
