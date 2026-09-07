#!/usr/bin/env bash
# Wait for the zs Gemini fill to finish, then REPLACE gdrive:Zero_Shot_193/
# AnimateBanana with the corrected set.
#
# The folder currently there was uploaded before the raster repair: 24 of its
# 155 videos are missing their spliced raster crops (a cached 402 from 09-02
# was replayed as if it were a detection result). Those are the "corrupt" ones.
# Purge is scoped to that ONE folder -- the four baseline models and
# Original_Images beside it must not be touched.
set -uo pipefail
REPO=/fsxvision_new/venkat.kesav/img_2_svg_pretraining
cd "$REPO"
LOG=$REPO/logs/drive; mkdir -p "$LOG"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/replace_gemini.log"; }

say "waiting for the zs gemini fill to finish"
while [ -f logs/zs_gemini/run.pid ] && kill -0 "$(cat logs/zs_gemini/run.pid 2>/dev/null)" 2>/dev/null; do sleep 120; done
say "generation finished"

python3 - <<'PY' | tee -a "$LOG/replace_gemini.log"
import glob,os,shutil,json,collections
from pathlib import Path
base=Path('data/animatebench_zs_or_cache/animatebench_zs')
DST=Path('data/drive_stage_zs_gemini')
SD={"progressive_reveal":"Progressive_Reveal","alpha_masking":"Alpha_Masking",
    "colour_pop":"Colour_Pop","hopping_bounding_box":"Hopping_Bbox",
    "sliding_bounding_box":"Sliding_Bbox"}
smap=json.load(open('data/zs_style_map.json'))
if DST.exists(): shutil.rmtree(DST)
per=collections.Counter()
for mp4 in sorted(glob.glob(str(base/'exports'/'*'/'*'/'animation.mp4'))):
    p=mp4.split(os.sep); lin,samp=p[-3],p[-2]
    st=next((s for s in SD if f"__{s}__" in lin), None) or smap.get(samp)
    if not st: continue
    out=DST/SD[st]; out.mkdir(parents=True,exist_ok=True)
    t=out/f"{samp}.mp4"
    if not t.exists():
        try: os.link(mp4,t)
        except OSError: shutil.copy2(mp4,t)
    per[SD[st]]+=1
for k,v in sorted(per.items()): print(f"  staged {k:<22} {v}")
print(f"  staged TOTAL {sum(per.values())}")
PY

say "purging the old AnimateBanana folder (155 corrupt files)"
~/bin/rclone purge "gdrive:Zero_Shot_193/AnimateBanana" >>"$LOG/replace_gemini.log" 2>&1
say "uploading corrected set"
~/bin/rclone copy data/drive_stage_zs_gemini "gdrive:Zero_Shot_193/AnimateBanana" \
  --transfers 8 --checkers 8 --drive-chunk-size 32M \
  --stats 30s --stats-one-line --log-file "$LOG/upload_gemini_fixed.log" --log-level INFO
n=$(~/bin/rclone ls "gdrive:Zero_Shot_193/AnimateBanana" 2>/dev/null | wc -l)
say "DONE -- AnimateBanana now holds $n files"
