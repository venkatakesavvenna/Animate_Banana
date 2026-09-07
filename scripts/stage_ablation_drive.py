"""Stage ablation animations as  <Experiment>/<Style>/<sample>.mp4  for Drive.

Layout is EXPERIMENT-FIRST, unlike stage_drive_upload.py which is style-first.
The ablation table is read one experiment at a time ("what did dropping the
Stage-2 critic do?"), so the experiment has to be the folder you open.

Videos are HARD-LINKED where possible: the export tree and this staging dir sit
on one filesystem, so the set costs no extra disk and is instant. A hard link
also means the staged file IS the artifact -- there is no way to ship a stale
copy that has drifted from what the pipeline produced.

The STYLE COMES FROM THE EXPORT LINEAGE, never from the config: style is baked
into the lineage directory that actually produced the bytes, and the config
carries only a default. Reading it from the config would mislabel every cell
whose style differs from that default.
"""
from __future__ import annotations
import os, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "ablation_cache"
DST = REPO / "data" / "drive_stage_ablation"

STYLE_DIR = {
    "progressive_reveal":   "Progressive_Reveal",
    "alpha_masking":        "Alpha_Masking",
    "colour_pop":           "Colour_Pop",
    "hopping_bounding_box": "Hopping_Bbox",
    "sliding_bounding_box": "Sliding_Bbox",
}
# directory name -> the row label used in the ablation table
EXPERIMENT = {
    "full":                 "00_AnimateBanana_Full",
    "stage1_no_critic":     "01_Stage1_no_Critic",
    "stage2_no_image":      "02_Stage2_no_Diagram_Image",
    "sequencer_no_xml":     "03_Sequencer_no_XML",
    "narration_no_context": "04_Narration_no_Context",
    "stage2_no_critic":     "05_Stage2_no_Critic",
    "designer_no_image":    "06_Designer_no_Image",
    "stage3_no_critic":     "07_Stage3_no_Critic",
}

def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): dst.unlink()
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)

def main() -> int:
    if DST.exists(): shutil.rmtree(DST)
    counts: dict[str, int] = {}
    unknown = []
    for mp4 in sorted(SRC.glob("*/animatebench_v5/exports/*/*/animation.mp4")):
        abl = mp4.parts[len(SRC.parts)]
        lineage, sample = mp4.parts[-3], mp4.parts[-2]
        exp = EXPERIMENT.get(abl)
        style = next((s for s in STYLE_DIR if f"__{s}__" in lineage), None)
        if not exp or not style:
            unknown.append(str(mp4)); continue
        link(mp4, DST / exp / STYLE_DIR[style] / f"{sample}.mp4")
        counts[exp] = counts.get(exp, 0) + 1
    for e in sorted(counts): print(f"  {e:<32} {counts[e]}")
    print(f"  TOTAL {sum(counts.values())}")
    if unknown:
        print(f"  UNRESOLVED: {len(unknown)}", file=sys.stderr)
        for u in unknown[:3]: print(f"    {u}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
