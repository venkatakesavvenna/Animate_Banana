#!/usr/bin/env python3
"""Put each talk's real mp4 where run_eval expects an animation's export.

`_run_animation` (run_eval.py:189-193) returns an error unless
`exports/<lineage>/<id>/frames/` holds at least one frame -- that check runs
BEFORE `--stages` is consulted, so a video-only scoring still needs the dir.

With `--video-source export` the judge reads `exports/<id>/animation.mp4` and
never looks at those frames, so the talk's own `extracted_frames` (2-26 of them,
arbitrary sampling) satisfy the gate without their count reaching any score.
That separation is the whole reason this is safe: `step_frames` would refuse
this deck anyway, and sss/gps/nas are not requested.

The lineage is taken from the config via CachePaths, never hardcoded -- a
hardcoded path silently writes to a directory the eval will not read.
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from img_2_svg_pretraining.pipeline.config import load_config          # noqa: E402
from img_2_svg_pretraining.pipeline.cache import CachePaths            # noqa: E402

CONFIG = REPO / "src/img_2_svg_pretraining/pipeline/configs/presentations_video.yaml"
RAW = REPO / "data/original_presentations_raw/Original_Presentations"
STYLE_MAP = json.loads((REPO / "data/presentations_style_map.json").read_text())


def main() -> int:
    cfg = load_config(CONFIG)
    made = skipped = 0
    for sid, style in sorted(STYLE_MAP.items()):
        src = next((p for p in RAW.glob(f"*/{sid}") if p.is_dir()), None)
        if src is None:
            print(f"  !! {sid}: not found under {RAW}", file=sys.stderr)
            return 1
        # Style is per sample, so CachePaths must be rebuilt per sample --
        # `animation_lineage` is style-keyed and one shared instance would file
        # every talk under whichever style happened to be the config default.
        # This mirrors `run_eval._paths_for` exactly: setting BOTH `cfg.style`
        # and `cfg.raw["animation_style"]` before `from_config`. Setting only
        # one of the two gives a lineage the eval will not look in.
        cfg.style = style
        cfg.raw["animation_style"] = style
        paths = CachePaths.from_config(cfg)
        out = paths.exports(sid)
        frames = out / "frames"
        frames.mkdir(parents=True, exist_ok=True)

        video = out / "animation.mp4"
        if not video.exists():
            shutil.copy2(src / "video.mp4", video)

        have = sorted(frames.glob("*.png"))
        if not have:
            for i, f in enumerate(sorted((src / "extracted_frames").glob("*.png"))):
                shutil.copy2(f, frames / f"frame_{i:04d}.png")
            have = sorted(frames.glob("*.png"))
        if not have:
            print(f"  !! {sid}: no extracted_frames to satisfy the gate", file=sys.stderr)
            skipped += 1
            continue
        made += 1
    print(f"populated {made} export cells ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
