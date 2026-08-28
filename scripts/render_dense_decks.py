"""Re-render an animation's frames at the ADAPTIVE sampling rate, for video judges.

WHY THIS EXISTS
---------------
`exporter._export_svg` calls `render_svg_frames(code, dir, fps=fps)` with the
config's `exporter.fps` (2 in every bench config). Because `fps` is not None,
`svg_frames.suggest_fps` -- the adaptive sampler written so that "no stage can
pass by unsampled" -- is NEVER reached. The deck is therefore a fixed 2 Hz
sampling of animation time, and any transition shorter than 500 ms simply is
not in it.

That is fatal for exactly one question: hopping vs sliding. The two styles are
distinguishable ONLY while the box is in transit. A slide whose transit fell
between two samples produces a deck in which the box only ever rests on
targets -- which is the literal definition the ASC prompt gives for HOPPING. The
judge then returns ACCEPT for a hopping cell, or DISCARD for a sliding one, with
a confident timestamp, from evidence that was never captured. No amount of
slowing playback down recovers it; the frames do not exist.

`suggest_fps` instead takes the shortest gap between stage transitions and
samples SAMPLES_PER_STAGE (3) times within it, bounded to [4, 30] fps.

WRITTEN BESIDE, NOT OVER. The dense deck lands in `frames_dense/`, leaving
`frames/` and `animation.mp4` byte-identical. Every number already scored off
the 2 Hz deck stays reproducible, and a judge can be pointed at either -- which
is what makes "did the density change the verdict?" an answerable question
rather than a lost one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from img_2_svg_pretraining.pipeline.cache import CachePaths
from img_2_svg_pretraining.pipeline.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "src/img_2_svg_pretraining/pipeline/configs"

# The styles whose whole verdict turns on motion between stages.
MOTION_STYLES = {"sliding_bounding_box", "hopping_bounding_box", "sliding_bbox"}
MOTION_FPS = 25
# Raised from the 600 default so a long animation is covered end to end rather
# than sampled up to the cap and silently cut short.
MAX_DENSE_FRAMES = 5000


def animation_code(paths: CachePaths, sample_id: str) -> Path | None:
    """The animation actually exported: the critic's output if it produced one."""
    final = paths.animation_final(sample_id)
    if final.exists():
        return final
    raw = paths.animation(sample_id)
    return raw if raw.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config filename in configs/")
    ap.add_argument("--style", required=True)
    ap.add_argument("--only", nargs="+", required=True, metavar="ID")
    ap.add_argument("--fps", type=int, default=None,
                    help="sampling rate. Default: adaptive (suggest_fps) for "
                         "most styles, MOTION_FPS for the bounding-box styles.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from img_2_svg_pretraining.pipeline.export.svg_frames import render_svg_frames

    cfg = load_config(CONFIG_DIR / args.config)
    cfg.style = args.style
    cfg.raw["animation_style"] = args.style
    paths = CachePaths.from_config(cfg)

    for sample_id in args.only:
        code_path = animation_code(paths, sample_id)
        if code_path is None:
            print(f"  !! {sample_id}: no animation on disk, skipping")
            continue
        exports = paths.exports(sample_id)
        sparse = len(list((exports / "frames").glob("*.png")))
        out_dir = exports / "frames_dense"
        if args.dry_run:
            print(f"  -- {sample_id}: would densify (sparse deck = {sparse})")
            continue

        # fps=None is what reaches suggest_fps(). For the bounding-box styles
        # that is not enough: suggest_fps derives its rate from the gap between
        # STAGE boundaries, which says nothing about how long a slide between
        # two stages takes. Measured on real sliding animations, unique states
        # keep climbing well past the suggested rate (26 -> 50 -> 76 -> 194 at
        # 2/4/10/25 fps), so the rate is pinned instead. Hopping animations are
        # genuinely discrete and return the same deck at every rate, so this
        # costs nothing where it is not needed.
        fps = args.fps or (MOTION_FPS if args.style in MOTION_STYLES else None)
        frames, used_fps = render_svg_frames(
            code_path.read_text(encoding="utf-8"), out_dir, fps=fps,
            # The default 600 stops a 41s animation at 20s once fps is raised.
            max_frames=MAX_DENSE_FRAMES)
        meta = {"sample": sample_id, "style": args.style,
                "config": args.config, "source": str(code_path),
                "sparse_frames": sparse, "dense_frames": len(frames),
                "dense_fps": used_fps}
        (out_dir / "dense.json").write_text(json.dumps(meta, indent=2),
                                            encoding="utf-8")
        print(f"  ok {sample_id}: {sparse} -> {len(frames)} frames @ {used_fps}fps")


if __name__ == "__main__":
    main()
