#!/usr/bin/env python3
"""Sample each talk video into the frame decks the video judges actually read.

WHY THIS IS NEEDED. `--video-source export` is honoured by `vfs_video` ONLY.
The two BAND judges -- `vfs_band` and `ascs_video` -- call `_slow_video()`
(animation_quality.py:1204-1235), which always builds its mp4 from
`frames_dense/` or `frames/` and never consults `video_source`. Scoring the
talks with only the 2-26 arbitrary `extracted_frames` in place therefore
produced a confident `BAND A` computed from as few as THREE stills, with
`frame_count: 3` the only sign in the record.

So the frames are made real: sampled from the talk's own mp4 at a fixed rate.
Both decks are written from the same video:

  frames/        1 fps  -- the walk deck, and what `vfs`/`ascs` see.
  frames_dense/  2 fps  -- preferred by `_slow_video`, then re-timed to 0.5 fps,
                          so Gemini's 1 fps inline sampler lands on every frame
                          at least twice. Same reasoning as video.py's, applied
                          to a source whose authored fps is 30 rather than 2.

A CAP IS APPLIED. These talks run 24-277s; at 2 fps a 277s talk is 554 frames,
which is a large judged video and a slow render for no gain. `MAX_DENSE` caps
the dense deck by widening the sampling interval, so a long talk is sampled
sparsely rather than truncated -- truncation would judge only the opening and
report nothing about the rest.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from img_2_svg_pretraining.pipeline.config import load_config          # noqa: E402
from img_2_svg_pretraining.pipeline.cache import CachePaths            # noqa: E402

import imageio_ffmpeg                                                   # noqa: E402

CONFIG = REPO / "src/img_2_svg_pretraining/pipeline/configs/presentations_video.yaml"
RAW = REPO / "data/original_presentations_raw/Original_Presentations"
STYLE_MAP = json.loads((REPO / "data/presentations_style_map.json").read_text())

WALK_FPS = 1.0
DENSE_FPS = 2.0
MAX_DENSE = 240
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _duration(video: Path) -> float:
    out = subprocess.run([FFMPEG, "-i", str(video)], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"no duration for {video}")


def _sample(video: Path, out_dir: Path, fps: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(video),
           "-vf", f"fps={fps}", str(out_dir / "frame_%04d.png")]
    subprocess.run(cmd, check=True, capture_output=True)
    return len(list(out_dir.glob("*.png")))


def main() -> int:
    cfg = load_config(CONFIG)
    rows = []
    for sid, style in sorted(STYLE_MAP.items()):
        src = next((p for p in RAW.glob(f"*/{sid}") if p.is_dir()), None)
        if src is None:
            print(f"  !! {sid}: missing source", file=sys.stderr)
            return 1
        video = src / "video.mp4"
        cfg.style = style
        cfg.raw["animation_style"] = style
        paths = CachePaths.from_config(cfg)
        cell = paths.exports(sid)

        dur = _duration(video)
        dense_fps = min(DENSE_FPS, MAX_DENSE / dur) if dur > 0 else DENSE_FPS
        n_walk = _sample(video, cell / "frames", WALK_FPS)
        n_dense = _sample(video, cell / "frames_dense", dense_fps)
        rows.append({"id": sid, "style": style, "duration_s": round(dur, 1),
                     "frames": n_walk, "frames_dense": n_dense,
                     "dense_fps": round(dense_fps, 3)})
        print(f"  {sid:28s} {dur:6.1f}s  walk={n_walk:3d}  dense={n_dense:3d} @ {dense_fps:.2f}fps")

    (REPO / "data/presentations_frames.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} samples; decks written from the real talk videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
