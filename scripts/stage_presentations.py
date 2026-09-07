#!/usr/bin/env python3
"""Stage Original_Presentations into a dataset + a pre-populated export tree.

WHAT THESE ARE. The 35 samples are the AUTHORS' REAL CONFERENCE TALKS -- human
recordings, not pipeline output. There is no sequence and no structure XML for
any of them, and none can be honestly synthesised: a sequence reverse-engineered
from a finished video would score near-perfectly by construction.

THAT IS WHY ONLY THE VIDEO METRICS RUN HERE. `run_eval._run_animation` builds
`step_frames_` only when BOTH `paths.sequence()` and `paths.xml()` exist, and
`animation_quality.run` gates sss/gps/nas on it -- so those three record
`stages_skipped` and produce nothing. `vfs_band` and `ascs_video` need only the
mp4 and the source image, which every sample has.

`_run_animation` ALSO hard-returns an error unless `exports/<id>/frames/` holds
at least one frame, whatever `--stages` asks for. The talk's own
`extracted_frames/` are copied there to satisfy that gate; with
`--video-source export` the judge reads `exports/<id>/animation.mp4` -- the real
talk video -- and never the frames, so the arbitrary frame count cannot reach a
score.
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data/original_presentations_raw/Original_Presentations"
DS = REPO / "data/presentations_ds"

# Folder name in the zip -> the pipeline's style token.
STYLE = {"Alpha_Masking": "alpha_masking", "Colour_Pop": "colour_pop",
         "Hopping_Bounding_Box": "hopping_bounding_box",
         "Progressive_Reveal": "progressive_reveal",
         "Sliding_Bounding_Box": "sliding_bounding_box"}


def main() -> int:
    if not SRC.is_dir():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    DS.mkdir(parents=True, exist_ok=True)
    style_map, rows = {}, []

    for style_dir in sorted(SRC.iterdir()):
        if not style_dir.is_dir():
            continue
        style = STYLE.get(style_dir.name)
        if style is None:
            print(f"  !! unknown style folder {style_dir.name}", file=sys.stderr)
            return 1
        for s in sorted(style_dir.iterdir()):
            if not s.is_dir():
                continue
            sid, out = s.name, DS / s.name
            out.mkdir(parents=True, exist_ok=True)

            # The source figure. Named `<id>.png` because PaperSample discovery
            # looks for exactly that, the same as every other bench dataset.
            img = s / "image.png"
            if not img.exists():
                print(f"  !! {sid}: no image.png", file=sys.stderr)
                return 1
            shutil.copy2(img, out / f"{sid}.png")

            # Context. The talk's metadata carries the paper title; there is no
            # caption/abstract/methods for these, so the tier stays `image_only`
            # -- stated here rather than discovered later as a silent downgrade.
            meta = {}
            mp = s / "metadata.json"
            if mp.exists():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8")) or {}
                except (OSError, json.JSONDecodeError):
                    meta = {}
            title = str(meta.get("title") or "").strip()
            if title:
                (out / "title.txt").write_text(title + "\n", encoding="utf-8")

            ref = out / "reference" / "original_presentation"
            ref.mkdir(parents=True, exist_ok=True)
            for name in ("video.mp4", "audio.mp3", "transcript.txt",
                         "transcript.json", "metadata.json"):
                if (s / name).exists():
                    shutil.copy2(s / name, ref / name)

            frames = sorted((s / "extracted_frames").glob("*.png"))
            style_map[sid] = style
            rows.append({"id": sid, "style": style, "frames": len(frames),
                         "timed_transcript": (s / "transcript.json").exists()})

    (REPO / "data/presentations_style_map.json").write_text(
        json.dumps(style_map, indent=2, sort_keys=True), encoding="utf-8")
    (REPO / "data/presentations_manifest.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print(f"staged {len(rows)} samples -> {DS}")
    from collections import Counter
    for k, v in sorted(Counter(r["style"] for r in rows).items()):
        print(f"   {k:24s} {v}")
    print(f"   timed transcripts: {sum(r['timed_transcript'] for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
