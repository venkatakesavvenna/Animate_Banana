#!/usr/bin/env python3
"""Sequences + per-timestep decks for the Original_Presentations talks.

WHAT MAKES THIS AN HONEST MEASUREMENT. SSS and GPS are judged on
(step_frames, source_image, style, xml_text) -- `selection_sensibility_bands`
and `granularity_pacing_bands` never see the sequence's own content. The
sequence exists only so `run_eval` can map a frame to a timestep, and so NAS has
per-step captions. So the judges are shown the HUMAN TALK'S OWN state frames
against the source figure and its structure XML: nothing about the animation is
inferred from the thing being scored.

The XML is Gemini 3.7 Flash's, reused from the AnimateBanana run rather than
regenerated -- it describes the FIGURE, which is the same figure either way.

STATES COME FROM THE EXTRACTION, NOT FROM US. Frames are named
`state_<n>_frame_<m>.png`; `<n>` is the distinct visual state and `<m>` its
video frame. The deck is [initial, state_1 .. state_N] = N+1 frames against N
timesteps, which is exactly the `n_steps + 1` case `frames.step_frames` accepts
(frame 0 is the pre-animation state) -- so no frame is ever guessed onto a step.

NARRATION is the talk's real transcript, sliced by each state's own time window
(frame number / fps). Only the 11 samples shipping `transcript.json` have
timings; the rest get no narration, so NAS reports them as unnarrated rather
than scoring invented text.
"""
from __future__ import annotations
import json, re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from img_2_svg_pretraining.pipeline.config import load_config      # noqa: E402
from img_2_svg_pretraining.pipeline.cache import CachePaths        # noqa: E402

CONFIG = REPO / "src/img_2_svg_pretraining/pipeline/configs/presentations_video.yaml"
RAW = REPO / "data/original_presentations_raw/Original_Presentations"
STYLE_MAP = json.loads((REPO / "data/presentations_style_map.json").read_text())
FPS = 30.0   # measured: every talk is h264 30fps


def state_frames(sample_dir: Path) -> list[tuple[int, int, Path]]:
    """(state_index, video_frame, path), ordered by video frame."""
    out = []
    for f in sorted((sample_dir / "extracted_frames").glob("*.png")):
        m = re.match(r"state_(\d+)_frame_(\d+)\.png$", f.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), f))
        elif re.match(r"0_frame_(\d+)\.png$", f.name):
            out.append((0, int(re.match(r"0_frame_(\d+)", f.name).group(1)), f))
    out.sort(key=lambda t: t[1])
    return out


def narration_for(transcript: list[dict], t0: float, t1: float) -> str:
    """Transcript text overlapping this state's own on-screen window."""
    parts = [s["text"].strip() for s in transcript
             if s.get("end", 0) > t0 and s.get("start", 0) < t1]
    return " ".join(parts).strip()


def main() -> int:
    cfg = load_config(CONFIG)
    built = no_narr = 0
    for sid, style in sorted(STYLE_MAP.items()):
        src = next((p for p in RAW.glob(f"*/{sid}") if p.is_dir()), None)
        if src is None:
            print(f"  !! {sid}: missing source", file=sys.stderr)
            return 1
        frames = state_frames(src)
        if len(frames) < 2:
            print(f"  !! {sid}: only {len(frames)} frame(s); cannot form steps", file=sys.stderr)
            continue

        tj = src / "transcript.json"
        transcript = []
        if tj.exists():
            try:
                transcript = json.loads(tj.read_text(encoding="utf-8")) or []
            except (OSError, json.JSONDecodeError):
                transcript = []
        if not transcript:
            no_narr += 1

        cfg.style = style
        cfg.raw["animation_style"] = style
        paths = CachePaths.from_config(cfg)

        # Deck: every frame, in order. N+1 frames -> N timesteps.
        steps_dir = paths.exports(sid) / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        for old in steps_dir.glob("*.png"):
            old.unlink()
        for i, (_, _, p) in enumerate(frames):
            shutil.copy2(p, steps_dir / f"step_{i:04d}.png")

        # One node per TRANSITION into a state, i.e. frames[1:].
        nodes, traversal = [], []
        for i in range(1, len(frames)):
            _, fnum, _ = frames[i]
            t0 = frames[i - 1][1] / FPS
            t1 = fnum / FPS
            nid = f"t{i}"
            traversal.append(nid)
            nodes.append({
                "id": nid, "parent": None, "depth": 1, "focus": [],
                "action": "reveal",
                "narration": narration_for(transcript, t0, t1) or None,
                "duration": round(t1 - t0, 2),
            })

        seq = {
            "style": style, "traversal_style": "original_presentation",
            "nodes": nodes, "traversal": traversal,
            "provenance": {
                "source": "human conference talk (Original_Presentations)",
                "states_from": "extracted_frames state_<n>_frame_<m>.png",
                "narration_from": "transcript.json" if transcript else None,
                "fps_assumed": FPS,
                "note": "sequence supplies timestep COUNT and captions only; "
                        "SSS/GPS judge the frames, figure and XML.",
            },
        }
        for target in (paths.sequence(sid), paths.sequence_narrated(sid)):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(seq, indent=2), encoding="utf-8")
        built += 1
        print(f"  {sid:28s} {style:22s} steps={len(traversal):3d} "
              f"narrated={'yes' if transcript else 'NO'}")

    print(f"\nbuilt {built}/35 sequences; {no_narr} without timed narration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
