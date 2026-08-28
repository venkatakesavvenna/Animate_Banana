"""Score the GROUND-TRUTH reference animation with the same metrics as a pipeline.

Why this is worth doing: every metric in the animation tree has been used only
to rank pipeline outputs against each other, with no measurement of what the
metric says about a human-authored animation. Running the identical judges over
the reference gives the one number the study actually needs -- a ceiling. A
metric that fails the ground truth is telling you about itself, not about the
pipeline.

The reference is not a pipeline artifact, so three things have to be supplied
that `run_eval` normally resolves from `CachePaths`:

  video   `reference/videos/<target>__<style>__full.mp4`, shipped in the bundle.
  frames  NOT shipped. The bundle has video only, so SSS/GPS -- which compare
          consecutive timesteps -- have nothing to run on until a deck is
          extracted. `animatebench.keyframes` does that, per style.
  xml     `reference/xml/<sample>_<target>.xml`, the human-authored structure.

TREATMENT IS HELD IDENTICAL to the pipeline path on purpose: same prompts, same
judges, same 4x slowdown, same 0.5 fps judged video. Any difference in the
scores then belongs to the animation rather than to how it was measured, which
is the only way the comparison means anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from img_2_svg_pretraining.animatebench import keyframes as kf
from img_2_svg_pretraining.animatebench.frames import frame_set
from img_2_svg_pretraining.animatebench.judge import Judge
from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq
from img_2_svg_pretraining.animatebench.video import SLOWDOWN_VIDEO_FPS, judge_video
from img_2_svg_pretraining.pipeline.config import load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "src/img_2_svg_pretraining/pipeline/configs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--target", default="svg")
    ap.add_argument("--evals-root", required=True)
    ap.add_argument("--video-backend", default="gemini_video")
    ap.add_argument("--frame-backend", default="qwen38_flash_judge")
    ap.add_argument("--stages", nargs="+",
                    default=["vfs_band", "ascs_video", "sss", "gps"])
    args = ap.parse_args()

    cfg = load_config(CONFIG_DIR / args.config)
    sample_dir = Path(cfg.dataset_root) / args.sample
    ref = sample_dir / "reference"

    source_image = sample_dir / f"{args.sample}.png"
    video = ref / "videos" / f"{args.target}__{args.style}__full.mp4"
    xml_path = ref / "xml" / f"{args.sample}_{args.target}.xml"
    if not video.exists():
        raise SystemExit(f"no reference video at {video}")

    out_root = Path(args.evals_root) / "reference" / args.style / args.sample
    out_root.mkdir(parents=True, exist_ok=True)
    deck_dir = out_root / "keyframes"

    record: dict = {"suite": "animation", "style": args.style,
                    "model": "reference (ground truth)",
                    "stages_requested": list(args.stages), "stages_run": [],
                    "stages_skipped": {}}

    # -- the deck --------------------------------------------------------
    keys = kf.extract(video, deck_dir, args.style)
    record["reference_keyframes"] = len(keys.paths)
    record["reference_keyframe_method"] = keys.method
    record["reference_source_frames"] = keys.source_frame_count
    print(f"  keyframes: {len(keys.paths)} via {keys.method} (from {keys.source_frame_count} source frames)")

    # The judged video is rebuilt from the extracted deck rather than sent as
    # shipped, so the reference gets exactly the treatment the pipeline videos
    # get: one frame per distinct state, played at the 4x-slowed rate.
    deck = frame_set(deck_dir, "all", 1568, cache_dir=out_root / "prepared")
    slow = judge_video(deck, out_root / f"judge_slow_{SLOWDOWN_VIDEO_FPS}fps.mp4",
                       fps=SLOWDOWN_VIDEO_FPS)
    meta = {"source": "reference_keyframes", "path": str(slow),
            "bytes": slow.stat().st_size, "fps": SLOWDOWN_VIDEO_FPS,
            "frame_count": len(deck.paths)}

    def attempt(name, fn):
        if name not in args.stages:
            return
        try:
            record.update(fn())
            record["stages_run"].append(name)
            print(f"  {name}: ok")
        except Exception as e:                       # noqa: BLE001 - one stage
            record["stages_skipped"][name] = f"{type(e).__name__}: {e}"
            print(f"  {name}: SKIPPED ({type(e).__name__}: {e})")

    responses = Path(cfg.cache_root) / Path(cfg.dataset_root).name / 'responses'
    video_judge = Judge(args.video_backend,
                        cfg.backend_cfg(args.video_backend),
                        cache_root=responses.parent)
    attempt("vfs_band", lambda: aq.video_fidelity_bands(
        video_judge, source_image, slow, args.style, meta))
    attempt("ascs_video", lambda: aq.style_compliance_video(
        video_judge, source_image, slow, args.style, meta))

    if {"sss", "gps"} & set(args.stages):
        xml_text = xml_path.read_text(encoding="utf-8") if xml_path.exists() else ""
        if not xml_text:
            for name in ("sss", "gps"):
                record["stages_skipped"][name] = f"no reference XML at {xml_path}"
        else:
            frame_judge = Judge(args.frame_backend,
                                cfg.backend_cfg(args.frame_backend),
                                cache_root=responses.parent)
            steps = [Path(p) for p in deck.paths]
            attempt("sss", lambda: aq.selection_sensibility_bands(
                frame_judge, source_image, steps, args.style, xml_text))
            attempt("gps", lambda: aq.granularity_pacing_bands(
                frame_judge, source_image, steps, args.style, xml_text))

    out = out_root / "animation.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  written {out}")


if __name__ == "__main__":
    main()
