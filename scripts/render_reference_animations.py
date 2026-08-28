"""Render a bench's human-authored SVG animations to mp4.

WHY THIS EXISTS
---------------
The comparison viewer shows the reference animation beside the pipeline's, and
it looks for `reference/videos/<target>__<style>__full.mp4`. The Set5 bundle
ships **21 SVG animations as HTML and not one as video** -- its only three mp4s
are TikZ. So without this pass every SVG row in the viewer has an empty
reference column: nothing to compare against, on a bench whose whole point is
the comparison.

The bundle authors rendered TikZ to video because a TikZ animation is a
multipage PDF and pdftoppm walks it. An SVG animation is CSS `@keyframes`,
which no rasteriser can seek -- so it needs a real browser, which is why the
annotation tool never produced these.

REUSES THE PIPELINE'S OWN EXPORT PATH. `render_svg_frames` + `frames_to_mp4`
are the same two calls `animator/exporter.py` makes for a generated animation.
That matters beyond tidiness: reference and prediction videos are then produced
by identical code at identical fps, so a visible difference between the two
panels is a difference in the animation rather than in how it was filmed.

Needs chromium, so it runs in the container:

    docker exec -u $(id -u):$(id -g) animatebanana-v4 bash -lc \\
      'cd /code && /environments/img_2_svg_pretraining/bin/python -u \\
         scripts/render_reference_animations.py --dataset data/animatebench_v4'
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from img_2_svg_pretraining.pipeline.export.render import frames_to_mp4  # noqa: E402
from img_2_svg_pretraining.pipeline.export.svg_frames import render_svg_frames  # noqa: E402

# `<style>_<sample>_<target>.html`, as written by import_bench into
# reference/animation/.
TARGETS = ("svg", "tikz")


def parse_name(stem: str, sample_id: str) -> tuple[str, str] | None:
    """(style, target) from `<style>_<sample_id>_<target>`."""
    for target in TARGETS:
        suffix = f"_{sample_id}_{target}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], target
    return None



def _register(sample_dir: Path, target: str, style: str, name: str) -> bool:
    """Record a rendered video in reference/index.json.

    WRITING THE MP4 IS NOT ENOUGH. The comparison viewer resolves a reference
    by looking up `"<target>|<style>|full"` in `reference/index.json`, NOT by
    globbing `reference/videos/` -- so a file this script renders but never
    registers is invisible, and the viewer shows an empty reference column
    while the mp4 sits right there on disk.

    That is exactly what happened to the Set5 bundle: `import_bench` writes the
    index once, indexing only the videos it copied (all TikZ), and this script
    then rendered 20 SVG videos 47 minutes later without touching it. Every
    SVG reference read as "not generated".

    Returns True when the index gained an entry.
    """
    index_path = sample_dir / "reference" / "index.json"
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    videos = data.setdefault("videos", {})
    key = f"{target}|{style}|full"
    if videos.get(key) == name:
        return False
    videos[key] = name
    index_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=2,
                    help="matches the exporter default, so reference and "
                         "prediction videos are filmed identically")
    ap.add_argument("--only", nargs="+", help="sample ids")
    ap.add_argument("--force", action="store_true",
                    help="re-render even where an mp4 already exists")
    args = ap.parse_args()

    samples = sorted(p for p in args.dataset.iterdir() if p.is_dir())
    if args.only:
        wanted = set(args.only)
        samples = [s for s in samples if s.name in wanted]

    made = skipped = failed = repaired = 0
    for sample in samples:
        anim_dir = sample / "reference" / "animation"
        if not anim_dir.is_dir():
            continue
        for src in sorted(anim_dir.glob("*.html")):
            parsed = parse_name(src.stem, sample.name)
            if not parsed:
                print(f"  ?? {sample.name}: cannot parse '{src.name}'")
                continue
            style, target = parsed
            # The name the viewer looks for. "full" is the context tier: these
            # are human-authored, which is the most context there is.
            out = sample / "reference" / "videos" / f"{target}__{style}__full.mp4"
            if out.exists() and not args.force:
                # Still register it: a previous run may have written the mp4
                # before this script learned to update the index, which is the
                # state that made 20 SVG references invisible.
                if _register(sample, target, style, out.name):
                    repaired += 1
                skipped += 1
                continue

            work = sample / "reference" / "_frames" / f"{style}_{target}"
            try:
                # base_dir = the document's OWN directory. These references
                # load crops as `../rasters/<name>.png`, relative to
                # reference/animation/ where they were authored -- not to the
                # scratch dir the frames are rendered into. Without this every
                # raster resolves to a path that does not exist, and chromium
                # paints an empty box rather than failing.
                frames, used_fps = render_svg_frames(
                    src.read_text(encoding="utf-8"), work, fps=args.fps,
                    base_dir=src.parent)
                if not frames:
                    raise RuntimeError("no frames rendered")
                out.parent.mkdir(parents=True, exist_ok=True)
                frames_to_mp4(frames, out, fps=used_fps)
            except Exception as exc:                       # noqa: BLE001
                # One unrenderable reference must not cost the other twenty.
                # These are hand-authored documents; a stray tag or a missing
                # crop is a fact about the annotation, not a reason to stop.
                failed += 1
                print(f"  !! {sample.name} {style}/{target}: "
                      f"{type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2, file=sys.stdout)
                continue
            made += 1
            _register(sample, target, style, out.name)
            print(f"  ok {sample.name} {style}/{target}: "
                  f"{len(frames)} frames @ {used_fps}fps -> {out.name}")

    print(f"\n{made} rendered, {skipped} already present "
          f"({repaired} index entries repaired), {failed} failed")
    if failed:
        print("failures above are per-animation; everything else was written")


if __name__ == "__main__":
    main()
