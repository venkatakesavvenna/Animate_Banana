"""Import an AnimateBench sample bundle into a pipeline dataset.

The benchmark ships one zip per sample, each holding the paper inputs, the
reference pipeline's intermediates (structure XML, sequences, narrations), its
rendered animations, and human review metadata. This turns that layout into
the flat per-sample directory `samples.py::discover_samples` expects, while
keeping the reference material alongside for side-by-side comparison.

Two layout quirks the bundles actually have, both handled here rather than
assumed away:

- **Video directories are named inconsistently between bundles.** pipe00002
  uses `codes/animation_tikz_videos/full_context/`; pipe00041 uses
  `codes/tikz_animations_full_context/`. Both mean the same thing, so videos
  are located by filename convention (`<target>_<style>_<tier>_<id>.mp4`)
  rather than by directory path.
- **The title lives in `<id>_title.txt`**, which `samples.py` does not look
  for -- it reads titles out of `arxiv_src/main.tex`. The importer writes a
  `title.txt` that the loader does find, so the `full` context tier resolves.

Reference artifacts land under `reference/` inside each sample directory,
where the pipeline ignores them (discovery only looks for the image and the
context files) but the comparison viewer can find them.

Usage:
    python -m img_2_svg_pretraining.pipeline.import_bench \\
        --src /path/to/extracted/bench --dest /code/data/animatebench
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

# `<target>_<style>_<tier>_<sample id>.mp4`, e.g.
# tikz_progressive_reveal_img_and_context_CVPR_2025_pipe00002.mp4
VIDEO_RE = re.compile(
    r"^(?P<target>tikz|svg)_(?P<style>.+?)_(?P<tier>img_and_context|img_only)_"
    r"(?P<sample>.+)\.mp4$"
)

# The bundles say "img_and_context"; the pipeline's tier vocabulary calls the
# same thing "full". Normalizing here keeps one name in everything downstream.
TIER_ALIAS = {"img_and_context": "full", "img_only": "image_only"}

CONTEXT_FILES = ("abstract.tex", "abstract.txt", "caption.txt",
                 "methods.tex", "methods.txt")


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def import_sample(sample_dir: Path, dest_root: Path) -> dict:
    """Import one extracted bundle. Returns a summary of what was found."""
    sample_id = sample_dir.name
    out = dest_root / sample_id
    out.mkdir(parents=True, exist_ok=True)

    summary = {"id": sample_id, "videos": 0, "sequences": 0, "narrations": 0,
               "rasters": 0, "context": [], "image": None}

    inputs = sample_dir / "inputs"

    image = inputs / f"{sample_id}.png"
    if image.exists():
        _copy(image, out / f"{sample_id}.png")
        summary["image"] = image.name

    for name in CONTEXT_FILES:
        src = inputs / name
        if src.exists():
            _copy(src, out / name)
            summary["context"].append(name)

    # samples.py reads the title from arxiv_src/main.tex, which these bundles
    # do not ship -- without this the `full` tier would silently lose the title.
    title = inputs / f"{sample_id}_title.txt"
    if title.exists():
        _copy(title, out / "title.txt")
        summary["context"].append("title.txt")

    ref = out / "reference"

    for sub in ("intermediates/xml", "intermediates/seq", "intermediates/narration"):
        src_dir = sample_dir / sub
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.iterdir()):
            if f.is_file():
                _copy(f, ref / Path(sub).name / f.name)
                if sub.endswith("seq"):
                    summary["sequences"] += 1
                elif sub.endswith("narration"):
                    summary["narrations"] += 1

    for f in sorted((sample_dir / "rasters").glob("*")) if (sample_dir / "rasters").is_dir() else []:
        if f.is_file():
            _copy(f, ref / "rasters" / f.name)
            summary["rasters"] += 1

    for f in sorted((sample_dir / "codes" / "diagram").glob("*")) if (sample_dir / "codes" / "diagram").is_dir() else []:
        if f.is_file():
            _copy(f, ref / "diagram" / f.name)

    # Videos: located by filename, since the directory naming differs between
    # bundles. Flattened into reference/videos/ under a canonical name so the
    # viewer can address one by (target, style, tier).
    index: dict[str, str] = {}
    for f in sorted(sample_dir.rglob("*.mp4")):
        m = VIDEO_RE.match(f.name)
        if not m:
            continue
        tier = TIER_ALIAS.get(m.group("tier"), m.group("tier"))
        key = f"{m.group('target')}|{m.group('style')}|{tier}"
        if key in index:
            continue   # same video reachable by two paths in some bundles
        name = f"{m.group('target')}__{m.group('style')}__{tier}.mp4"
        _copy(f, ref / "videos" / name)
        index[key] = name
        summary["videos"] += 1

    # Human review metadata: which attempts were rejected and why.
    reviews = []
    for f in sorted(sample_dir.rglob("*.meta.json")):
        try:
            reviews.append({"file": f.name, **json.loads(f.read_text())})
        except (OSError, json.JSONDecodeError):
            continue

    (ref / "index.json").parent.mkdir(parents=True, exist_ok=True)
    (ref / "index.json").write_text(json.dumps({
        "sample_id": sample_id,
        "videos": index,
        "reviews": reviews,
    }, indent=2), encoding="utf-8")
    summary["reviews"] = len(reviews)

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="directory holding the extracted sample bundles")
    ap.add_argument("--dest", required=True, type=Path,
                    help="dataset root to write (e.g. /code/data/animatebench)")
    args = ap.parse_args()

    bundles = [d for d in sorted(args.src.iterdir())
               if d.is_dir() and (d / "inputs").is_dir()]
    if not bundles:
        raise SystemExit(f"no sample bundles (dirs containing inputs/) under {args.src}")

    args.dest.mkdir(parents=True, exist_ok=True)
    for bundle in bundles:
        s = import_sample(bundle, args.dest)
        print(f"{s['id']}: image={s['image']} context={len(s['context'])} "
              f"videos={s['videos']} seq={s['sequences']} narration={s['narrations']} "
              f"rasters={s['rasters']} reviews={s['reviews']}")
    print(f"\n{len(bundles)} sample(s) -> {args.dest}")


if __name__ == "__main__":
    main()
