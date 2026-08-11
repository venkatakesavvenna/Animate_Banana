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


# -- v2 bundle layout -------------------------------------------------------
#
# The August 2026 bench reorganised every sample: context files moved to
# `diagram_context/` under `<id>_<field>` names, artifacts are nested by
# style then target rather than flattened, and each sample gained an
# `original_presentation/` -- the author's real conference talk, which is a
# genuinely new reference the earlier bundles had no equivalent of.
#
# Both layouts are supported rather than migrating, because the v1 dataset is
# already scored and re-importing it would invalidate every cached alignment.

# Style directory name -> the suffix used in that style's file names.
V2_STYLE_CODE = {
    "alpha_masking": "AM",
    "colour_pop": "CP",
    "hopping_bounding_box": "HB",
    "progressive_reveal": "PR",
    "sliding_bounding_box": "SB",
}

# `<id>_<suffix>` in diagram_context/ -> the name discovery expects.
V2_CONTEXT = {
    "abstract.tex": "abstract.tex",
    "caption.txt": "caption.txt",
    "method.tex": "methods.tex",     # singular in the bundle, plural in ours
    "title.txt": "title.txt",
}


def is_v2(directory: Path) -> bool:
    return (directory / "diagram_context").is_dir()


# Phrases that identify the CVPR submission template rather than a paper.
# Matched case-insensitively against the whole shipped title.
_BOILERPLATE_MARKERS = (
    "guidelines for author response",
    "author guidelines",
    "paper template",
    "submission and formatting",
)


def _looks_like_boilerplate(title: str) -> bool:
    """Is this the LaTeX template's own title rather than the paper's?"""
    lowered = title.lower()
    return any(marker in lowered for marker in _BOILERPLATE_MARKERS)


def _v2_talk_title(sample_dir: Path) -> str | None:
    """The paper title as given by the author's own conference talk.

    Stripped of the venue prefix the uploads carry ("[CVPR 2025] ..."), so it
    reads the same way a title.txt would.
    """
    meta = sample_dir / "original_presentation" / "metadata.json"
    if not meta.exists():
        return None
    try:
        title = str((json.loads(meta.read_text(encoding="utf-8")) or {}).get("title", ""))
    except (OSError, json.JSONDecodeError):
        return None
    title = re.sub(r"^\s*\[[^\]]*\]\s*", "", title).strip()
    return title or None


def import_sample_v2(sample_dir: Path, dest_root: Path) -> dict:
    """Import one v2 bundle. Same output contract as `import_sample`."""
    sample_id = sample_dir.name
    out = dest_root / sample_id
    out.mkdir(parents=True, exist_ok=True)

    summary = {"id": sample_id, "videos": 0, "sequences": 0, "narrations": 0,
               "rasters": 0, "context": [], "image": None, "reviews": 0,
               "layout": "v2"}

    ctx = sample_dir / "diagram_context"
    image = ctx / f"{sample_id}.png"
    if image.exists():
        _copy(image, out / f"{sample_id}.png")
        summary["image"] = image.name

    for suffix, target in V2_CONTEXT.items():
        src = ctx / f"{sample_id}_{suffix}"
        if src.exists():
            _copy(src, out / target)
            summary["context"].append(target)

    # The shipped title is sometimes the CVPR template boilerplate rather than
    # the paper's own title (`\LaTeX\ Guidelines for Author Response` on
    # pipe00004). The title feeds the full-context tier, so a wrong one grounds
    # the strategizer and the narration in the wrong paper entirely. The talk
    # metadata carries the real title, so prefer it when the shipped file is
    # recognisably boilerplate. Substitutions are recorded, never silent.
    title_path = out / "title.txt"
    talk_title = _v2_talk_title(sample_dir)
    if talk_title:
        shipped = title_path.read_text(encoding="utf-8").strip() if title_path.exists() else ""
        if not shipped or _looks_like_boilerplate(shipped):
            title_path.write_text(talk_title, encoding="utf-8")
            summary["title_replaced"] = {"was": shipped[:80], "now": talk_title[:80]}
            if "title.txt" not in summary["context"]:
                summary["context"].append("title.txt")

    ref = out / "reference"

    # Structure XML, per target.
    for target in ("tikz", "svg"):
        src = sample_dir / "intermediates" / "xml" / target / f"{sample_id}_{target}_xml.xml"
        if src.exists():
            _copy(src, ref / "xml" / f"{sample_id}_{target}.xml")

    # Animation and narration sequences, renamed to `<style>_<id>_<target>`
    # so `gt.py` finds them without knowing about the style-code suffixes.
    for kind, folder, out_dir in (("aniseq", "animation_sequence", "seq"),
                                  ("narrseq", "narration_sequence", "narration")):
        for style, code in V2_STYLE_CODE.items():
            for target in ("tikz", "svg"):
                src = (sample_dir / "intermediates" / folder / style / target /
                       f"{sample_id}_{kind}_{target}_{code}.json")
                if not src.exists():
                    continue
                _copy(src, ref / out_dir / f"{style}_{sample_id}_{target}.json")
                summary["sequences" if kind == "aniseq" else "narrations"] += 1

    # Rendered animations, flattened to the canonical comparison name.
    index: dict[str, str] = {}
    for style, code in V2_STYLE_CODE.items():
        for target in ("tikz", "svg"):
            src = (sample_dir / "animation" / style / target / "videos" /
                   f"{sample_id}_video_{target}_{code}.mp4")
            if not src.exists():
                continue
            # The v2 bundle ships one video per style, produced with full
            # context; keep the same key shape the comparison viewer uses.
            name = f"{target}__{style}__full.mp4"
            _copy(src, ref / "videos" / name)
            index[f"{target}|{style}|full"] = name
            summary["videos"] += 1

    for f in sorted((sample_dir / "rasters").glob("*")) if (sample_dir / "rasters").is_dir() else []:
        if f.is_file():
            _copy(f, ref / "rasters" / f.name)
            summary["rasters"] += 1

    for target in ("tikz", "svg"):
        for f in sorted((sample_dir / "diagram_codes" / target).glob("*")):
            if f.is_file():
                _copy(f, ref / "diagram" / f.name)

    # New in v2: the author's actual conference talk. Not scored yet, but it
    # is the only human-authored presentation of these figures in existence,
    # so it is imported rather than dropped.
    orig = sample_dir / "original_presentation"
    presentation = {}
    if orig.is_dir():
        for f in sorted(orig.iterdir()):
            if f.is_file():
                _copy(f, ref / "original_presentation" / f.name)
        meta = orig / "metadata.json"
        if meta.exists():
            try:
                presentation = json.loads(meta.read_text())
            except (OSError, json.JSONDecodeError):
                presentation = {}

    (ref / "index.json").parent.mkdir(parents=True, exist_ok=True)
    (ref / "index.json").write_text(json.dumps({
        "sample_id": sample_id, "layout": "v2", "videos": index,
        "reviews": [], "original_presentation": presentation,
    }, indent=2), encoding="utf-8")

    summary["presentation"] = bool(presentation)
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
               if d.is_dir() and ((d / "inputs").is_dir() or is_v2(d))]
    if not bundles:
        raise SystemExit(
            f"no sample bundles under {args.src} "
            f"(expected dirs containing inputs/ or diagram_context/)")

    args.dest.mkdir(parents=True, exist_ok=True)
    for bundle in bundles:
        s = (import_sample_v2 if is_v2(bundle) else import_sample)(bundle, args.dest)
        extra = " talk" if s.get("presentation") else ""
        print(f"{s['id']}: [{s.get('layout', 'v1')}] image={s['image']} "
              f"context={len(s['context'])} videos={s['videos']} seq={s['sequences']} "
              f"narration={s['narrations']} rasters={s['rasters']}{extra}")
    print(f"\n{len(bundles)} sample(s) -> {args.dest}")


if __name__ == "__main__":
    main()
