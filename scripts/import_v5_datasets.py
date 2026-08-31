"""Import the v5 "Full_eval_dir_*" bundles into one pipeline dataset.

WHY A NEW IMPORTER
------------------
`pipeline/import_bench.py` recognises two bundle layouts: v1 (an `inputs/`
directory) and v2 (`diagram_context/`). These bundles are a THIRD shape --
flat per-sample files plus a `references/` directory (note the plural; v2 used
the singular `reference/`, which is also what the pipeline dataset itself
uses). Neither detector fires, so `import_bench.py` exits with "no sample
bundles found" rather than doing anything wrong. This converts the new shape
into exactly the on-disk layout `data/animatebench_v3` already has, so every
downstream tool -- bench_v3_styles.py, the metric suites, the compare viewer --
works unchanged.

    python3 scripts/import_v5_datasets.py --src data/v5_datasets --dest data/animatebench_v5

MAPPING
    <id>.png                                    -> <id>/<id>.png
    <id>_title.txt                              -> <id>/title.txt
    <id>_abstract.tex                           -> <id>/abstract.tex
    <id>_method.tex                             -> <id>/methods.tex     (NOTE: plural)
    <id>_caption.tex                            -> <id>/caption.tex
    references/intermediates/xml/svg/*.xml      -> <id>/reference/xml/<id>_svg.xml
    references/diagram_codes/svg/*.html         -> <id>/reference/diagram/<id>_diag_svg.html
    references/intermediates/animation_sequence/
        <style>/svg/*.json                      -> <id>/reference/seq/<style>_<id>_svg.json
    references/rasters/                         -> <id>/reference/rasters/

`methods.tex` is deliberately renamed from the bundle's `method.tex`: the
pipeline reads the plural, and a silently missing methods file degrades context
rather than failing, which is the kind of gap that is only noticed in results.

The seq filename convention `<style>_<id>_svg.json` is what
`scripts/bench_v3_styles.py` probes to decide which (sample, style) cells are
MEASURABLE. Get it wrong and every cell reports as unmeasurable -- or worse,
a cell with no reference still writes a record whose GT fields are all null,
which is indistinguishable from a bad score.
"""
from __future__ import annotations

import argparse, json, shutil, sys
from pathlib import Path

STYLES = ("progressive_reveal", "hopping_bounding_box", "sliding_bounding_box",
          "colour_pop", "alpha_masking")


def import_one(src: Path, dest_root: Path, target: str = "svg") -> dict:
    sid = src.name
    out = dest_root / sid
    (out / "reference").mkdir(parents=True, exist_ok=True)

    stat = {"id": sid, "image": False, "context": 0, "xml": 0, "diagram": 0,
            "seq": 0, "rasters": 0, "styles": []}

    img = src / f"{sid}.png"
    if img.exists():
        shutil.copy2(img, out / f"{sid}.png"); stat["image"] = True

    for suffix, target_name in (("_title.txt", "title.txt"),
                                ("_abstract.tex", "abstract.tex"),
                                ("_method.tex", "methods.tex"),
                                ("_caption.tex", "caption.tex")):
        f = src / f"{sid}{suffix}"
        if f.exists():
            shutil.copy2(f, out / target_name); stat["context"] += 1

    refs = src / "references"
    if not refs.is_dir():
        return stat

    xml = refs / "intermediates" / "xml" / target
    if xml.is_dir():
        for f in xml.glob("*.xml"):
            d = out / "reference" / "xml"; d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, d / f"{sid}_{target}.xml"); stat["xml"] += 1

    dc = refs / "diagram_codes" / target
    if dc.is_dir():
        # Top level only: versions/ holds V1_/V2_ history, and copying those in
        # would put several files where the pipeline expects exactly one.
        for f in dc.glob("*.html"):
            d = out / "reference" / "diagram"; d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, d / f"{sid}_diag_{target}.html"); stat["diagram"] += 1

    seq_root = refs / "intermediates" / "animation_sequence"
    for style in STYLES:
        sd = seq_root / style / target
        if not sd.is_dir():
            continue
        for f in sorted(sd.glob("*.json")):
            d = out / "reference" / "seq"; d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, d / f"{style}_{sid}_{target}.json")
            stat["seq"] += 1; stat["styles"].append(style)
            break        # exactly one reference sequence per (style, target)

    rasters = refs / "rasters"
    if rasters.is_dir():
        d = out / "reference" / "rasters"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(rasters, d)
        stat["rasters"] = len(list(d.glob("*.png")))

    meta = refs / ".benchmark_meta.json"
    if meta.exists():
        shutil.copy2(meta, out / "reference" / "index.json")
    return stat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--target", default="svg")
    args = ap.parse_args()

    # Bundles nest one level: Full_eval_dir_X/Full_eval_dir_X/<sample>/
    bundles = []
    for top in sorted(p for p in args.src.iterdir() if p.is_dir()):
        inner = next((d for d in sorted(top.iterdir()) if d.is_dir()), None)
        if inner is None:
            continue
        for s in sorted(p for p in inner.iterdir() if p.is_dir()):
            if (s / f"{s.name}.png").exists():
                bundles.append(s)
    if not bundles:
        raise SystemExit(f"no sample bundles under {args.src}")

    args.dest.mkdir(parents=True, exist_ok=True)
    seen, stats = {}, []
    for b in bundles:
        # Duplicate ids across bundles would silently overwrite each other.
        if b.name in seen:
            print(f"  SKIP duplicate {b.name} (already from {seen[b.name]})")
            continue
        seen[b.name] = b.parent.parent.name
        stats.append(import_one(b, args.dest, args.target))

    ok = [s for s in stats if s["image"] and s["seq"]]
    print(f"\nimported {len(stats)} sample(s) -> {args.dest}")
    print(f"  with image + >=1 reference sequence: {len(ok)}")
    print(f"  measurable (sample,style) cells:     {sum(s['seq'] for s in stats)}")
    print(f"  with xml:     {sum(1 for s in stats if s['xml'])}")
    print(f"  with diagram: {sum(1 for s in stats if s['diagram'])}")
    print(f"  with rasters: {sum(1 for s in stats if s['rasters'])}")
    bad = [s["id"] for s in stats if not (s["image"] and s["seq"])]
    if bad:
        print(f"  INCOMPLETE ({len(bad)}): {bad[:6]}{' ...' if len(bad) > 6 else ''}")


if __name__ == "__main__":
    main()
