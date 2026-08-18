"""Pull the presentable failure cases out of a bench run.

The primary table says how often the pipeline is right. A demo needs the other
half: specific, legible cases where it is wrong, with the artifact that shows
it. This ranks scored cells by how badly and how *visibly* they failed, and
prints the file you would put on the slide next to each one.

Ordering is by severity class first, then by the metric inside the class, so a
diagram that does not compile always outranks one that merely renders poorly:

    1  stage-1 compile failure     nothing downstream is meaningful
    2  stage-3 compile failure     the animation does not exist
    3  low rendering fidelity      it compiled and drew the wrong picture
    4  structural errors           XML got the hierarchy or the edges wrong
    5  sequence-rule violations    the animation order breaks its own style
    6  coverage loss               elements the reference animates, we drop

Judge prose is carried through verbatim where a metric has it. A number on a
slide invites the question "says who"; the judge's own sentence answers it.

    python scripts/bench_v3_failures.py                 # ranked, all classes
    python scripts/bench_v3_failures.py --top 8
    python scripts/bench_v3_failures.py --class compile
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EVALS = REPO / "data" / "animatebench_v3_cache" / "animatebench_v3" / "evals"
DEFAULT_DATA = REPO / "data" / "animatebench_v3"

# Judged metrics whose prose is worth quoting on a slide.
NOTES = {"rendering_fidelity": "rendering_fidelity_notes",
         "aif": "aif_note"}


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def cells(evals: Path) -> list[dict]:
    """One dict per (config, style, sample), merging its four suite records."""
    out = []
    for sample_dir in sorted(evals.glob("*/*/*")):
        if not sample_dir.is_dir() or sample_dir.parent.parent.name in (
                "alignment", "checklist", "raw", "_frames"):
            continue
        cell = {"config": sample_dir.parent.parent.name,
                "style": sample_dir.parent.name, "sample": sample_dir.name}
        found = False
        for suite in ("stage1", "xml", "sequence", "stage3"):
            record = _load(sample_dir / f"{suite}.json")
            if record:
                found = True
                cell[suite] = record
        if found:
            out.append(cell)
    return out


def findings(cell: dict) -> list[dict]:
    """Every presentable defect in one cell, each with its severity class."""
    s1 = cell.get("stage1", {})
    s3 = cell.get("stage3", {})
    xml = cell.get("xml", {})
    seq = cell.get("sequence", {})
    out = []

    def add(rank, cls, headline, **extra):
        out.append({"rank": rank, "class": cls, "headline": headline, **extra})

    if s1.get("compiles") is False:
        add(1, "compile", "diagram code does not compile",
            detail=(s1.get("compile_log") or "")[-400:])
    if s3 and s3.get("anim_compiles") is False:
        add(2, "compile", "animation code does not compile",
            detail=(s3.get("anim_compile_log") or "")[-400:],
            note="repair attempted and failed" if s3.get("raw_compiles") is False
                 else "compiled before the critic touched it")

    fidelity = s1.get("rendering_fidelity")
    if isinstance(fidelity, (int, float)) and fidelity < 0.7:
        worst = ""
        breakdown = s1.get("rendering_fidelity_breakdown") or {}
        if breakdown:
            axis, value = min(breakdown.items(), key=lambda kv: kv[1])
            worst = f"weakest axis: {axis} {value}"
        add(3, "fidelity", f"rendering fidelity {fidelity:.2f}",
            detail=s1.get("rendering_fidelity_notes", ""), note=worst,
            score=fidelity)

    paa = xml.get("paa")
    if isinstance(paa, (int, float)) and paa < 0.7:
        add(4, "structure", f"parent accuracy {paa:.2f}", score=paa,
            note=f"depth violations {xml.get('depth_violation_rate', 0):.2f}")
    if xml.get("missed_gt_edges"):
        missed = xml["missed_gt_edges"]
        add(4, "structure", f"{len(missed)} reference edge(s) missing",
            detail=", ".join("->".join(e) for e in missed[:6]))

    if seq.get("sscr_pass") is False:
        add(5, "sequence", "style rules violated",
            detail="; ".join(str(v) for v in (seq.get("sscr_violations") or [])[:4]))
    dovr = seq.get("dovr")
    if isinstance(dovr, (int, float)) and dovr > 0:
        add(5, "sequence", f"depth-order violations {dovr:.2f}",
            detail="; ".join(str(v) for v in (seq.get("dovr_violations") or [])[:4]),
            score=dovr)

    recall = seq.get("coverage_recall")
    if isinstance(recall, (int, float)) and recall < 0.9:
        add(6, "coverage", f"coverage recall {recall:.2f}", score=recall,
            detail=", ".join(seq.get("missed_groups") or [])[:300])

    for suite in ("stage1", "xml", "sequence", "stage3"):
        err = (cell.get(suite) or {}).get("error")
        if err:
            add(1, "error", f"{suite} did not score", detail=err)
    return out


def artifacts(cell: dict, data_root: Path, evals: Path) -> dict:
    """The files worth putting on a slide beside this cell."""
    sample = cell["sample"]
    out = {"figure": data_root / sample / f"{sample}.png"}
    render = (cell.get("stage1") or {}).get("render_path")
    if render:
        # Records written inside the container carry /code paths.
        out["our_render"] = Path(str(render).replace("/code/", str(REPO) + "/"))
    ref = data_root / sample / "reference" / "videos" / f"tikz__{cell['style']}__full.mp4"
    if ref.exists():
        out["reference_video"] = ref
    return {k: v for k, v in out.items() if v and Path(v).exists()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals", type=Path, default=DEFAULT_EVALS)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--top", type=int, default=0, help="0 = all")
    ap.add_argument("--class", dest="cls", help="only this severity class")
    ap.add_argument("--json", type=Path, help="also write the findings as JSON")
    args = ap.parse_args()

    if not args.evals.is_dir():
        raise SystemExit(f"no evals under {args.evals}")

    rows = []
    for cell in cells(args.evals):
        for finding in findings(cell):
            if args.cls and finding["class"] != args.cls:
                continue
            rows.append({**finding, "cell": cell})
    rows.sort(key=lambda r: (r["rank"], r.get("score", 0.0)))
    if args.top:
        rows = rows[:args.top]

    if not rows:
        print("no failures found (either nothing scored yet, or nothing failed)")
        return

    print(f"{len(rows)} presentable failure(s)\n")
    for i, row in enumerate(rows, 1):
        cell = row["cell"]
        print(f"{i:2d}. [{row['class']}] {cell['sample']} / {cell['style']}")
        print(f"    {row['headline']}")
        if row.get("note"):
            print(f"    {row['note']}")
        if row.get("detail"):
            detail = " ".join(str(row["detail"]).split())
            print(f"    \"{detail[:320]}\"")
        for name, path in artifacts(cell, args.data, args.evals).items():
            print(f"    {name}: {path}")
        print()

    if args.json:
        args.json.write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "cell"} | {
                "sample": r["cell"]["sample"], "style": r["cell"]["style"]}
             for r in rows], indent=2), encoding="utf-8")
        print(f"written: {args.json}")


if __name__ == "__main__":
    main()
