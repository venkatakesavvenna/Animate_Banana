"""Which samples of a bench have a reference sequence for which style.

v3 is not a full grid. Four of its 32 samples ship all five reference styles;
the other 28 ship exactly one. Running a sample in a style it has no reference
for still writes a `sequence` record -- with every GT-derived field null. That
is indistinguishable at a glance from a sample that scored badly, so the run is
restricted to the cells that can actually be measured instead.

    python scripts/bench_v3_styles.py                     # the coverage table
    python scripts/bench_v3_styles.py --style colour_pop  # ids, space separated

The second form is what feeds `run_pipeline --only` and `run_eval --only`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "animatebench_v3"
STYLES = ("progressive_reveal", "alpha_masking", "colour_pop",
          "hopping_bounding_box", "sliding_bounding_box")


def coverage(root: Path, target: str = "tikz") -> dict[str, list[str]]:
    """style -> sample ids holding a reference sequence for it."""
    out: dict[str, list[str]] = {s: [] for s in STYLES}
    for sample in sorted(p for p in root.iterdir() if p.is_dir()):
        for style in STYLES:
            if (sample / "reference" / "seq"
                    / f"{style}_{sample.name}_{target}.json").exists():
                out[style].append(sample.name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--target", default="tikz", choices=("tikz", "svg"))
    ap.add_argument("--style", choices=STYLES,
                    help="print just this style's ids, space separated")
    args = ap.parse_args()

    cov = coverage(args.root, args.target)
    if args.style:
        print(" ".join(cov[args.style]))
        return

    samples = sorted({s for ids in cov.values() for s in ids})
    width = max(len(s) for s in samples) + 1
    print(f"{'sample':{width}s}" + "".join(f"{s[:4]:>6s}" for s in STYLES))
    for sample in samples:
        row = "".join(f"{'  Y' if sample in cov[s] else '  .':>6s}" for s in STYLES)
        print(f"{sample:{width}s}{row}")
    print()
    for style in STYLES:
        print(f"{style:24s} {len(cov[style]):3d}")
    cells = sum(len(v) for v in cov.values())
    print(f"\n{len(samples)} sample(s), {cells} measurable (sample, style) cell(s)")
    # Styles in descending size: running them in this order covers the most
    # distinct samples soonest, which is the point when the run may be cut off.
    seen: set[str] = set()
    print("\nrun order (new samples each style adds):")
    for style in sorted(STYLES, key=lambda s: -len(cov[s])):
        new = [s for s in cov[style] if s not in seen]
        seen.update(new)
        print(f"  {style:24s} {len(cov[style]):3d} cells, {len(new):3d} new "
              f"-> {len(seen)}/{len(samples)} samples covered")


if __name__ == "__main__":
    main()
