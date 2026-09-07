"""Seed stage-1 ARTIFACTS from one cache_root into another.

WHY RESPONSE SEEDING ALONE WAS NOT ENOUGH -- measured, not theorised. The
raster integrator writes ABSOLUTE paths into the SVG it produces
(`<image href="/fsx.../<cache_root>/rasters/...">`). Every ablation has its own
cache_root, so a freshly-integrated SVG in each ablation embeds different path
strings -- and every downstream request that carries the code text (parse,
sequence, critic, designer...) therefore hashes differently and misses the
response cache even when nothing meaningful differs. Observed: ablations that
diverge only at narration still paid for everything from parse onward
(10 fresh calls where ~4 belong).

Seeding the stage-1 artifact family fixes the identity at its root: the
ablation carries the baseline's byte-identical code -- same embedded paths,
same downstream request text, same fingerprints -- so cache hits reach all the
way to the ablation's true divergence point. The referenced raster files live
in the SOURCE cache and remain readable there; nothing rewrites them.

COPIES, NOT HARDLINKS, unlike the response seeder. Response entries are
immutable by construction (tmp + atomic rename). Artifacts are not: a stage
re-run with --force rewrites its output file, and through a hardlink that
write would corrupt the source cache's copy silently. `cp -p` preserves
mtimes, which the freshness machinery compares (this repo's own rule: compare
mtimes, not existence).

WHAT IS SEEDED, deliberately minimal:
  code/           converter output      identical for all eight configs
  rasters/        crops + detections    identical for all eight
  code_final/     raster-spliced code   identical for all eight
  code_reviewed/  stage-1 critic output identical for the seven configs whose
                  stage-1 critic is ON; --skip code_reviewed excludes it for
                  stage1_no_critic, where its presence would make resolve_code
                  serve the critic's output in the very ablation that removes
                  the critic.
Nothing deeper. Deeper artifacts genuinely differ per ablation, and the
response cache regenerates identical ones for $0 once the text identity holds.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

FAMILIES = ("code", "rasters", "code_final", "code_reviewed")


def seed_tree(src: Path, dst: Path) -> tuple[int, int]:
    copied = skipped = 0
    if not src.is_dir():
        return 0, 0
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        out = dst / rel
        if f.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        if out.exists():
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        copied += 1
    return copied, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--dataset", default="animatebench_v5")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="artifact families to exclude (e.g. code_reviewed)")
    args = ap.parse_args()

    bad = set(args.skip) - set(FAMILIES)
    if bad:
        sys.exit(f"unknown --skip families: {sorted(bad)}")

    for fam in FAMILIES:
        if fam in args.skip:
            print(f"  {fam:14} skipped by request")
            continue
        c, s = seed_tree(Path(args.src) / args.dataset / fam,
                         Path(args.dst) / args.dataset / fam)
        print(f"  {fam:14} +{c} copied, {s} already present")


if __name__ == "__main__":
    main()
