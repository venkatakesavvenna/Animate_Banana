"""Seed one cache_root's response cache from another, by hardlink.

THE WHOLE STAGE-REUSE STRATEGY LIVES HERE, so it is worth stating plainly.

Each ablation config gets its own cache_root (the eight share one CachePaths
lineage, so shared artifact trees would collide). That isolation means a naive
run pays for every stage of every config -- including stages whose inputs are
byte-identical to the baseline's, where a fresh call at temperature > 0 would
produce a DIFFERENT but equally valid output, silently breaking the ablation's
"everything else held fixed" premise.

Instead of copying artifacts (and hand-maintaining a table of which stages each
ablation may reuse -- one wrong entry silently invalidates a row), we seed the
RESPONSE cache. The backend fingerprints every call on (model, messages,
params); a stage whose input is unchanged reproduces the baseline's exact
output for $0, and a stage the ablation actually affects -- changed prompt,
withheld image, different upstream artifact -- misses the cache and pays. The
fingerprint decides what is reusable, mechanically, per call.

Measured proof of both directions: a pilot run WITHOUT seeding shared exactly 1
fingerprint with round one (the code converter -- identical image+prompt) while
its other 9 calls forked, because the unseeded converter call re-sampled at
temperature 0.2 and every downstream input changed with it. Seed first and that
fork never happens.

Hardlinks, not copies: both caches sit on one filesystem, the entries are
immutable once written (writes go through tmp+atomic-rename, which replaces the
link rather than mutating the shared inode), and 91 samples of 65k-token
responses are worth not duplicating. --no-clobber: an entry the destination
already has is never overwritten.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def seed(src_root: Path, dst_root: Path) -> tuple[int, int, int]:
    linked = skipped = missing = 0
    if not src_root.is_dir():
        print(f"  !! source has no responses dir: {src_root}")
        return 0, 0, 1
    for backend_dir in sorted(src_root.iterdir()):
        if not backend_dir.is_dir():
            continue
        out = dst_root / backend_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for f in backend_dir.glob("*.json"):
            dst = out / f.name
            if dst.exists():
                skipped += 1
                continue
            try:
                os.link(f, dst)
            except OSError:
                shutil.copy2(f, dst)        # cross-device fallback
            linked += 1
    return linked, skipped, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="source cache_root (contains <dataset>/responses)")
    ap.add_argument("dst", help="destination cache_root")
    ap.add_argument("--dataset", default="animatebench_v5")
    args = ap.parse_args()

    src = Path(args.src) / args.dataset / "responses"
    dst = Path(args.dst) / args.dataset / "responses"
    linked, skipped, missing = seed(src, dst)
    print(f"  seeded {Path(args.dst).name}: +{linked} linked, "
          f"{skipped} already present")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
