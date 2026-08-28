"""Extract keyframe decks for every ground-truth reference animation.

The 64 reference videos under `<sample>/reference/videos/` ship with no frame
decks, so nothing in the animation tree can score them per frame. This builds
those decks.

Output goes under the EVALS root, not into `data/animatebench_v3/`: the dataset
tree is an imported bundle (pipeline/import_bench.py) and staying byte-identical
keeps re-imports and `--dataset` overrides safe.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from img_2_svg_pretraining.animatebench import keyframes as kf  # noqa: E402

OUT = REPO / "data/animatebench_v3_cache/animatebench_v3/evals/_gt_frames"
DATASET = REPO / "data/animatebench_v3"


def main() -> None:
    vids = sorted(DATASET.glob("*/reference/videos/*__*__full.mp4"))
    print(f"{len(vids)} reference video(s)\n")
    print(f"{'target':6} {'style':22} {'sample':30} {'src':>4} {'kept':>5} "
          f"{'dwell':>6}  method")
    methods = Counter()
    saved = 0
    for v in vids:
        target, style, _ = v.stem.split("__")
        sample = v.parents[2].name
        try:
            r = kf.extract(v, OUT / target / style / sample, style)
        except Exception as e:                       # noqa: BLE001
            print(f"{target:6} {style:22} {sample[:30]:30} FAILED "
                  f"{type(e).__name__}: {e}")
            continue
        methods[r.method] += 1
        saved += r.source_frame_count - len(r.paths)
        print(f"{target:6} {style:22} {sample[:30]:30} {r.source_frame_count:>4} "
              f"{len(r.paths):>5} {r.params.get('dwell_fraction'):>6}  {r.method}")

    print("\nmethod chosen:")
    for m, n in methods.most_common():
        print(f"  {n:>3}  {m}")
    print(f"\nframes dropped by extraction across the corpus: {saved}")


if __name__ == "__main__":
    main()
