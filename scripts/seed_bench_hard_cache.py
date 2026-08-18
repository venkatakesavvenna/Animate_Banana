"""Seed the bench-hard debug cache with the bundle's halfway-corrected codes.

The 24-sample-bench-hard bundles ship a corrected TikZ document and a
corrected SVG per sample under `reference/diagram/` after import. The
annotation tool only reads stage-1a code from `CachePaths.code()`, so this
copies each into the right lineage directory for *both* targets -- which is
exactly the setup the target-crossover bugs need to reproduce.

Idempotent: re-running overwrites the seeded 1a with the bundle version but
touches nothing else (no code_final/code_reviewed/human dirs).

Run from the repo root:
    PYTHONPATH=src python scripts/seed_bench_hard_cache.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from img_2_svg_pretraining.pipeline.cache import CachePaths  # noqa: E402
from img_2_svg_pretraining.pipeline.config import load_config  # noqa: E402

CONFIG = REPO / "src/img_2_svg_pretraining/pipeline/configs/annotate_bench_hard.yaml"


def main() -> None:
    cfg = load_config(str(CONFIG))
    dataset = Path(cfg.dataset_root)
    seeded = {"tikz": 0, "svg": 0}

    for sample_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
        sid = sample_dir.name
        diagram = sample_dir / "reference" / "diagram"
        if not diagram.is_dir():
            continue

        # (target, source file) pairs; the bundle's SVG ships as .html but is
        # a bare <svg> document, so it is the .svg artifact verbatim.
        sources = {
            "tikz": diagram / f"{sid}_diag_tikz.tex",
            "svg": diagram / f"{sid}_diag_svg.html",
        }
        for target, src in sources.items():
            if not src.is_file():
                continue
            cfg.target = target
            cfg.raw["target"] = target
            out = CachePaths.from_config(cfg).code(sid)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            seeded[target] += 1

    # Restore for clarity if this ever grows more steps.
    cfg.target = "tikz"
    cfg.raw["target"] = "tikz"
    print(f"seeded 1a code: {seeded['tikz']} tikz, {seeded['svg']} svg "
          f"-> {CachePaths.from_config(cfg).root}")


if __name__ == "__main__":
    main()
