"""Place the pre-computed Zeroshot Gemini 3.1 SVGs where the exporter expects them.

NO API CALLS. These animations were produced outside this repo; this script only
unzips them and copies each into the cache directory `CachePaths` computes for
the zeroshot config, so `run_pipeline export` can rasterise them exactly like any
pipeline output.

WHY A RENAME IS THE WHOLE TRANSFORM. The shipped `.html` files are bare
`<svg xmlns=...>` roots with an inline `<style>` block -- not documents.
`export/svg_frames.render_svg_frames` wraps whatever it is given in its own
`_HTML_WRAPPER`, so handing it the file contents verbatim is correct. Rewriting
them into full HTML would double-wrap and change what is rendered.

STYLE IS progressive_reveal, MEASURED NOT ASSUMED: every file animates
`.reveal { opacity: 0; animation: fadeIn ... forwards }` over five staggered
delays -- cumulative, nothing re-hidden.

TWO FILES ARE SKIPPED. `CVPR_2025_pipe00011` and `CVPR_2025_pipe01224` contain
zero `@keyframes` -- they are static pictures. `render_svg_frames` handles a
zero-duration document by emitting exactly one frame, which `frames_to_mp4`
turns into a ~0.5s video that every video judge scores as a catastrophic
failure. That verdict would be right about the artifact and wrong about the
model, so they are excluded rather than scored.

TWO MORE HAVE NO BENCH SAMPLE. `CVPR_2025_pipe01224` (also static) and
`WACV_2022_set5_000006` exist nowhere in the dataset -- no source figure, no GT
reference, no XML -- so nothing could be generated or scored for them anyway.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from img_2_svg_pretraining.pipeline.cache import CachePaths
from img_2_svg_pretraining.pipeline.config import load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "src/img_2_svg_pretraining/pipeline/configs"
ZIP = REPO / "data/metrics/Zeroshot_Gemini_3.1_SVGs.zip"
STYLE = "progressive_reveal"

# Static -- no @keyframes. See module docstring.
SKIP_STATIC = {"CVPR_2025_pipe00011", "CVPR_2025_pipe01224"}
# No bench sample anywhere in data/. Nothing to compare against.
SKIP_NO_SAMPLE = {"CVPR_2025_pipe01224", "WACV_2022_set5_000006"}


def dataset_root_for(sample_id: str) -> Path | None:
    """v3 first, then v2 -- the 6 recovered samples only exist in v2."""
    for root in ("animatebench_v3", "animatebench_v2"):
        if (REPO / "data" / root / sample_id / "reference").is_dir():
            return REPO / "data" / root
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="bench_v3_zeroshot_gemini31.yaml")
    ap.add_argument("--only", nargs="*", help="restrict to these sample ids")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(tmp)
    svgs = sorted((tmp / "Zeroshot_Gemini_3.1_SVGs" / "SVGs").glob("*.html"))

    placed, skipped = [], []
    for src in svgs:
        sample = src.stem
        if args.only and sample not in args.only:
            continue
        if sample in SKIP_STATIC:
            skipped.append((sample, "static: no @keyframes")); continue
        if sample in SKIP_NO_SAMPLE:
            skipped.append((sample, "no bench sample in data/")); continue
        root = dataset_root_for(sample)
        if root is None:
            skipped.append((sample, "no bench sample in data/")); continue

        # The v2 samples need the v2 config: `CachePaths.root` is
        # `cache_root / <dataset dir name>`, so overriding only `dataset.root`
        # on the v3 config writes to the v3 cache tree while `run_pipeline`
        # (which cannot take a --dataset override) later reads the v2 one.
        # Two configs, two trees -- they must agree.
        cfg_name = ("bench_v2_zeroshot_gemini31.yaml"
                    if root.name == "animatebench_v2" else args.config)
        cfg = load_config(CONFIG_DIR / cfg_name)
        cfg.raw["dataset"]["root"] = str(root)
        cfg.style = STYLE
        cfg.raw["animation_style"] = STYLE
        paths = CachePaths.from_config(cfg)
        dst = paths.animation_final(sample)

        if args.dry_run:
            print(f"  would place {sample:30} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        placed.append((sample, root.name, dst))
        print(f"  placed {sample:30} ({root.name}) -> {dst.parent.name}/")

    for s, why in skipped:
        print(f"  SKIP   {s:30} {why}")
    print(f"\n{len(placed)} placed, {len(skipped)} skipped")
    if placed:
        print("\nNext: run the export stage (no API calls) per sample, e.g.")
        print(f"  run_pipeline export --config {args.config} --style {STYLE} --only <id>")


if __name__ == "__main__":
    main()
