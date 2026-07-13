"""Offline Molmo ingest: image -> proposed instances (spec section 13).

Runs Molmo pointing over a batch of images and writes one annotation JSON
per image with every point as a `proposed` instance (source="molmo", no mask
-- masks are computed live in the review app, where a human confirms them).

This tool is scoped to RASTER regions only (embedded photos/icons/plots/
screenshots), not structural diagram nodes -- uses `point_rasters()`
(`RASTER_QUERY`), never `point_nodes()`. Node annotation is a separate,
not-yet-built concern; see data_engine/pointing/common.py's docstring for
the query text.

This is deliberately NOT part of the review app: re-running Molmo with a new
checkpoint means re-running this script against affected images, never a UI
button. By default images that already have an annotation JSON are skipped
so human work is never clobbered; --force overwrites (only sensible before
review has started on those images).

Must run in the dedicated Molmo venv (transformers==4.57.1 -- Molmo remote
code breaks under the main venv's transformers; see
data_engine/pointing/molmo_point.py), inside the container:

    docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash -c \\
      "source /environments/molmo_point/bin/activate && cd /code && \\
       python -m img_2_svg_pretraining.annotation_tool.ingest \\
         --images-dir data/train/original_images --limit 20"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from img_2_svg_pretraining.annotation_tool import store
from img_2_svg_pretraining.annotation_tool.datamodel import (
    ImageRecord, SessionStats, make_proposed_instance,
)
from img_2_svg_pretraining.data_engine.pointing import make_point_runner
from img_2_svg_pretraining.data_engine.pointing.models import get_pointing_model

_REPO_ROOT = Path(__file__).resolve().parents[3]


def ingest_image(runner, image_path: Path) -> int:
    """Molmo raster points -> one proposed instance per point ->
    {image_id}.json. Returns the number of proposed instances written."""
    image_id = image_path.stem
    img = Image.open(image_path)
    points = runner.point_rasters(image_path)

    instances = {}
    for p in points:
        inst = make_proposed_instance(image_id, p.x, p.y)
        instances[inst.id] = inst
    record = ImageRecord(
        id=image_id, file_path=str(image_path),
        width=img.width, height=img.height,
        instances=list(instances), session_stats=SessionStats(),
    )
    store.save_image_record(record, instances)
    return len(instances)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images-dir", type=Path,
                    default=_REPO_ROOT / "data" / "train" / "original_images")
    ap.add_argument("--pointing-model", default="molmo2-8b",
                    choices=["molmo2-8b", "molmo-point-8b"])
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N images (sorted)")
    ap.add_argument("--image-ids", nargs="*", default=None,
                    help="explicit image ids (stems) instead of the whole dir")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing annotation JSONs (drops any "
                         "human review already recorded in them!)")
    args = ap.parse_args()

    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in args.images_dir.iterdir()
                   if p.suffix.lower() in exts)
    if args.image_ids:
        wanted = set(args.image_ids)
        paths = [p for p in paths if p.stem in wanted]
    if args.limit:
        paths = paths[: args.limit]

    existing = set(store.list_annotated_image_ids())
    skipped = [p for p in paths if p.stem in existing and not args.force]
    paths = [p for p in paths if p.stem not in existing or args.force]
    if skipped:
        print(f"skipping {len(skipped)} already-annotated images "
              f"(--force to overwrite)")
    if not paths:
        print("nothing to do")
        return

    runner = make_point_runner(get_pointing_model(args.pointing_model))
    total = 0
    for i, path in enumerate(paths):
        n = ingest_image(runner, path)
        total += n
        print(f"[{i + 1}/{len(paths)}] {path.stem}: {n} proposed instances")
    print(f"done: {total} proposed instances over {len(paths)} images "
          f"-> {store.annotations_dir()}")


if __name__ == "__main__":
    main()
