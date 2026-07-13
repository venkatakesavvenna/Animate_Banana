"""Standalone debug-viewer script: dump every annotated image's confirmed
points + masks to disk as PNGs. No Streamlit, no SAM3 model load -- masks
are already stored as RLE in the annotation JSONs, so this is pure
image/JSON I/O.

For each image with at least one instance in a terminal state
(accepted/rejected/merged -- i.e. actually reviewed, not just a raw Molmo
proposal), writes to the debug folder:
    {image_id}_overlay.png   -- original image + every accepted instance's
                                 mask (distinct color per instance) + its
                                 confirmed points, plus rejected instances'
                                 points in gray for context
    {image_id}_mask.png      -- accepted instances only, flat label mask
                                 (0 = background, N = instance N's pixels),
                                 for anyone who wants raw arrays instead of
                                 a human-readable picture
A single summary.json is also written with per-image and totals stats.

Run (inside the docker container, main venv -- no SAM3/torch needed but the
venv has them anyway):

docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash -c "cd /code && python -m img_2_svg_pretraining.annotation_tool.export_debug"


Options: --annotations-dir, --out-dir (default data/debug/annotation_export),
--state {accepted,all} to control which instances count as "reviewed"
enough to render (default: accepted only -- rejected/merged are drawn for
context but never the reason a file gets written).
"""
from __future__ import annotations

import argparse
import json
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from img_2_svg_pretraining.annotation_tool import store
from img_2_svg_pretraining.annotation_tool.datamodel import Instance, rle_to_mask

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "debug" / "annotation_export"

POINT_RADIUS = 5
_REJECTED_PT = (140, 140, 140, 255)   # gray: rejected, shown for context only
_NEGATIVE_PT = (150, 40, 200, 255)    # purple: negative point
_POSITIVE_EDGE = (255, 255, 255, 255)


def _instance_color(i: int, n: int) -> tuple[int, int, int]:
    """Distinct, high-contrast RGB per instance index via evenly spaced hue."""
    hue = (i / max(n, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _overlay_mask(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int],
                  alpha: int = 100) -> None:
    if mask.shape != (base.height, base.width):
        return  # stale/foreign mask shape -- skip rather than crash the export
    layer = Image.new("RGBA", base.size, (*color, alpha))
    layer.putalpha(Image.fromarray((mask.astype(np.uint8) * alpha), mode="L"))
    base.alpha_composite(layer)


def render_overlay(image: Image.Image, instances: dict[str, Instance],
                   order: list[str], include_rejected: bool = True) -> Image.Image:
    """One PNG: original image + every accepted instance's mask (distinct
    color) + its confirmed points, plus rejected instances' points in gray
    so a reviewer can see what was screened out."""
    base = image.convert("RGBA")
    accepted = [instances[i] for i in order
               if i in instances and instances[i].state == "accepted"]

    for idx, inst in enumerate(accepted):
        if inst.mask_rle:
            color = _instance_color(idx, len(accepted))
            _overlay_mask(base, rle_to_mask(inst.mask_rle), color)

    draw = ImageDraw.Draw(base)
    for idx, inst in enumerate(accepted):
        color = (*_instance_color(idx, len(accepted)), 255)
        for p in inst.points:
            r = POINT_RADIUS
            fill = _NEGATIVE_PT if p.label == 0 else color
            draw.ellipse([p.x - r, p.y - r, p.x + r, p.y + r],
                        fill=fill, outline=_POSITIVE_EDGE, width=1)
        # small index label near the first point, for cross-referencing
        # against the flat label mask / summary.json
        if inst.points:
            p0 = inst.points[0]
            draw.text((p0.x + 7, p0.y - 7), str(idx + 1), fill=(255, 255, 0, 255))

    if include_rejected:
        for iid in order:
            inst = instances.get(iid)
            if inst is None or inst.state != "rejected":
                continue
            for p in inst.points:
                r = POINT_RADIUS - 1
                draw.ellipse([p.x - r, p.y - r, p.x + r, p.y + r],
                            outline=_REJECTED_PT, width=2)

    return base.convert("RGB")


def render_label_mask(image: Image.Image, instances: dict[str, Instance],
                      order: list[str]) -> np.ndarray:
    """uint16 HxW array, 0 = background, instance rank (1-indexed) elsewhere.
    Later-accepted instances draw on top where masks overlap."""
    label = np.zeros((image.height, image.width), dtype=np.uint16)
    rank = 0
    for iid in order:
        inst = instances.get(iid)
        if inst is None or inst.state != "accepted" or not inst.mask_rle:
            continue
        rank += 1
        mask = rle_to_mask(inst.mask_rle)
        if mask.shape != label.shape:
            rank -= 1
            continue
        label[mask] = rank
    return label


def export_image(image_id: str, out_dir: Path, include_rejected: bool) -> dict | None:
    loaded = store.load_image_record(image_id)
    if loaded is None:
        return None
    record, instances = loaded
    n_accepted = sum(1 for i in instances.values() if i.state == "accepted")
    n_rejected = sum(1 for i in instances.values() if i.state == "rejected")
    n_merged = sum(1 for i in instances.values() if i.state == "merged")
    n_unresolved = sum(1 for i in instances.values()
                       if i.state in ("proposed", "needs_point_review",
                                      "needs_mask_review"))
    if n_accepted + n_rejected + n_merged == 0:
        return None  # nothing reviewed yet -- skip, not part of the export

    image_path = Path(record.file_path)
    if not image_path.is_absolute():
        image_path = _REPO_ROOT / image_path
    image = Image.open(image_path).convert("RGB")

    overlay = render_overlay(image, instances, record.instances,
                             include_rejected=include_rejected)
    overlay.save(out_dir / f"{image_id}_overlay.png")

    label_mask = render_label_mask(image, instances, record.instances)
    if label_mask.max() > 0:
        # scale into 0-255 for a viewable PNG; still recoverable since ranks
        # are small integers and evenly spaced by 255/max
        scaled = (label_mask.astype(np.float32) * (255.0 / label_mask.max())
                 ).astype(np.uint8)
        Image.fromarray(scaled, mode="L").save(out_dir / f"{image_id}_mask.png")

    return {
        "image_id": image_id,
        "width": record.width, "height": record.height,
        "accepted": n_accepted, "rejected": n_rejected, "merged": n_merged,
        "unresolved": n_unresolved,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations-dir", type=Path, default=None,
                    help="defaults to ANNOTATIONS_DIR env var or data/annotations")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--image-ids", nargs="*", default=None,
                    help="explicit image ids instead of every annotated image")
    ap.add_argument("--no-rejected", action="store_true",
                    help="don't draw rejected instances' points for context")
    args = ap.parse_args()

    if args.annotations_dir is not None:
        import os
        os.environ["ANNOTATIONS_DIR"] = str(args.annotations_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    image_ids = args.image_ids or store.list_annotated_image_ids()
    results = []
    for i, image_id in enumerate(image_ids):
        stats = export_image(image_id, args.out_dir,
                             include_rejected=not args.no_rejected)
        if stats is not None:
            results.append(stats)
            print(f"[{i + 1}/{len(image_ids)}] {image_id}: "
                  f"{stats['accepted']} accepted, {stats['rejected']} rejected, "
                  f"{stats['merged']} merged, {stats['unresolved']} unresolved")

    totals = {
        "images_exported": len(results),
        "images_scanned": len(image_ids),
        "total_accepted": sum(r["accepted"] for r in results),
        "total_rejected": sum(r["rejected"] for r in results),
        "total_merged": sum(r["merged"] for r in results),
        "total_unresolved": sum(r["unresolved"] for r in results),
    }
    summary = {"totals": totals, "images": results}
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\ndone: {totals['images_exported']} / {totals['images_scanned']} "
          f"annotated images had reviewed instances -> {args.out_dir}")
    print(f"totals: {totals['total_accepted']} accepted, "
          f"{totals['total_rejected']} rejected, "
          f"{totals['total_merged']} merged, "
          f"{totals['total_unresolved']} still unresolved")


if __name__ == "__main__":
    main()
