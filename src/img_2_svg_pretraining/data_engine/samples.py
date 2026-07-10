"""Sample discovery for `data/partial_test_set`, kept dependency-light (no
torch/transformers/cairosvg imports) so single-stage debug scripts
(run_pointing.py) don't have to import the rest of the pipeline's heavier
stage modules just to list samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataEngineSample:
    id: str
    image_path: Path
    gt_xml_path: Path | None = None
    gt_tex_path: Path | None = None


def discover_images(root: Path, image_subdir: str = "images") -> list[DataEngineSample]:
    """Generic discovery: lists every `<root>/<image_subdir>/*.png` and pairs
    each with an optional ground-truth tex (`tex_files/<id>.tex`) and layout
    xml (`xml_files/layout_<id>.xml`) if present. Works for any dataset that
    keeps its images in a single flat subdir -- e.g. `data/partial_test_set`
    (image_subdir="images") or `data/train` (image_subdir="original_images").
    Ground-truth tex/xml are only used as comparison targets and are not
    required for the pipeline stages to run."""
    root = Path(root)
    image_dir, tex_dir, xml_dir = root / image_subdir, root / "tex_files", root / "xml_files"
    samples = []
    for image_path in sorted(image_dir.glob("*.png")):
        sample_id = image_path.stem
        tex_path = tex_dir / f"{sample_id}.tex"
        xml_path = xml_dir / f"layout_{sample_id}.xml"
        samples.append(DataEngineSample(
            id=sample_id,
            image_path=image_path,
            gt_xml_path=xml_path if xml_path.exists() else None,
            gt_tex_path=tex_path if tex_path.exists() else None,
        ))
    return samples


def discover_partial_test_set(root: Path) -> list[DataEngineSample]:
    """Loads the `data/partial_test_set` layout (images/<id>.png). Thin
    wrapper over `discover_images` kept for backward compatibility with
    existing callers."""
    return discover_images(root, image_subdir="images")
