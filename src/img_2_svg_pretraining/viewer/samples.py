"""Discovers (input image, TikZ source) sample pairs from a data directory.

Supports two layouts:

1. The real Stage-1 dataset (data/Set-2/...), which has a flat, ID-keyed
   layout:
     tex_files/<id>.tex            ground-truth TikZ source
     original_images/<id>.png      ground-truth input diagram (the Stage-1 input)
     tex_images/<id>.png           pre-rendered PNG of the compiled ground-truth TikZ
     tex_pdfs/<id>.pdf             pre-rendered PDF of the compiled ground-truth TikZ
     failure_cases/<id>.{log,aux}  latexmk logs for ids that failed to compile
   IDs look like "<VENUE>_<YEAR>_<kind><NNNNN>", e.g. CVPR_2025_arch00033,
   WACV_2026_pipe00042. kind is "arch" (architecture diagram) or "pipe"
   (pipeline diagram). Not every id has a tex_images/tex_pdfs pair -- those
   are the ones in failure_cases (bad TikZ that needs correcting).

2. A generic/legacy layout (examples/animation/...), permissive fallback:
   any directory containing .tex files, each paired with a best-guess
   ground-truth image via naming heuristics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
PREFERRED_IMAGE_NAMES = ["input", "image", "gt", "ground_truth", "original", "target"]

SAMPLE_ID_RE = re.compile(r"^(?P<venue>[A-Za-z]+)_(?P<year>\d{4})_(?P<kind>arch|pipe)(?P<num>\d+)$")


@dataclass
class Sample:
    id: str
    tex_path: Path
    image_path: Path | None
    rendered_image_path: Path | None = None
    rendered_pdf_path: Path | None = None
    failed_compile: bool = False
    venue: str | None = None
    year: str | None = None
    kind: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tex_path": str(self.tex_path),
            "image_path": str(self.image_path) if self.image_path else None,
            "has_rendered": self.rendered_image_path is not None,
            "failed_compile": self.failed_compile,
            "venue": self.venue,
            "year": self.year,
            "kind": self.kind,
        }


def _find_ground_truth_image(tex_path: Path) -> Path | None:
    directory = tex_path.parent

    for ext in IMAGE_EXTS:
        candidate = directory / f"{tex_path.stem}{ext}"
        if candidate.exists():
            return candidate

    for name in PREFERRED_IMAGE_NAMES:
        for ext in IMAGE_EXTS:
            candidate = directory / f"{name}{ext}"
            if candidate.exists():
                return candidate

    images = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS)
    if images:
        return images[0]

    crops_dir = directory / "crops"
    if crops_dir.is_dir():
        nested = sorted(crops_dir.rglob("*"))
        images = [p for p in nested if p.is_file() and p.suffix in IMAGE_EXTS]
        if images:
            return images[0]

    return None


def _discover_set2_layout(root: Path) -> list[Sample] | None:
    """Detect and load the ID-keyed Set-2 dataset layout. Returns None if root
    doesn't look like this layout (caller should fall back to generic scan)."""
    tex_dir = root / "tex_files"
    image_dir = root / "original_images"
    if not tex_dir.is_dir() or not image_dir.is_dir():
        return None

    rendered_dir = root / "tex_images"
    pdf_dir = root / "tex_pdfs"
    failure_dir = root / "failure_cases"

    failed_ids = set()
    if failure_dir.is_dir():
        failed_ids = {p.stem for p in failure_dir.glob("*.log")}

    samples: list[Sample] = []
    for tex_path in sorted(tex_dir.glob("*.tex")):
        sample_id = tex_path.stem
        image_path = image_dir / f"{sample_id}.png"
        if not image_path.exists():
            image_path = None

        rendered_image_path = rendered_dir / f"{sample_id}.png"
        if not rendered_image_path.exists():
            rendered_image_path = None

        rendered_pdf_path = pdf_dir / f"{sample_id}.pdf"
        if not rendered_pdf_path.exists():
            rendered_pdf_path = None

        m = SAMPLE_ID_RE.match(sample_id)
        venue, year, kind = (m.group("venue"), m.group("year"), m.group("kind")) if m else (None, None, None)

        samples.append(Sample(
            id=sample_id,
            tex_path=tex_path,
            image_path=image_path,
            rendered_image_path=rendered_image_path,
            rendered_pdf_path=rendered_pdf_path,
            failed_compile=sample_id in failed_ids,
            venue=venue,
            year=year,
            kind=kind,
        ))

    return samples


def _discover_generic_layout(root: Path) -> list[Sample]:
    """Recursively find sample directories under root, one per .tex file found."""
    samples: list[Sample] = []
    tex_files = sorted(root.rglob("*.tex"))
    for tex_path in tex_files:
        image_path = _find_ground_truth_image(tex_path)
        rel = tex_path.relative_to(root)
        sample_id = str(rel.with_suffix(""))
        samples.append(Sample(id=sample_id, tex_path=tex_path, image_path=image_path))
    return samples


def discover_samples(root: Path) -> list[Sample]:
    root = Path(root)
    if not root.exists():
        return []

    set2_samples = _discover_set2_layout(root)
    if set2_samples is not None:
        return set2_samples

    return _discover_generic_layout(root)
