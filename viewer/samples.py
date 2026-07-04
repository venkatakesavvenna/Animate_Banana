"""Discovers (input image, TikZ source) sample pairs from a data directory.

Layout is intentionally permissive since the real Stage-1 dataset
(data/Set-2-001.zip) has not been unpacked/inspected yet. A "sample" is any
directory containing exactly one primary .tex file; the paired ground-truth
image is resolved heuristically so the same scanner works for the current
examples/ folder and the eventual Set-2 dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
PREFERRED_IMAGE_NAMES = ["input", "image", "gt", "ground_truth", "original", "target"]


@dataclass
class Sample:
    id: str
    tex_path: Path
    image_path: Path | None
    extra_tex_paths: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tex_path": str(self.tex_path),
            "image_path": str(self.image_path) if self.image_path else None,
            "extra_tex_paths": [str(p) for p in self.extra_tex_paths],
        }


def _find_ground_truth_image(tex_path: Path) -> Path | None:
    directory = tex_path.parent

    # Same basename as the .tex file, any image extension.
    for ext in IMAGE_EXTS:
        candidate = directory / f"{tex_path.stem}{ext}"
        if candidate.exists():
            return candidate

    # Common ground-truth filenames directly beside the .tex file.
    for name in PREFERRED_IMAGE_NAMES:
        for ext in IMAGE_EXTS:
            candidate = directory / f"{name}{ext}"
            if candidate.exists():
                return candidate

    # Fall back to any image file directly in the sample directory.
    images = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS)
    if images:
        return images[0]

    # Fall back further into a conventional crops/ subdirectory.
    crops_dir = directory / "crops"
    if crops_dir.is_dir():
        nested = sorted(crops_dir.rglob("*"))
        images = [p for p in nested if p.is_file() and p.suffix in IMAGE_EXTS]
        if images:
            return images[0]

    return None


def discover_samples(root: Path) -> list[Sample]:
    """Recursively find sample directories under root, one per .tex file found.

    A directory with multiple .tex files (e.g. alpha_mask.tex, hop_bbox.tex,
    prog_reveal.tex, slide_bbox.tex as in examples/animation/*) yields one
    Sample per .tex file, all sharing the same resolved ground-truth image.
    """
    root = Path(root)
    samples: list[Sample] = []
    if not root.exists():
        return samples

    tex_files = sorted(root.rglob("*.tex"))
    for tex_path in tex_files:
        image_path = _find_ground_truth_image(tex_path)
        rel = tex_path.relative_to(root)
        sample_id = str(rel.with_suffix(""))
        samples.append(Sample(id=sample_id, tex_path=tex_path, image_path=image_path))

    return samples


def list_data_roots(base_dirs: list[Path]) -> list[Path]:
    return [d for d in base_dirs if d.exists()]
