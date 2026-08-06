"""Dataset discovery for the per-sample-directory layout.

Each sample is a self-contained directory:

    <root>/<id>/
        <id>.png          the diagram figure
        caption.txt       the figure caption
        abstract.tex|txt  paper abstract
        methods.tex|txt   paper method section
        arxiv_src/        (optional) full paper source, used for the title
        paper.md          (optional) whole paper as markdown, used as a
                          fallback when section files are absent

This differs from the flat Set-2 shape that `data_engine/samples.py` and
`viewer/samples.py` read (parallel `original_images/`, `tex_files/` dirs
keyed by filename), so neither of those discoverers works here.

The four context tiers correspond directly to `prompts/animation_planner/
strategizer.yaml`'s four prompt variants.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# Ordered by increasing context. `context_for` returns the fields each needs.
TIERS = ("image_only", "caption", "abstract", "full")

TIER_FIELDS: dict[str, tuple[str, ...]] = {
    "image_only": (),
    "caption": ("caption",),
    "abstract": ("caption", "abstract"),
    "full": ("caption", "abstract", "methods"),
}

_TITLE_RE = re.compile(r"\\title\s*\{", re.IGNORECASE)
# Strip LaTeX comments, but not an escaped \% which is a literal percent.
_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)


def _read_first(directory: Path, stem: str) -> str | None:
    """Read `<stem>.tex` or `<stem>.txt`, whichever exists.

    The extension varies per sample in this dataset -- most ship .tex, a
    couple ship .txt -- so both must be tried rather than assuming one.
    """
    for suffix in (".tex", ".txt", ".md"):
        path = directory / f"{stem}{suffix}"
        if path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="replace").strip() or None
            except OSError:
                return None
    return None


def _find_image(directory: Path, sample_id: str) -> Path | None:
    """Locate the figure image.

    Prefers `<id>.<ext>`, but falls back to any single image in the directory
    because at least one sample in this dataset has a typo'd filename
    (`CVPR_2025_pipe00011/CVPR_2025_pipe000011.png` -- an extra zero), and
    silently dropping a sample over a typo would be worse than accepting it.
    """
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate

    loose = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(loose) == 1:
        return loose[0]
    # Ambiguous: several images and none named for the sample. Prefer one whose
    # stem is a near-miss of the id (differs only by repeated characters).
    for path in loose:
        if path.stem.replace("0", "") == sample_id.replace("0", ""):
            return path
    return None


def _balanced_brace_content(text: str, open_index: int) -> str | None:
    """Extract the content of a {...} group starting at `open_index`."""
    depth = 0
    for i in range(open_index, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:i]
    return None


def _extract_title(arxiv_src: Path) -> str | None:
    """Pull \\title{...} out of the paper source.

    Tries main.tex / _main.tex first (both appear in this dataset) before
    falling back to scanning every .tex, so a \\title in rebuttal.tex doesn't
    win over the real one.
    """
    if not arxiv_src.is_dir():
        return None

    preferred = [arxiv_src / "main.tex", arxiv_src / "_main.tex"]
    others = sorted(p for p in arxiv_src.glob("*.tex") if p not in preferred)

    for path in [p for p in preferred if p.exists()] + others:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = _TITLE_RE.search(text)
        if not match:
            continue
        content = _balanced_brace_content(text, match.end() - 1)
        if not content:
            continue
        title = _COMMENT_RE.sub("", content)
        title = " ".join(title.split())
        if title:
            return title
    return None


@dataclass
class PaperSample:
    id: str
    directory: Path
    image_path: Path
    caption: str | None = None
    abstract: str | None = None
    methods: str | None = None
    title: str | None = None
    arxiv_src: Path | None = None
    extras: dict = None  # e.g. {"paper_md": Path, "video": Path}

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}

    def available_tiers(self) -> list[str]:
        """Tiers this sample has the context to run."""
        return [t for t in TIERS if self.supports_tier(t)]

    def supports_tier(self, tier: str) -> bool:
        if tier not in TIER_FIELDS:
            raise KeyError(f"unknown tier '{tier}'. Known: {list(TIERS)}")
        return all(getattr(self, field) for field in TIER_FIELDS[tier])

    def missing_for_tier(self, tier: str) -> list[str]:
        return [f for f in TIER_FIELDS[tier] if not getattr(self, f)]

    def context_for(self, tier: str) -> dict[str, str]:
        """Context fields to interpolate into a prompt for this tier.

        Title is included whenever known -- the abstract and full tiers name
        it in their prompts -- but is never required, since it comes from an
        optional arxiv_src.
        """
        if not self.supports_tier(tier):
            raise ValueError(
                f"sample '{self.id}' cannot run tier '{tier}': "
                f"missing {self.missing_for_tier(tier)}"
            )
        ctx = {f: getattr(self, f) for f in TIER_FIELDS[tier]}
        if self.title and tier in ("abstract", "full"):
            ctx["title"] = self.title
        return ctx


def load_sample(directory: Path) -> PaperSample | None:
    """Build a PaperSample from one directory, or None if it has no image."""
    directory = Path(directory)
    sample_id = directory.name
    image_path = _find_image(directory, sample_id)
    if image_path is None:
        return None

    arxiv_src = directory / "arxiv_src"
    arxiv_src = arxiv_src if arxiv_src.is_dir() else None

    extras: dict = {}
    paper_md = directory / "paper.md"
    if paper_md.exists():
        extras["paper_md"] = paper_md
    videos = sorted(p for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in (".webm", ".mp4", ".gif"))
    if videos:
        extras["videos"] = videos

    return PaperSample(
        id=sample_id,
        directory=directory,
        image_path=image_path,
        caption=_read_first(directory, "caption"),
        abstract=_read_first(directory, "abstract"),
        methods=_read_first(directory, "methods"),
        # A plain title.txt wins over parsing the paper source: it is what the
        # AnimateBench bundles ship (they carry no arxiv_src at all), and where
        # both exist the explicit file is the more reliable of the two.
        title=(_read_first(directory, "title")
               or (_extract_title(arxiv_src) if arxiv_src else None)),
        arxiv_src=arxiv_src,
        extras=extras,
    )


def discover_samples(root: Path, limit: int | None = None,
                     only: list[str] | None = None) -> list[PaperSample]:
    """Enumerate samples under `root`, sorted by id.

    Raises on a missing root rather than returning [] -- `viewer/samples.py`
    returns an empty list for a nonexistent root, which turns a typo'd path
    into a silent zero-sample run.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    wanted = set(only) if only else None
    samples = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if wanted is not None and directory.name not in wanted:
            continue
        sample = load_sample(directory)
        if sample is not None:
            samples.append(sample)

    if wanted:
        found = {s.id for s in samples}
        for missing in sorted(wanted - found):
            raise ValueError(f"requested sample '{missing}' not found under {root}")

    if limit:
        samples = samples[:limit]
    return samples
