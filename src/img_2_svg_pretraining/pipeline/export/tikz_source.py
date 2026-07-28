"""Source-level TikZ transforms for export.

Extracted from the `export/tikz/*.py` scripts, which held this logic in
`__main__` blocks with hardcoded input paths. The algorithms are unchanged;
only the plumbing is now callable.

The two transforms take opposite approaches to the same animated source:
`to_multipage_pdf_source` unrolls it into one PDF page per frame (for
PPTX/MP4/GIF export), while `to_svg_source` keeps the `animate` package
intact for a genuinely animated SVG.
"""
from __future__ import annotations

import re

_DOCCLASS_OPTS_RE = re.compile(r"\\documentclass\[(.*?)\]\{standalone\}")
_MULTIFRAME_RE = re.compile(r"\\multiframe\{(\d+)\}\{\s*iFrame\s*=\s*(\d+)\+1\s*\}\{")


def frame_count(tikz_code: str) -> int | None:
    """Number of frames declared by `\\multiframe`, if present."""
    match = _MULTIFRAME_RE.search(tikz_code)
    return int(match.group(1)) if match else None


def to_multipage_pdf_source(tikz_code: str) -> str:
    """Rewrite animated TikZ so each frame becomes its own PDF page.

    `multi=tikzpicture` on the documentclass is what forces a page break per
    picture; without it every frame renders side by side on one enormous page.
    """
    # 1. One page per tikzpicture.
    tikz_code = _DOCCLASS_OPTS_RE.sub(
        r"\\documentclass[\1,multi=tikzpicture]{standalone}", tikz_code)
    if "multi=tikzpicture" not in tikz_code:
        tikz_code = tikz_code.replace(
            r"\documentclass{standalone}",
            r"\documentclass[multi=tikzpicture]{standalone}")

    # 2. Drop the animate machinery -- pages replace it.
    tikz_code = re.sub(r"\\usepackage\{animate\}\n?", "", tikz_code)
    tikz_code = re.sub(r"\\begin\{animateinline\}.*\n", "", tikz_code)
    tikz_code = re.sub(r"\\end\{animateinline\}", "", tikz_code)

    # 3. \multiframe{N}{iFrame=S+1}{ -> \foreach \iFrame in {S,...,S+N-1} {
    match = _MULTIFRAME_RE.search(tikz_code)
    if match:
        total, start = int(match.group(1)), int(match.group(2))
        end = start + total - 1
        replacement = f"\\foreach \\iFrame in {{{start},...,{end}}} {{"
        tikz_code = tikz_code[:match.start()] + replacement + tikz_code[match.end():]

    return tikz_code


def to_svg_source(tikz_code: str) -> str:
    """Prepare animated TikZ for dvisvgm.

    Keeps `animate`/`\\multiframe` intact -- the SVG stays animated. Adds the
    `dvisvgm` documentclass option, and `transparency group` to any scope that
    sets opacity, without which overlapping shapes blend against each other
    instead of fading as a unit.
    """
    if "dvisvgm" not in tikz_code:
        tikz_code = _DOCCLASS_OPTS_RE.sub(
            r"\\documentclass[\1,dvisvgm]{standalone}", tikz_code)
        if "dvisvgm" not in tikz_code:
            tikz_code = tikz_code.replace(
                r"\documentclass{standalone}",
                r"\documentclass[dvisvgm]{standalone}")

    def _add_transparency_group(match: re.Match) -> str:
        options = match.group(1)
        if "opacity" in options and "transparency group" not in options:
            return f"\\begin{{scope}}[{options}, transparency group]"
        return match.group(0)

    return re.sub(r"\\begin\{scope\}\[([^\]]*)\]", _add_transparency_group, tikz_code)
