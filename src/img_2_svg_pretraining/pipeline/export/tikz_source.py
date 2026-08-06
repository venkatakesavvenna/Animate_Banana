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

# `\opT1`, `\op2` -- a control sequence whose name ends in digits.
_DIGIT_MACRO_RE = re.compile(r"\\([a-zA-Z]+)(\d+)\b")

# Digits map to letters so the rewritten name stays a legal control sequence.
_DIGIT_LETTERS = str.maketrans("0123456789", "abcdefghij")


def fix_digit_macro_names(tikz_code: str) -> str:
    """Rename macros whose names end in digits, e.g. `\\opT1` -> `\\opTb`.

    A TeX control sequence is letters only, so `\\pgfmathsetmacro{\\opT1}{...}`
    does not define `\\opT1` -- it defines `\\opT` and leaves a literal `1`.
    Eight such lines therefore redefine one macro eight times, and the first
    use fails with:

        ! Use of \\opT doesn't match its definition.

    Gemma 4 produced exactly this in both of its animations while Gemini (16)
    and Qwen (4) produced none, so it is a per-model habit rather than a
    prompt-wide problem, and cheap to repair at the source level.

    The rename is applied uniformly to definitions and uses alike, so the
    mapping stays consistent. `\\opT1` and `\\opTb` colliding would require the
    source to already use both spellings, which would be broken anyway.
    """
    if not _DIGIT_MACRO_RE.search(tikz_code):
        return tikz_code

    def rename(match: re.Match) -> str:
        name, digits = match.group(1), match.group(2)
        return "\\" + name + digits.translate(_DIGIT_LETTERS)

    return _DIGIT_MACRO_RE.sub(rename, tikz_code)


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

    # 1b. `\usetikzlibrary` is defined by the tikz package, so a preamble that
    # calls it without loading tikz first dies with "Undefined control
    # sequence" and produces no PDF at all. Observed once in 14 generated
    # animations -- a rare model slip, but one that costs the whole render, so
    # it is cheaper to repair here than to lose the sample. Only ever *adds* a
    # missing package; a preamble that already loads tikz is untouched.
    if "usetikzlibrary" in tikz_code and not re.search(
            r"\\usepackage(\[[^]]*\])?\{tikz\}", tikz_code):
        tikz_code = re.sub(r"(\\documentclass[^\n]*\n)",
                           r"\1\\usepackage{tikz}\n", tikz_code, count=1)

    # 1c. Repair control sequences whose names end in digits, which TeX cannot
    # define. See fix_digit_macro_names.
    tikz_code = fix_digit_macro_names(tikz_code)

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
