"""Find raster placeholders in TikZ, and splice real crops into them.

The transmuter marks undrawable content with `xml class=raster_node`. This
module locates those nodes, and rewrites each one to draw an actual image via
`\\includegraphics` while keeping everything the rest of the pipeline depends
on: the `xml id` (which sequence `focus` lists reference), the TikZ node name,
the position, and the declared size.

Sizing is the subtle part. A placeholder is usually a `\\node` with
`minimum width`/`minimum height`, and its box is what the surrounding layout
was built around. The spliced image must therefore be constrained to that box
rather than drawn at its natural size, or the diagram reflows and the
pixel-coordinate mapping the transmuter established is lost.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# \node[...opts...] (name) at (x, y) {body};
_NODE_RE = re.compile(
    r"\\node\s*\[(?P<opts>[^\]]*)\]\s*"
    r"(?:\((?P<name>[^)]*)\)\s*)?"
    r"(?:at\s*\(\s*(?P<x>-?[\d.]+)\s*,\s*(?P<y>-?[\d.]+)\s*\)\s*)?"
    r"(?P<body>\{(?:[^{}]|\{[^{}]*\})*\})\s*;",
    re.DOTALL,
)

_XML_ID_RE = re.compile(r"xml\s+id\s*=\s*(?P<id>[\w:.-]+)")
_XML_CLASS_RE = re.compile(r"xml\s+class\s*=\s*(?P<cls>[\w:.-]+)")
_MIN_W_RE = re.compile(r"minimum\s+width\s*=\s*(?P<v>[\d.]+)\s*(?P<unit>[a-z]*)")
_MIN_H_RE = re.compile(r"minimum\s+height\s*=\s*(?P<v>[\d.]+)\s*(?P<unit>[a-z]*)")


@dataclass
class RasterPlaceholder:
    xml_id: str
    node_name: str
    text: str                       # visible label, used as a matching hint
    start: int                      # span of the whole \node ...; statement
    end: int
    body_start: int = 0             # span of the {...} body within the source
    body_end: int = 0
    x: float | None = None
    y: float | None = None
    width: float | None = None      # in pt, when declared
    height: float | None = None
    raw: str = ""

    def bbox(self) -> tuple[float, float, float, float] | None:
        """Approximate box in TikZ coordinates, for SoM overlays on a render.

        Only meaningful when the transmuter used the pixel-mapping convention
        (`x=1pt, y=-1pt`, top-left origin), which its prompt mandates.
        """
        if self.x is None or self.y is None:
            return None
        w = self.width or 60.0
        h = self.height or 40.0
        cy = abs(self.y)
        return (self.x - w / 2, cy - h / 2, self.x + w / 2, cy + h / 2)


def _to_pt(value: str, unit: str) -> float:
    """Normalize a TikZ length to points."""
    factor = {"pt": 1.0, "cm": 28.4528, "mm": 2.84528, "in": 72.27, "": 1.0}
    return float(value) * factor.get(unit, 1.0)


def find_placeholders(code: str) -> list[RasterPlaceholder]:
    """Every `xml class=raster_node` node in the source, in document order."""
    out = []
    for match in _NODE_RE.finditer(code):
        opts = match.group("opts")
        cls = _XML_CLASS_RE.search(opts)
        if not cls or cls.group("cls") != "raster_node":
            continue
        xml_id = _XML_ID_RE.search(opts)
        if not xml_id:
            continue

        width = height = None
        if (m := _MIN_W_RE.search(opts)):
            width = _to_pt(m.group("v"), m.group("unit"))
        if (m := _MIN_H_RE.search(opts)):
            height = _to_pt(m.group("v"), m.group("unit"))

        body = match.group("body")[1:-1].strip()
        out.append(RasterPlaceholder(
            xml_id=xml_id.group("id"),
            node_name=(match.group("name") or xml_id.group("id")).strip(),
            text=body,
            start=match.start(), end=match.end(),
            # Record the body's exact span. Locating it later by searching for
            # a brace is unsafe: a body like `{$\Phi(\mathbf{x})$}` contains
            # nested braces, and rindex would find the wrong one.
            body_start=match.start("body"), body_end=match.end("body"),
            x=float(match.group("x")) if match.group("x") else None,
            y=float(match.group("y")) if match.group("y") else None,
            width=width, height=height,
            raw=match.group(0),
        ))
    return out


def _includegraphics(image_rel: str, width: float | None, height: float | None) -> str:
    """An \\includegraphics constrained to the placeholder's declared box.

    Both dimensions are given when known, with `keepaspectratio` so the crop
    fits inside the box instead of being stretched to it -- a distorted
    screenshot reads worse than a letterboxed one.
    """
    if width and height:
        opts = f"width={width:.1f}pt, height={height:.1f}pt, keepaspectratio"
    elif width:
        opts = f"width={width:.1f}pt"
    elif height:
        opts = f"height={height:.1f}pt"
    else:
        opts = "width=60pt"
    return f"\\includegraphics[{opts}]{{{image_rel}}}"


def splice(code: str, replacements: dict[str, str]) -> tuple[str, list[str]]:
    """Replace each named placeholder's body with an \\includegraphics.

    `replacements` maps `xml id` to the image path to draw, relative to the
    .tex file. Returns the rewritten source and the ids actually replaced.

    Edits are applied back-to-front so earlier spans stay valid, and
    `graphicx` is added when missing -- `\\includegraphics` is undefined
    without it, which would fail the whole document.
    """
    placeholders = find_placeholders(code)
    targets = [p for p in placeholders if p.xml_id in replacements]
    if not targets:
        return code, []

    replaced = []
    for placeholder in sorted(targets, key=lambda p: p.body_start, reverse=True):
        image = replacements[placeholder.xml_id]
        graphic = _includegraphics(image, placeholder.width, placeholder.height)

        # Replace only the node body, by its recorded span. Everything else --
        # the options carrying xml id/class/parent, the position, the styling --
        # is left byte-for-byte intact, so nothing downstream loses its
        # reference.
        code = code[: placeholder.body_start] + f"{{{graphic}}}" + code[placeholder.body_end:]
        replaced.append(placeholder.xml_id)

    if replaced and "\\usepackage{graphicx}" not in code:
        if "\\documentclass" in code:
            end = code.index("\n", code.index("\\documentclass"))
            code = code[: end + 1] + "\\usepackage{graphicx}\n" + code[end + 1:]

    return code, list(reversed(replaced))


def document_extent(code: str) -> tuple[float, float] | None:
    """TikZ has no declared drawing extent, and that is the honest answer.

    An SVG states its coordinate space up front in `viewBox`. A TikZ picture's
    extent is whatever the compiled result happens to occupy -- it emerges from
    every node, path and label in the document and is not knowable without
    running LaTeX. Guessing it from the union of raster placeholders would be
    wrong in the one direction that matters, since placeholders are by
    definition a subset of the drawing.

    Returning `None` is not a gap here. The rendered figure is a crop of the
    finished picture, so the picture's aspect and the image's aspect already
    agree, and the caller's correction factor collapses to 1 -- which is exactly
    the behaviour that was correct for TikZ all along.
    """
    return None
