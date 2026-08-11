"""Find raster placeholders in SVG, and splice real crops into them.

The SVG counterpart to `tikz_rasters.py`, exposing the same two names
(`find_placeholders`, `splice`) over the same `RasterPlaceholder` record, so
`raster_integrator` and the annotation tool dispatch on target and are
otherwise unchanged.

The converter marks a placeholder as a group carrying `data-class="raster_node"`
holding a dashed `<rect>` and a centred `<text>` label -- verified against real
converter output:

    <g id="raster_node_1" data-id="raster_node_1" data-class="raster_node" data-parent="root">
      <rect id="rect_raster_1" ... x="5" y="25" width="115" height="85"
            stroke-dasharray="5,5" ... />
      <text ... >[3D Smooth Convex]</text>
    </g>

Two things are easier here than in TikZ. Geometry is already absolute
top-left `x`/`y`/`width`/`height` in user units, so `bbox()` needs no
centre-offset arithmetic and no unit table. And SVG is real XML, so this
parses with ElementTree instead of a regex over brace-nesting -- which is
what makes "replace only the placeholder, leave everything else byte-identical"
straightforward to guarantee.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from dataclasses import dataclass

from .tikz_rasters import RasterPlaceholder

SVG_NS = "http://www.w3.org/2000/svg"
RASTER_CLASS = "raster_node"


@dataclass
class SvgRasterPlaceholder(RasterPlaceholder):
    """A raster placeholder whose coordinates are SVG user units.

    Subclassed rather than reusing `RasterPlaceholder` directly so `bbox()`
    carries the right geometry with the record. The base implementation
    assumes TikZ's centre-anchored coordinates with a flipped y axis, so a
    caller that got a plain `RasterPlaceholder` back from this module would
    silently receive every box shifted by half its own size -- and
    `raster_integrator._integrate` calls `.bbox()` without knowing which
    target produced the record.
    """

    def bbox(self) -> tuple[float, float, float, float] | None:
        """Top-left box in SVG user units. Rects are already top-left."""
        if self.x is None or self.y is None:
            return None
        w = self.width or 60.0
        h = self.height or 40.0
        return (self.x, self.y, self.x + w, self.y + h)

# Matches one <g ...> open tag carrying data-class="raster_node", capturing
# enough to locate the element and its id. Used only to find *spans* in the
# original text; all attribute reading goes through ElementTree.
_GROUP_OPEN_RE = re.compile(
    r"<g\b[^>]*?data-class\s*=\s*[\"']raster_node[\"'][^>]*>", re.IGNORECASE)
_ID_RE = re.compile(r"\b(?:data-id|id)\s*=\s*[\"']([^\"']+)[\"']")


def _local(tag: str) -> str:
    """Tag name without its namespace."""
    return tag.rsplit("}", 1)[-1]


def _group_end(code: str, start: int) -> int:
    """Index just past the `</g>` closing the group opened at `start`.

    Counted rather than regex-matched, since raster groups can nest inside
    other groups and a non-greedy match would stop at the wrong closer.
    """
    depth = 0
    for match in re.finditer(r"<g\b[^>]*?>|</g\s*>|<g\b[^>]*?/>", code[start:]):
        text = match.group(0)
        if text.endswith("/>"):
            continue                      # self-closing, no depth change
        depth += 1 if not text.startswith("</") else -1
        if depth == 0:
            return start + match.end()
    return -1


def find_placeholders(code: str) -> list[RasterPlaceholder]:
    """Every `data-class="raster_node"` group, in document order."""
    out: list[RasterPlaceholder] = []
    for match in _GROUP_OPEN_RE.finditer(code):
        start = match.start()
        end = _group_end(code, start)
        if end < 0:
            continue                      # unbalanced; skip rather than guess
        raw = code[start:end]

        id_match = _ID_RE.search(match.group(0))
        if not id_match:
            continue                      # nothing to address it by
        xml_id = id_match.group(1)

        # An already-filled raster is not a placeholder. The group keeps its
        # id and class after a splice -- downstream stages address elements by
        # id, so it has to -- which means the only thing distinguishing filled
        # from empty is the content. Without this check a second detection
        # pass would try to re-fill filled rasters, and with the `<rect>` gone
        # there would be no geometry left to guide the crop. TikZ gets this
        # for free because its splice rewrites the node body.
        if "<image" in raw:
            continue

        x = y = width = height = None
        label = ""
        try:
            element = ET.fromstring(raw)
        except ET.ParseError:
            element = None
        if element is not None:
            for child in element.iter():
                name = _local(child.tag)
                if name == "rect" and x is None:
                    try:
                        x = float(child.get("x", "0"))
                        y = float(child.get("y", "0"))
                        width = float(child.get("width", "0"))
                        height = float(child.get("height", "0"))
                    except (TypeError, ValueError):
                        x = y = width = height = None
                elif name == "text" and not label:
                    label = "".join(child.itertext()).strip().strip("[]")

        out.append(SvgRasterPlaceholder(
            xml_id=xml_id,
            node_name=xml_id,
            text=label,
            start=start, end=end,
            # The whole group is the replaceable body: unlike TikZ, where only
            # the node's `{...}` is rewritten, here the dashed rect and its
            # label are together the thing the image replaces.
            body_start=start, body_end=end,
            x=x, y=y, width=width, height=height,
            raw=raw,
        ))
    return out


def _image_element(placeholder: RasterPlaceholder, href: str) -> str:
    """An `<image>` occupying exactly the placeholder's box.

    `preserveAspectRatio="xMidYMid meet"` letterboxes the crop inside the box
    rather than stretching it, matching the `keepaspectratio` the TikZ splice
    passes to `\\includegraphics` -- a distorted screenshot reads worse than a
    letterboxed one.
    """
    x = placeholder.x if placeholder.x is not None else 0.0
    y = placeholder.y if placeholder.y is not None else 0.0
    w = placeholder.width or 60.0
    h = placeholder.height or 40.0
    return (
        f'<g id="{placeholder.xml_id}" data-id="{placeholder.xml_id}" '
        f'data-class="{RASTER_CLASS}">'
        f'<image href="{href}" xlink:href="{href}" '
        f'x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
        f'preserveAspectRatio="xMidYMid meet" />'
        f"</g>"
    )


def splice(code: str, replacements: dict[str, str]) -> tuple[str, list[str]]:
    """Replace each named placeholder group with an `<image>`.

    `replacements` maps `data-id` to the image path to draw. Returns the
    rewritten source and the ids actually replaced.

    Edits are applied back-to-front so earlier spans stay valid, and the
    `xlink` namespace is declared when missing -- older renderers still read
    `xlink:href`, and a missing declaration makes the document malformed.
    Everything outside the replaced groups is left byte-for-byte intact, which
    is what lets a re-splice swap crops without disturbing a critic's repairs.
    """
    placeholders = find_placeholders(code)
    targets = [p for p in placeholders if p.xml_id in replacements]
    if not targets:
        return code, []

    replaced: list[str] = []
    for placeholder in sorted(targets, key=lambda p: p.start, reverse=True):
        element = _image_element(placeholder, replacements[placeholder.xml_id])
        code = code[: placeholder.start] + element + code[placeholder.end:]
        replaced.append(placeholder.xml_id)

    if replaced and "xmlns:xlink" not in code:
        code = re.sub(r"<svg\b", '<svg xmlns:xlink="http://www.w3.org/1999/xlink"',
                      code, count=1)

    return code, list(reversed(replaced))
