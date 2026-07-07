"""Shared read/write schema for diagram layout XML.

Extends the hand-authored format found in `data/partial_test_set/xml_files`
(confirmed by inspecting all 100 files: exactly 4 tags -- `diagram`, `block`,
`node`, `arrow` -- with attributes `i` (id), `t` (label text), `b` (bbox as
"x0 y0 x1 y1", TikZ-style coordinates which may be negative), and `s`/`d`
(source/dest id) on `arrow`. Blocks may be self-closing or contain nested
`node`/`arrow` children; `arrow` may appear nested inside a `block` or as a
top-level child of `diagram`) with optional pipeline-provenance attributes:
`mask`, `conf`, `src` on `block`/`node`/`arrow`, plus a new `raster` leaf for
embedded photo/plot regions that hand files never contain. All new
attributes are optional so every existing hand-authored file round-trips
unchanged (parse -> serialize -> parse is structurally identical).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree import ElementTree as ET
from xml.dom import minidom


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def parse(cls, s: str) -> "BBox":
        x0, y0, x1, y1 = (float(v) for v in s.split())
        return cls(x0, y0, x1, y1)

    def render(self) -> str:
        return f"{self.x0:.2f} {self.y0:.2f} {self.x1:.2f} {self.y1:.2f}"


@dataclass
class Node:
    id: str
    text: str
    bbox: BBox
    mask: str | None = None   # path to a mask PNG, if segmented by the pipeline
    conf: float | None = None  # detector confidence, if pipeline-produced
    src: str = "manual"        # "manual" | "pointing" | "sam"


@dataclass
class Raster:
    """An embedded photo/plot region -- distinct from `Node`, which is a
    labeled shape/box. Hand-authored files never contain this tag."""
    id: str
    text: str
    bbox: BBox
    mask: str | None = None
    conf: float | None = None
    src: str = "pointing"


@dataclass
class Arrow:
    src_id: str
    dst_id: str
    conf: float | None = None
    src: str = "manual"


@dataclass
class Block:
    id: str
    text: str
    bbox: BBox
    nodes: list[Node] = field(default_factory=list)
    rasters: list[Raster] = field(default_factory=list)
    arrows: list[Arrow] = field(default_factory=list)
    mask: str | None = None
    conf: float | None = None
    src: str = "manual"


@dataclass
class Diagram:
    blocks: list[Block] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)      # top-level nodes
    rasters: list[Raster] = field(default_factory=list)  # top-level rasters
    arrows: list[Arrow] = field(default_factory=list)    # top-level arrows

    def all_nodes(self) -> list[Node]:
        """Every Node in the diagram, top-level and nested in blocks."""
        result = list(self.nodes)
        for block in self.blocks:
            result.extend(block.nodes)
        return result

    def all_rasters(self) -> list[Raster]:
        result = list(self.rasters)
        for block in self.blocks:
            result.extend(block.rasters)
        return result

    def all_arrows(self) -> list[Arrow]:
        result = list(self.arrows)
        for block in self.blocks:
            result.extend(block.arrows)
        return result

    def find_id(self, node_id: str) -> Node | Block | Raster | None:
        for collection in (self.all_nodes(), self.blocks, self.all_rasters()):
            for item in collection:
                if item.id == node_id:
                    return item
        return None


def _opt_float(el: ET.Element, key: str) -> float | None:
    v = el.get(key)
    return float(v) if v is not None else None


def _parse_node(el: ET.Element) -> Node:
    return Node(
        id=el.get("i", ""),
        text=el.get("t", ""),
        bbox=BBox.parse(el.get("b", "0 0 0 0")),
        mask=el.get("mask"),
        conf=_opt_float(el, "conf"),
        src=el.get("src", "manual"),
    )


def _parse_raster(el: ET.Element) -> Raster:
    return Raster(
        id=el.get("i", ""),
        text=el.get("t", ""),
        bbox=BBox.parse(el.get("b", "0 0 0 0")),
        mask=el.get("mask"),
        conf=_opt_float(el, "conf"),
        src=el.get("src", "pointing"),
    )


def _parse_arrow(el: ET.Element) -> Arrow:
    return Arrow(
        src_id=el.get("s", ""),
        dst_id=el.get("d", ""),
        conf=_opt_float(el, "conf"),
        src=el.get("src", "manual"),
    )


def _parse_block(el: ET.Element) -> Block:
    block = Block(
        id=el.get("i", ""),
        text=el.get("t", ""),
        bbox=BBox.parse(el.get("b", "0 0 0 0")),
        mask=el.get("mask"),
        conf=_opt_float(el, "conf"),
        src=el.get("src", "manual"),
    )
    for child in el:
        if child.tag == "node":
            block.nodes.append(_parse_node(child))
        elif child.tag == "raster":
            block.rasters.append(_parse_raster(child))
        elif child.tag == "arrow":
            block.arrows.append(_parse_arrow(child))
    return block


def parse_xml(xml_text: str) -> Diagram:
    root = ET.fromstring(xml_text)
    diagram = Diagram()
    for child in root:
        if child.tag == "block":
            diagram.blocks.append(_parse_block(child))
        elif child.tag == "node":
            diagram.nodes.append(_parse_node(child))
        elif child.tag == "raster":
            diagram.rasters.append(_parse_raster(child))
        elif child.tag == "arrow":
            diagram.arrows.append(_parse_arrow(child))
    return diagram


def parse_xml_file(path) -> Diagram:
    return parse_xml(open(path, encoding="utf-8").read())


def _set_common(el: ET.Element, mask: str | None, conf: float | None, src: str) -> None:
    if mask is not None:
        el.set("mask", mask)
    if conf is not None:
        el.set("conf", f"{conf:.4f}")
    if src != "manual":
        el.set("src", src)


def _build_node(parent: ET.Element, node: Node) -> None:
    el = ET.SubElement(parent, "node", i=node.id, t=node.text, b=node.bbox.render())
    _set_common(el, node.mask, node.conf, node.src)


def _build_raster(parent: ET.Element, raster: Raster) -> None:
    el = ET.SubElement(parent, "raster", i=raster.id, t=raster.text, b=raster.bbox.render())
    _set_common(el, raster.mask, raster.conf, raster.src)


def _build_arrow(parent: ET.Element, arrow: Arrow) -> None:
    el = ET.SubElement(parent, "arrow", s=arrow.src_id, d=arrow.dst_id)
    _set_common(el, None, arrow.conf, arrow.src)


def _build_block(parent: ET.Element, block: Block) -> None:
    el = ET.SubElement(parent, "block", i=block.id, t=block.text, b=block.bbox.render())
    _set_common(el, block.mask, block.conf, block.src)
    for node in block.nodes:
        _build_node(el, node)
    for raster in block.rasters:
        _build_raster(el, raster)
    for arrow in block.arrows:
        _build_arrow(el, arrow)


def to_xml(diagram: Diagram, pretty: bool = True) -> str:
    root = ET.Element("diagram")
    for block in diagram.blocks:
        _build_block(root, block)
    for node in diagram.nodes:
        _build_node(root, node)
    for raster in diagram.rasters:
        _build_raster(root, raster)
    for arrow in diagram.arrows:
        _build_arrow(root, arrow)

    raw = ET.tostring(root, encoding="unicode")
    if not pretty:
        return raw
    return minidom.parseString(raw).toprettyxml(indent="  ").split("\n", 1)[1].strip() + "\n"


def write_xml_file(diagram: Diagram, path, pretty: bool = True) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_xml(diagram, pretty=pretty))
