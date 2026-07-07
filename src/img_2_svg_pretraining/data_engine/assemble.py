"""Combines pointing + segmentation output into a Diagram (schema.py), with
empty arrows -- edges are filled in later by edges.py. Pure geometry/data
structuring, no model calls, so it's independently unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass

from img_2_svg_pretraining.data_engine.schema import BBox, Block, Diagram, Node, Raster


@dataclass
class DetectedItem:
    """One point+mask pair from the pointing/segmentation stages, before
    being classified into a Node/Block/Raster."""
    id: str
    text: str
    bbox: BBox
    mask_path: str | None
    conf: float | None
    is_raster: bool


def _contains(outer: BBox, inner: BBox, tol: float = 1.0) -> bool:
    return (
        outer.x0 - tol <= inner.x0
        and outer.y0 - tol <= inner.y0
        and inner.x1 <= outer.x1 + tol
        and inner.y1 <= outer.y1 + tol
    )


def assemble_diagram(items: list[DetectedItem]) -> Diagram:
    """Classify detected node items into top-level Nodes vs Blocks (a Block
    is any detected node whose bbox strictly contains one or more other
    detected nodes' bboxes -- i.e. a container), and detected raster items
    into top-level Rasters or nested under whichever Block contains them.
    Arrows are left empty; edges.py fills them in from the assembled node
    ids.
    """
    node_items = [item for item in items if not item.is_raster]
    raster_items = [item for item in items if item.is_raster]

    # A block is any node item that strictly contains at least one other
    # node item -- sort largest-area first so containers are identified
    # before their children.
    def area(item: DetectedItem) -> float:
        b = item.bbox
        return (b.x1 - b.x0) * (b.y1 - b.y0)

    sorted_items = sorted(node_items, key=area, reverse=True)
    children_of: dict[str, list[DetectedItem]] = {item.id: [] for item in node_items}
    claimed: set[str] = set()

    for outer in sorted_items:
        for inner in node_items:
            if inner.id == outer.id or inner.id in claimed:
                continue
            if _contains(outer.bbox, inner.bbox):
                children_of[outer.id].append(inner)
                claimed.add(inner.id)

    diagram = Diagram()
    for item in node_items:
        if item.id in claimed:
            continue  # this item is itself a child of some other block
        kids = children_of.get(item.id, [])
        if kids:
            block = Block(
                id=item.id, text=item.text, bbox=item.bbox,
                mask=item.mask_path, conf=item.conf, src="sam",
            )
            for kid in kids:
                block.nodes.append(
                    Node(id=kid.id, text=kid.text, bbox=kid.bbox,
                         mask=kid.mask_path, conf=kid.conf, src="sam")
                )
            diagram.blocks.append(block)
        else:
            diagram.nodes.append(
                Node(id=item.id, text=item.text, bbox=item.bbox,
                     mask=item.mask_path, conf=item.conf, src="sam")
            )

    for item in raster_items:
        raster = Raster(
            id=item.id, text=item.text, bbox=item.bbox,
            mask=item.mask_path, conf=item.conf, src="sam",
        )
        owning_block = next(
            (b for b in diagram.blocks if _contains(b.bbox, item.bbox)), None,
        )
        if owning_block is not None:
            owning_block.rasters.append(raster)
        else:
            diagram.rasters.append(raster)

    return diagram
