"""Read, edit and re-serialize the parsed structure XML.

The pipeline only ever *writes* this XML (`planner/parser.py`) and reads ids
out of it; nothing in the repo edits it, so the round-trip lives here.

Two rules the rest of the tool depends on:

- Serialization must reproduce the model's own 2-space pretty-printed layout,
  or every save shows a whole-file diff. `ET.tostring` alone emits one line;
  `ET.indent` is what makes the round-trip stable.
- `depth` is derived from nesting, never edited. It is redundant with the tree
  structure, and the two disagreeing is a defect the annotator has no reason
  to introduce -- so it is recomputed on every serialize.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

ROOT_TAG = "Diagram"

# The element classes the parser prompt defines. The tag *is* the class.
CLASSES = (
    "block", "standalone_node", "child_node", "raster_node",
    "composite_whole", "composite_part", "edge",
)

def parse(xml_text: str) -> ET.Element:
    """Parse, raising ET.ParseError with the original message on failure."""
    return ET.fromstring(xml_text)


def serialize(root: ET.Element) -> str:
    """Back to text, matching the parser's own formatting."""
    _renumber_document(root)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def _renumber_depth(root: ET.Element, depth: int = 1) -> None:
    """Set `depth` from nesting, matching the parser's own convention.

    `<Diagram>` and its direct children are both depth 1 -- the root is not a
    level of its own -- so numbering starts at the root's children rather than
    below the root. `composite_part` is the other exception: it shares its
    `composite_whole`'s depth rather than sitting one below it.
    """
    root.set("depth", str(depth))
    for child in root:
        child_depth = depth if child.tag == "composite_part" else depth + 1
        _renumber_depth(child, child_depth)


def _renumber_document(root: ET.Element) -> None:
    """Depth numbering for a whole document, root included."""
    root.set("depth", "1")
    for child in root:
        _renumber_depth(child, 1)


def to_tree(root: ET.Element) -> list[dict]:
    """The document as nested dicts, for the tree editor."""
    def node(el: ET.Element) -> dict:
        out = {
            "tag": el.tag,
            "id": el.get("id"),
            "depth": el.get("depth"),
            "children": [node(c) for c in el],
        }
        if el.tag == "edge":
            out["source"] = el.get("source")
            out["target"] = el.get("target")
        return out

    return [node(c) for c in root]


def from_tree(nodes: list[dict]) -> ET.Element:
    """Rebuild a document from the tree editor's payload."""
    root = ET.Element(ROOT_TAG)

    def build(parent: ET.Element, entry: dict) -> None:
        tag = entry.get("tag") or "standalone_node"
        el = ET.SubElement(parent, tag)
        if entry.get("id"):
            el.set("id", str(entry["id"]))
        if tag == "edge":
            for attr in ("source", "target"):
                if entry.get(attr):
                    el.set(attr, str(entry[attr]))
        for child in entry.get("children") or []:
            build(el, child)

    for entry in nodes:
        build(root, entry)
    return root


def id_diff(new_xml: str, old_xml: str | None) -> tuple[list[str], list[str]]:
    """(added, removed) element ids, new against old.

    Removals are the dangerous direction: these ids are the authoritative set
    every downstream `focus` entry is checked against, so dropping one breaks
    every sequence step that referenced it.
    """
    from ..planner.parser import element_ids

    new_ids = element_ids(new_xml)
    old_ids = element_ids(old_xml) if old_xml else set()
    return sorted(new_ids - old_ids), sorted(old_ids - new_ids)


def focus_conflicts(removed_ids: list[str], sequence_path) -> list[str]:
    """Which removed ids the existing sequence still focuses.

    Cheap local set intersection, and the most useful warning on the XML
    screen: it turns "you removed an id" into "you broke these steps".
    """
    from pathlib import Path

    from ..schema import AnimationSequence

    path = Path(sequence_path)
    if not removed_ids or not path.exists():
        return []
    try:
        seq = AnimationSequence.load(path)
    except Exception:
        return []
    focused = seq.focus_ids()
    return sorted(set(removed_ids) & focused)
