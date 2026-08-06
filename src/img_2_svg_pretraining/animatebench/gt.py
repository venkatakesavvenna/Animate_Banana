"""Locate and parse the benchmark's ground-truth artifacts for one sample.

The reference bundle is imported by `pipeline/import_bench.py` into
`<dataset_root>/<sample>/reference/`, and this module is the only place that
knows that layout. Everything is returned parsed: XML as element records,
sequences as `AnimationSequence`, so the metrics never touch file paths.

The XML parser here is deliberately not a general XML library wrapper: the
schema is flat and regular (every element is one tag carrying id/depth and
optionally source/target), and what the metrics need is the *element table* --
id, class, parent, depth, edges -- not a DOM.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from img_2_svg_pretraining.pipeline.schema import AnimationSequence

# Classes the design doc excludes from cross-method scoring: sub-part
# decomposition is language-dependent (evaluation doc §2.1.3 note).
EXCLUDED_CLASSES = ("composite_part",)

EDGE_CLASS = "edge"


@dataclass
class XmlElement:
    id: str
    cls: str                      # tag name = semantic class
    parent: str | None            # id of parent element, None for root children
    depth: int                    # as declared in the document
    rel_depth: int                # normalized: root children = 1
    source: str | None = None     # edges only
    target: str | None = None


@dataclass
class DiagramXml:
    """Element table for one structure XML document."""

    elements: dict[str, XmlElement] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "DiagramXml":
        root = ET.fromstring(text)
        table: dict[str, XmlElement] = {}

        # Depth conventions differ between documents: the reference bundle
        # declares top-level blocks at depth=2 (the Diagram itself is 1), our
        # pipeline declares them at depth=1. rel_depth counts nesting from the
        # root instead, so the two are comparable.
        def walk(node: ET.Element, parent_id: str | None, level: int) -> None:
            for child in node:
                el_id = child.get("id")
                if not el_id:
                    continue
                declared = child.get("depth")
                table[el_id] = XmlElement(
                    id=el_id,
                    cls=child.tag,
                    parent=parent_id,
                    depth=int(declared) if declared and declared.isdigit() else level,
                    rel_depth=level,
                    source=child.get("source"),
                    target=child.get("target"),
                )
                walk(child, el_id, level + 1)

        walk(root, None, 1)
        return cls(elements=table)

    @classmethod
    def load(cls, path: Path) -> "DiagramXml":
        return cls.parse(Path(path).read_text(encoding="utf-8"))

    # -- views the metrics consume ---------------------------------------

    def ids(self) -> set[str]:
        return set(self.elements)

    def scorable(self) -> dict[str, XmlElement]:
        """Non-edge elements that cross-method metrics may score."""
        return {i: e for i, e in self.elements.items()
                if e.cls != EDGE_CLASS and e.cls not in EXCLUDED_CLASSES}

    def edges(self) -> dict[str, XmlElement]:
        return {i: e for i, e in self.elements.items() if e.cls == EDGE_CLASS}

    def edge_pairs(self) -> set[tuple[str, str]]:
        """(source, target) id pairs for every edge that declares both."""
        return {(e.source, e.target) for e in self.edges().values()
                if e.source and e.target}


@dataclass
class GroundTruth:
    """Everything the reference bundle provides for one sample."""

    sample_id: str
    directory: Path               # <dataset_root>/<sample>/reference
    image_path: Path

    def xml_path(self, target: str = "tikz") -> Path:
        return self.directory / "xml" / f"{self.sample_id}_{target}.xml"

    def xml(self, target: str = "tikz") -> DiagramXml:
        return DiagramXml.load(self.xml_path(target))

    def sequence_path(self, style: str, target: str = "tikz") -> Path:
        return self.directory / "seq" / f"{style}_{self.sample_id}_{target}.json"

    def sequence(self, style: str, target: str = "tikz") -> AnimationSequence:
        return AnimationSequence.load(self.sequence_path(style, target))

    def diagram_code_path(self, target: str = "tikz") -> Path:
        ext = ".tex" if target == "tikz" else ".svg"
        return self.directory / "diagram" / f"{self.sample_id}_{target}{ext}"

    def has_style(self, style: str, target: str = "tikz") -> bool:
        return self.sequence_path(style, target).exists()


def load_ground_truth(dataset_root: Path, sample_id: str) -> GroundTruth:
    directory = Path(dataset_root) / sample_id / "reference"
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no reference bundle for '{sample_id}' at {directory}. "
            f"Run pipeline/import_bench.py first."
        )
    image = Path(dataset_root) / sample_id / f"{sample_id}.png"
    return GroundTruth(sample_id=sample_id, directory=directory, image_path=image)


# Elements referenced by a sequence, taken from the bucket split when present
# (bench dialect) and from focus otherwise. Style-aware filtering happens in
# the metric, not here.
def sequence_elements(seq: AnimationSequence) -> dict[str, set[str]]:
    """All element ids a sequence ever animates, split by bucket."""
    out: dict[str, set[str]] = {"blocks": set(), "nodes": set(),
                                "text": set(), "arrows": set()}
    for node in seq.nodes:
        if node.element_classes:
            for bucket, ids in node.element_classes.items():
                out.setdefault(bucket, set()).update(ids)
        else:
            out["nodes"].update(node.focus)
    return out


def first_appearance(seq: AnimationSequence) -> dict[str, int]:
    """Element id -> the first traversal step (1-based) that animates it."""
    order: dict[str, int] = {}
    by_id = {n.id: n for n in seq.nodes}
    for index, node_id in enumerate(seq.traversal, 1):
        node = by_id.get(node_id)
        if node is None:
            continue
        ids = ([i for ids in (node.element_classes or {}).values() for i in ids]
               if node.element_classes else list(node.focus))
        for el in ids:
            order.setdefault(el, index)
    return order
