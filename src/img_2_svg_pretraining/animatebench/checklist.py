"""The element checklist: what an animation committed to putting on screen.

Elimination 3 (omitted elements) and the unnecessary-repetition contributor
both begin with the same question -- what is the full set of elements this
animation is supposed to show -- and the design document reproduces the prompt
for it byte-identically in both, under a heading in capitals saying it is the
same step. So it is built once, validated, cached and shared, the treatment
`alignment.py` gives the GT<->prediction grouping and for the same reason: two
metrics that disagree about what "the elements" are cannot be read against each
other.

**It is derived from the sequence, not the XML.** The document is explicit that
the XML cannot serve -- "not all elements in the XML may explicitly make it to
the actual animated list". That has a consequence which must travel with every
number computed from it: an element the sequencer never scheduled can never be
flagged as omitted. This measures whether the animation rendered what it
promised, not whether it covered the figure. `coverage_recall` in the sequence
suite is the metric that answers the other question.

**The judge is not trusted.** A hallucinated checklist entry is unfalsifiable --
nothing on screen can ever match it, so it counts as an omission in every frame
forever, and would silently depress the score of a perfectly good animation.
`validate` therefore drops what it cannot use and records what it dropped, the
same enforcement `alignment.validate` applies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from img_2_svg_pretraining.pipeline.prompts import load_and_render
from img_2_svg_pretraining.pipeline.schema import AnimationSequence

from .gt import DiagramXml
from .judge import PROMPTS_ROOT

# XML tag -> checklist bucket. `composite_part` is excluded because the prompt
# asks for the whole element's label only ("Do not clutter the checklist with
# the names of internal sub-components"), and `text_node` because the prompt
# excludes standalone text outright ("We only care if the structural
# components ... are influenced by the style").
_BUCKET_BY_CLASS = {
    "block": "blocks",
    "edge": "edges",
    "standalone_node": "nodes",
    "child_node": "nodes",
    "raster_node": "nodes",
    "composite_whole": "nodes",
}
_SKIP_CLASSES = ("composite_part", "text_node")

BUCKETS = ("blocks", "nodes", "edges")


class ChecklistError(Exception):
    pass


def sequence_view(seq: AnimationSequence, xml: DiagramXml) -> str:
    """The animation sequence as bucketed ids, one line per timestep.

    Buckets come from the **XML**, not from the sequence. 27 of the 30 scored
    cells store the native dialect, where `element_classes` is empty and
    `gt.sequence_elements` drops every id into `nodes` -- so reading buckets
    off the sequence would report no blocks and no arrows for 90% of the data,
    and the checklist would come back with an empty `edges` list that looked
    like a finding rather than a plumbing bug.
    """
    by_id = {n.id: n for n in seq.nodes}
    lines: list[str] = []

    for index, node_id in enumerate(seq.traversal, 1):
        node = by_id.get(node_id)
        if node is None:
            continue
        if node.element_classes:
            ids = [i for ids in node.element_classes.values() for i in ids]
        else:
            ids = list(node.focus)

        found: dict[str, list[str]] = {b: [] for b in BUCKETS}
        unknown: list[str] = []
        for el_id in ids:
            element = xml.elements.get(el_id)
            if element is None:
                # Animated but absent from the structure XML. Kept: the judge
                # can still ground it from the code, and dropping it silently
                # would understate what the animation promised.
                unknown.append(el_id)
                continue
            if element.cls in _SKIP_CLASSES:
                continue
            found[_BUCKET_BY_CLASS.get(element.cls, "nodes")].append(el_id)

        parts = [f"{b}={found[b]}" for b in BUCKETS if found[b]]
        if unknown:
            parts.append(f"not_in_xml={unknown}")
        lines.append(f"  t{index} ({node_id}): " + (" ".join(parts) or "(nothing)"))

    return "\n".join(lines) if lines else "  (sequence animates nothing)"


@dataclass
class Entry:
    """One checklist label and how many physically distinct elements carry it.

    Frequency exists because a diagram routinely contains several distinct
    elements with the same text ("Layer Norm" three times over). Deduplicating
    by label alone collapsed them into one, so an animation that showed one of
    the three scored as having shown all of them, and two real omissions
    vanished. `source_ids` is what makes the count auditable -- it names which
    ids the judge counted.
    """
    label: str
    frequency: int = 1
    source_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"label": self.label, "frequency": self.frequency,
                "source_ids": list(self.source_ids)}


@dataclass
class Checklist:
    blocks: list[Entry] = field(default_factory=list)
    nodes: list[Entry] = field(default_factory=list)
    edges: list[Entry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def items(self, include_edges: bool = True) -> list[Entry]:
        out = list(self.blocks) + list(self.nodes)
        return out + list(self.edges) if include_edges else out

    def total(self, include_edges: bool = True) -> int:
        """Total INSTANCES, not distinct labels.

        The omission rate divides by this, and an animation is responsible for
        every instance it promised -- three "Layer Norm" nodes are three things
        that have to appear, not one.
        """
        return sum(e.frequency for e in self.items(include_edges))

    def to_dict(self) -> dict:
        return {"blocks": [e.to_dict() for e in self.blocks],
                "nodes": [e.to_dict() for e in self.nodes],
                "edges": [e.to_dict() for e in self.edges],
                "notes": self.notes, "provenance": self.provenance}

    @classmethod
    def from_dict(cls, data: dict) -> "Checklist":
        def entries(bucket: str) -> list[Entry]:
            out: list[Entry] = []
            for value in (data.get(bucket) or []):
                # Plain strings are checklists cached before frequencies
                # existed; they read back as one instance each rather than
                # forcing every stored checklist to be rebuilt.
                if isinstance(value, str):
                    out.append(Entry(label=value))
                elif isinstance(value, dict) and value.get("label"):
                    out.append(Entry(label=str(value["label"]),
                                     frequency=int(value.get("frequency") or 1),
                                     source_ids=list(value.get("source_ids") or [])))
            return out

        return cls(blocks=entries("blocks"), nodes=entries("nodes"),
                   edges=entries("edges"),
                   notes=list(data.get("notes") or []),
                   provenance=dict(data.get("provenance") or {}))

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Checklist":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate(raw: dict) -> Checklist:
    """Keep what is usable from the judge's reply, and say what was dropped."""
    payload = raw.get("animation_checklist")
    if not isinstance(payload, dict):
        payload = raw if any(b in raw for b in BUCKETS) else {}

    checklist = Checklist()
    # Duplicate labels are rejected only WITHIN a bucket. Across buckets they
    # are legitimate and the prompt says so explicitly: a node and a block may
    # share a label and are distinct entities. Rejecting a cross-bucket repeat
    # would delete a real element the animation is on the hook for.
    seen: dict[str, set[str]] = {b: set() for b in BUCKETS}
    any_kept = False

    for bucket in BUCKETS:
        values = payload.get(bucket)
        if values is None:
            continue
        if not isinstance(values, list):
            checklist.notes.append(f"{bucket}: expected a list, got "
                                   f"{type(values).__name__}; ignored")
            continue
        for value in values:
            if isinstance(value, str):
                label, frequency, source_ids = value.strip(), 1, []
            elif isinstance(value, dict):
                label = str(value.get("label") or "").strip()
                raw_freq = value.get("frequency", 1)
                try:
                    frequency = int(raw_freq)
                except (TypeError, ValueError):
                    checklist.notes.append(
                        f"{bucket}: {label!r} had a non-integer frequency "
                        f"{raw_freq!r}; treated as 1")
                    frequency = 1
                source_ids = [str(i) for i in (value.get("source_ids") or [])
                              if isinstance(i, (str, int))]
            else:
                checklist.notes.append(f"{bucket}: dropped a non-string, "
                                       f"non-object entry")
                continue

            if not label:
                checklist.notes.append(f"{bucket}: dropped an entry with no label")
                continue
            if frequency < 1:
                checklist.notes.append(f"{bucket}: {label!r} had frequency "
                                       f"{frequency}; treated as 1")
                frequency = 1
            # A frequency claiming more instances than ids the judge can name
            # is unfalsifiable padding -- every unnamed instance would count as
            # an omission forever. Trust the ids when they contradict it.
            if source_ids and frequency > len(source_ids):
                checklist.notes.append(
                    f"{bucket}: {label!r} claimed frequency {frequency} but "
                    f"named only {len(source_ids)} id(s); used {len(source_ids)}")
                frequency = len(source_ids)
            if label in seen[bucket]:
                checklist.notes.append(f"{bucket}: dropped duplicate {label!r}")
                continue
            seen[bucket].add(label)
            any_kept = True
            getattr(checklist, bucket).append(
                Entry(label=label, frequency=frequency, source_ids=source_ids))

    if not any_kept:
        checklist.notes.append("judge returned no usable checklist entries")
    return checklist


def build(judge, image_path: Path, code: str, seq_view: str) -> Checklist:
    """One judged call: figure + diagram code + sequence -> named elements."""
    prompt = load_and_render("animation_checklist.yaml#prompt", {
        "diagram_code": code,
        "sequence_view": seq_view,
    }, root=PROMPTS_ROOT)
    raw = judge.ask_json(prompt, images=[Path(image_path)], tag="animation_checklist")
    checklist = validate(raw)
    checklist.provenance = judge.provenance()
    return checklist


def checklist_path(root: Path, config_name: str, style: str, sample_id: str) -> Path:
    """Style-keyed: the checklist is derived from that style's own sequence."""
    return Path(root) / "checklist" / config_name / style / f"{sample_id}.json"
