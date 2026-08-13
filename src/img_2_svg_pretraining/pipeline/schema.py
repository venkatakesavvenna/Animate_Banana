"""The animation sequence: the contract between Stage 2, Stage 3, and the benchmark.

Shape follows section 4.3 of the AnimateBench design document -- a flat node
list carrying parent links (the hierarchy) plus an explicit traversal (the
playback order). Keeping the two separate is what lets two systems identify
the same semantic units while choosing different valid orders.

`focus` entries are `xml id` values from the transmuted diagram code, which
is what makes element-reference validity mechanically checkable against the
parsed XML rather than a matter of judgment.

`validate()` returns the same violation strings the benchmark's validity
metrics report, so the planner critic and the scorer agree on what "invalid"
means instead of implementing it twice.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .styles import ALL_ACTIONS, check_style

TRAVERSAL_STYLES = ("OVERVIEW_FIRST", "DETAIL_FIRST")


@dataclass
class SequenceNode:
    id: str
    parent: str | None = None
    depth: int = 1
    focus: list[str] = field(default_factory=list)
    action: str = "reveal"
    narration: str | None = None
    duration: float | None = None
    # Stage 2e. `narration` is the sequencer's one-line gloss of what the step
    # shows; `narrative` is the spoken script written against the paper (or the
    # figure alone) and is what gets sent to TTS. They are kept separate so a
    # re-run of the narrative writer never destroys the planner's own reasoning,
    # and so the benchmark can tell a planned step apart from a narrated one.
    narrative: str | None = None

    # AnimateBench dialect only. `focus` flattens the four element buckets,
    # which is what most consumers want; these keep the split so a sequence
    # read from the bench schema can be written back to it unchanged. The
    # bounding-box styles are defined partly by `text`/`arrows` being empty,
    # so the distinction is load-bearing and cannot be recomputed from `focus`.
    element_classes: dict[str, list[str]] = field(default_factory=dict)
    timestamp: int | None = None


@dataclass
class AnimationSequence:
    style: str
    traversal_style: str | None = None
    nodes: list[SequenceNode] = field(default_factory=list)
    traversal: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "style": self.style,
            "traversal_style": self.traversal_style,
            "nodes": [asdict(n) for n in self.nodes],
            "traversal": list(self.traversal),
            "provenance": self.provenance,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "AnimationSequence":
        if not isinstance(data, dict):
            raise ValueError("animation sequence must be a JSON object")

        # AnimateBench dialect, which the tikz_/svg_sequencer prompts emit.
        if "sequence" in data and "nodes" not in data:
            return cls._from_bench(data)

        nodes = []
        for raw in data.get("nodes") or []:
            if not isinstance(raw, dict) or "id" not in raw:
                raise ValueError(f"malformed node (needs at least an 'id'): {raw!r}")
            focus = raw.get("focus") or []
            if isinstance(focus, str):  # tolerate a single id given unwrapped
                focus = [focus]
            nodes.append(SequenceNode(
                id=str(raw["id"]),
                parent=raw.get("parent"),
                depth=int(raw.get("depth", 1) or 1),
                focus=[str(f) for f in focus],
                action=raw.get("action") or "reveal",
                narration=raw.get("narration"),
                duration=raw.get("duration"),
                narrative=raw.get("narrative"),
                # Present when this sequence originated in the bench dialect
                # and was written back out in ours. Without reading them here,
                # a save/load cycle silently drops the bucket split and the
                # sequence stops being convertible back to bench format.
                element_classes={k: [str(i) for i in v]
                                 for k, v in (raw.get("element_classes") or {}).items()},
                timestamp=raw.get("timestamp"),
            ))
        return cls(
            style=data.get("style") or "",
            traversal_style=data.get("traversal_style"),
            nodes=nodes,
            traversal=[str(t) for t in (data.get("traversal") or [])],
            provenance=data.get("provenance") or {},
        )

    # -- AnimateBench dialect --------------------------------------------
    #
    # The benchmark (and the tikz_/svg_sequencer prompts) express a sequence as
    # a flat, time-indexed list:
    #
    #   {"metadata": {"traversal_order": ..., "animation_style": ...},
    #    "sequence": [{"timestamp": 1, "narrative": ...,
    #                  "to_be_animated": {"blocks": [], "nodes": [],
    #                                     "text": [], "arrows": []}}]}
    #
    # Ours is a hierarchy plus a traversal order. The two carry the same
    # information for our purposes: one timestamp becomes one node, and the
    # union of its four element buckets becomes that node's `focus`. The
    # buckets are kept verbatim in `element_classes` so converting back is
    # lossless -- which matters because the bench's bounding-box styles are
    # defined partly in terms of the text/arrows buckets being empty, and
    # collapsing them to a flat focus list would erase that distinction.

    _BUCKETS = ("blocks", "nodes", "text", "arrows")

    @classmethod
    def _from_bench(cls, data: dict) -> "AnimationSequence":
        meta = data.get("metadata") or {}
        traversal_style = meta.get("traversal_order")
        if isinstance(traversal_style, str):
            traversal_style = traversal_style.upper()

        nodes: list[SequenceNode] = []
        for index, step in enumerate(data.get("sequence") or [], 1):
            if not isinstance(step, dict):
                continue
            timestamp = step.get("timestamp", index)
            animated = step.get("to_be_animated") or {}

            focus: list[str] = []
            classes: dict[str, list[str]] = {}
            depths: list[int] = []
            for bucket in cls._BUCKETS:
                ids = []
                for entry in animated.get(bucket) or []:
                    if isinstance(entry, dict) and entry.get("id") is not None:
                        ids.append(str(entry["id"]))
                        if isinstance(entry.get("depth"), int):
                            depths.append(entry["depth"])
                    elif isinstance(entry, str):
                        ids.append(entry)
                classes[bucket] = ids
                focus.extend(ids)

            nodes.append(SequenceNode(
                id=f"t{timestamp}",
                # Parent is filled in below: it follows from the depth ladder,
                # which is only knowable once every step has been read.
                parent=None,
                # The bench carries depth per element, not per step; the
                # shallowest element is the step's own level.
                depth=min(depths) if depths else 1,
                focus=focus,
                action="reveal",
                narrative=step.get("narrative"),
                duration=step.get("duration"),
                element_classes=classes,
                timestamp=timestamp,
            ))

        cls._link_bench_parents(nodes)

        return cls(
            style=meta.get("animation_style") or data.get("style") or "",
            traversal_style=traversal_style,
            nodes=nodes,
            traversal=[n.id for n in nodes],
            provenance=data.get("provenance") or {},
        )

    @staticmethod
    def _link_bench_parents(nodes: list["SequenceNode"]) -> None:
        """Recover step parentage from the depth ladder, in place.

        The bench dialect has no per-step `parent`: its steps are timestamps,
        and the hierarchy lives in the *elements* each step reveals. Copying
        those element depths onto the step while leaving `parent` unset made
        every nested step look like a root sitting at depth 2 -- twelve of
        pipe00137's fifteen steps, all reported as "root node 't2' has depth 2,
        expected 1" when nothing was actually wrong with them.

        Playback order is the hierarchy: a step at depth N belongs to the most
        recent step at depth N-1, exactly as an indented outline nests. That is
        the same reading the prompt describes -- animate a parent block, then
        its internals -- so this recovers what the dialect implies rather than
        inventing structure.
        """
        # Deepest ancestor seen at each level, as playback walks forward.
        open_at_depth: dict[int, str] = {}
        for node in nodes:
            depth = node.depth or 1
            parent = open_at_depth.get(depth - 1)
            if depth > 1 and parent:
                node.parent = parent
            elif depth > 1:
                # Nothing shallower has been seen yet, so this cannot be a
                # child of anything. Treat it as the root it actually is
                # rather than leaving an unsatisfiable depth behind.
                node.depth = 1
            open_at_depth[node.depth] = node.id
            # A new node at this level closes anything deeper that was open.
            for deeper in [d for d in open_at_depth if d > node.depth]:
                open_at_depth.pop(deeper, None)

    def relink_orphan_depths(self) -> int:
        """Repair steps that claim a depth but no parent. Returns how many.

        The bench->native conversion used to drop step parentage, and those
        sequences are already on disk in native form -- so re-reading them will
        never re-run the conversion that now gets it right. This applies the
        same depth-ladder reading to an existing sequence, which is what lets a
        stale artifact be repaired without regenerating it from the model.

        Only touches nodes that are inconsistent (depth > 1 with no parent);
        a sequence whose hierarchy is already stated survives untouched.
        """
        orphaned = [n for n in self.nodes if (n.depth or 1) > 1 and not n.parent]
        if not orphaned:
            return 0
        before = [(n.id, n.parent, n.depth) for n in self.nodes]
        self._link_bench_parents(self.nodes)
        return sum(1 for b, n in zip(before, self.nodes)
                   if (n.parent, n.depth) != (b[1], b[2]))

    def to_bench_dict(self) -> dict:
        """Serialize in the AnimateBench dialect, for direct comparison.

        Steps come out in traversal order, which is playback order -- the same
        thing the bench's `timestamp` means.
        """
        by_id = {n.id: n for n in self.nodes}
        sequence = []
        for index, node_id in enumerate(self.traversal, 1):
            node = by_id.get(node_id)
            if node is None:
                continue
            classes = node.element_classes or {}
            if not classes:
                # A sequence authored in our own dialect has no bucket
                # information; everything it focuses is reported as a node,
                # which is the bucket the renderer treats most generally.
                classes = {"blocks": [], "nodes": list(node.focus),
                           "text": [], "arrows": []}
            sequence.append({
                "timestamp": node.timestamp if node.timestamp is not None else index,
                "narrative": node.narrative,
                "to_be_animated": {
                    bucket: [{"id": i} for i in classes.get(bucket, [])]
                    for bucket in self._BUCKETS
                },
            })
        return {
            "metadata": {
                "traversal_order": (self.traversal_style or "").lower() or None,
                "animation_style": self.style,
            },
            "sequence": sequence,
        }

    def to_bench_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_bench_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "AnimationSequence":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: Path) -> "AnimationSequence":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(path)

    # -- queries ---------------------------------------------------------

    def node_by_id(self, node_id: str) -> SequenceNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def focus_ids(self) -> set[str]:
        return {f for n in self.nodes for f in n.focus}

    def max_depth(self) -> int:
        return max((n.depth for n in self.nodes), default=0)

    def has_narrative(self) -> bool:
        """Whether the narrative writer has run and produced any spoken text."""
        return any(n.narrative for n in self.nodes)

    def narration_script(self) -> list[dict]:
        """The spoken script, one entry per playback step, in traversal order.

        Timestamps are 1-based and index the *traversal*, not the node list --
        the animation renders one frame per traversal entry, so this is the
        numbering the frames and the audio clips must agree on. A step whose
        narrative is null still gets an entry with `text: None`; dropping it
        would shift every later timestamp and desync the audio from the video.
        """
        by_id = {n.id: n for n in self.nodes}
        script = []
        for index, node_id in enumerate(self.traversal, 1):
            node = by_id.get(node_id)
            script.append({
                "timestamp": index,
                "node_id": node_id,
                "text": node.narrative if node else None,
                "duration": node.duration if node else None,
            })
        return script

    def narration_jsonl(self) -> str:
        """The script as timestamp-indexed JSONL, the TTS stage's input format."""
        return "".join(
            json.dumps({"timestamp": e["timestamp"], "text": e["text"]}) + "\n"
            for e in self.narration_script()
        )

    # -- validation ------------------------------------------------------

    def validate(self, element_ids: set[str] | None = None,
                 max_depth: int | None = None) -> list[str]:
        """Structural problems with this sequence, worst-first.

        `element_ids` is the set of ids present in the parsed diagram XML;
        when given, every `focus` entry is checked against it. `max_depth`
        enforces the requested explanation depth.
        """
        problems: list[str] = []

        if not self.nodes:
            problems.append("sequence has no nodes")
            return problems

        ids = [n.id for n in self.nodes]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            problems.append(f"duplicate node id(s): {duplicates}")
        id_set = set(ids)

        # parent references
        for node in self.nodes:
            if node.parent is not None and node.parent not in id_set:
                problems.append(f"node '{node.id}' has unknown parent '{node.parent}'")
            if node.parent == node.id:
                problems.append(f"node '{node.id}' is its own parent")

        problems.extend(self._cycles(id_set))

        # depth consistency: a child sits exactly one level below its parent.
        by_id = {n.id: n for n in self.nodes}
        for node in self.nodes:
            if node.parent and node.parent in by_id:
                expected = by_id[node.parent].depth + 1
                if node.depth != expected:
                    problems.append(
                        f"node '{node.id}' has depth {node.depth} but parent "
                        f"'{node.parent}' is at depth {by_id[node.parent].depth} "
                        f"(expected {expected})"
                    )
            elif node.parent is None and node.depth != 1:
                problems.append(f"root node '{node.id}' has depth {node.depth}, expected 1")

        if max_depth is not None and self.max_depth() > max_depth:
            problems.append(
                f"sequence reaches depth {self.max_depth()}, exceeding the "
                f"requested maximum of {max_depth}"
            )

        # traversal
        if not self.traversal:
            problems.append("sequence has an empty traversal")
        unknown = sorted({t for t in self.traversal if t not in id_set})
        if unknown:
            problems.append(f"traversal references unknown node(s): {unknown}")
        unvisited = sorted(id_set - set(self.traversal))
        if unvisited:
            problems.append(f"node(s) never visited by the traversal: {unvisited}")

        # actions
        bad_actions = sorted({n.action for n in self.nodes
                              if n.action and n.action not in ALL_ACTIONS})
        if bad_actions:
            problems.append(f"unknown action(s): {bad_actions} (known: {list(ALL_ACTIONS)})")

        # element references
        if element_ids is not None:
            dangling = sorted(self.focus_ids() - element_ids)
            if dangling:
                problems.append(
                    f"focus references element(s) absent from the diagram: {dangling}"
                )

        if self.traversal_style and self.traversal_style not in TRAVERSAL_STYLES:
            problems.append(
                f"unknown traversal_style '{self.traversal_style}'. "
                f"Known: {list(TRAVERSAL_STYLES)}"
            )

        # style constraints
        if self.style:
            try:
                problems.extend(check_style(self))
            except KeyError as e:
                problems.append(str(e))

        return problems

    def _cycles(self, id_set: set[str]) -> list[str]:
        by_id = {n.id: n for n in self.nodes}
        problems = []
        for start in id_set:
            seen = set()
            current = start
            while current is not None and current in by_id:
                if current in seen:
                    problems.append(f"cycle in hierarchy reachable from '{start}'")
                    break
                seen.add(current)
                current = by_id[current].parent
        # One report per cycle is enough; dedupe the per-start reports.
        return sorted(set(problems))[:1]

    def is_valid(self, element_ids: set[str] | None = None,
                 max_depth: int | None = None) -> bool:
        return not self.validate(element_ids=element_ids, max_depth=max_depth)
