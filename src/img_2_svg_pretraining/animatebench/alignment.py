"""GT <-> prediction element alignment: judged once, validated, reused.

Every ground-truth-dependent metric needs to know which predicted element is
"the same thing" as which reference element. Ids never match textually, and
the correspondence is not 1:1 -- on pipe00041 the reference draws three RGB
thumbnails (`rgb1..rgb3`) where the prediction has one `img_rgb_images`, and
the reference's `gear_icon` has no predicted counterpart at all. So the
alignment is a set of **groups**, each holding one or more GT ids and one or
more predicted ids that denote the same visual entity, plus explicit
unmatched lists on both sides.

The judge (a VLM, sees both XMLs and the figure) proposes the grouping ONCE
per (candidate, sample); the result is validated mechanically -- hallucinated
ids are dropped, duplicates rejected -- and cached to disk. All metrics then
consume the same artifact, so PAA, edge P/R and sequence coverage cannot
disagree about element identity. Judge misbehaviour degrades match coverage
(reported), never corrupts a score silently.

`composite_part` elements are excluded on both sides before the judge ever
sees the ids: the design doc (§2.1.3) excludes sub-part decomposition from
cross-method scoring because it is inherently language- and model-dependent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .gt import DiagramXml


@dataclass
class Alignment:
    """Validated element correspondence between one GT and one predicted XML."""

    # group key -> {"gt": [...], "pred": [...]}
    groups: dict[str, dict] = field(default_factory=dict)
    gt_unmatched: list[str] = field(default_factory=list)
    pred_unmatched: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)      # validation findings
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self._gt_to_group: dict[str, str] = {}
        self._pred_to_group: dict[str, str] = {}
        for key, group in self.groups.items():
            for g in group.get("gt", []):
                self._gt_to_group[g] = key
            for p in group.get("pred", []):
                self._pred_to_group[p] = key

    # -- lookups ----------------------------------------------------------

    def group_of_gt(self, gt_id: str | None) -> str | None:
        return self._gt_to_group.get(gt_id) if gt_id else None

    def group_of_pred(self, pred_id: str | None) -> str | None:
        return self._pred_to_group.get(pred_id) if pred_id else None

    def pred_for(self, gt_id: str) -> str | None:
        """A representative predicted id for a GT element, or None."""
        key = self._gt_to_group.get(gt_id)
        if key is None:
            return None
        preds = self.groups[key].get("pred", [])
        return preds[0] if preds else None

    def matched_gt(self) -> set[str]:
        return set(self._gt_to_group)

    def matched_pred(self) -> set[str]:
        return set(self._pred_to_group)

    # -- io ---------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"groups": self.groups, "gt_unmatched": self.gt_unmatched,
                "pred_unmatched": self.pred_unmatched, "notes": self.notes,
                "provenance": self.provenance}

    @classmethod
    def from_dict(cls, data: dict) -> "Alignment":
        return cls(groups=data.get("groups") or {},
                   gt_unmatched=data.get("gt_unmatched") or [],
                   pred_unmatched=data.get("pred_unmatched") or [],
                   notes=data.get("notes") or [],
                   provenance=data.get("provenance") or {})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Alignment":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _alignable_ids(xml: DiagramXml) -> set[str]:
    """Ids the judge is allowed to use: everything except composite parts."""
    return {i for i, e in xml.elements.items() if e.cls != "composite_part"}


def validate(raw: dict, gt: DiagramXml, pred: DiagramXml) -> Alignment:
    """Turn the judge's proposal into a validated Alignment.

    Enforcement, not trust: an id that is not in its XML is dropped with a
    note; an id appearing in two groups keeps its first placement. Whatever
    survives is internally consistent by construction, so the metrics never
    need to re-check.
    """
    gt_ok, pred_ok = _alignable_ids(gt), _alignable_ids(pred)
    notes: list[str] = []
    groups: dict[str, dict] = {}
    seen_gt: set[str] = set()
    seen_pred: set[str] = set()

    for index, entry in enumerate(raw.get("groups") or []):
        if not isinstance(entry, dict):
            continue
        gt_ids, pred_ids = [], []
        for g in entry.get("gt") or []:
            g = str(g)
            if g not in gt_ok:
                notes.append(f"dropped GT id '{g}': not in reference XML")
            elif g in seen_gt:
                notes.append(f"dropped GT id '{g}': already in another group")
            else:
                gt_ids.append(g); seen_gt.add(g)
        for p in entry.get("pred") or []:
            p = str(p)
            if p not in pred_ok:
                notes.append(f"dropped pred id '{p}': not in predicted XML")
            elif p in seen_pred:
                notes.append(f"dropped pred id '{p}': already in another group")
            else:
                pred_ids.append(p); seen_pred.add(p)
        if gt_ids and pred_ids:
            groups[f"g{index:03d}"] = {"gt": gt_ids, "pred": pred_ids,
                                       "label": str(entry.get("label", ""))}
        elif gt_ids or pred_ids:
            notes.append(f"group {index} one-sided after validation; discarded")

    return Alignment(
        groups=groups,
        gt_unmatched=sorted(gt_ok - seen_gt),
        pred_unmatched=sorted(pred_ok - seen_pred),
        notes=notes,
    )


def _xml_listing(xml: DiagramXml) -> str:
    """The element table as the judge sees it: id, class, parent."""
    lines = []
    for el in xml.elements.values():
        if el.cls == "composite_part":
            continue
        extra = f" source={el.source} target={el.target}" if el.cls == "edge" else ""
        lines.append(f"  {el.id}  [{el.cls}]  parent={el.parent or 'root'}{extra}")
    return "\n".join(lines)


def align(judge, gt: DiagramXml, pred: DiagramXml, image_path: Path) -> Alignment:
    """One judge call producing the full grouping, then validation."""
    from img_2_svg_pretraining.pipeline.prompts import load_and_render

    from .judge import PROMPTS_ROOT

    prompt = load_and_render("alignment.yaml#prompt", {
        "gt_elements": _xml_listing(gt),
        "pred_elements": _xml_listing(pred),
    }, root=PROMPTS_ROOT)

    raw = judge.ask_json(prompt, images=[image_path])
    alignment = validate(raw, gt, pred)
    alignment.provenance = judge.provenance()
    return alignment
