"""Stage-2 XML metrics: PAA, edge precision/recall, depth consistency.

PAA and edge P/R compare the predicted structure XML against the reference
XML *through the element alignment* (see `alignment.py`): predicted ids are
first contracted onto ground-truth element groups, and only then compared.
Nothing here assumes ids match textually -- verified on pipe00041, where the
same figure yields `rgb1,rgb2,rgb3` in the reference and one `img_rgb_images`
in the prediction.

Depth consistency is deliberately alignment-free: it is a self-consistency
rule (every element's declared depth is one more than its parent's), so it
scores the prediction against its own declarations, per the design doc §2.1.4.
"""
from __future__ import annotations

from ..alignment import Alignment
from ..gt import DiagramXml


def depth_consistency(pred: DiagramXml) -> dict:
    """Fraction of nested elements violating depth == 1 + parent's depth.

    Only elements that *declare a parent* are checked, which is what §2.1.4
    actually specifies. Root children are deliberately exempt: the two XML
    conventions in this benchmark disagree about whether `<Diagram>` counts
    as a level, and worse, the reference bundle mixes both within one
    document -- its top-level blocks declare depth=2 while its top-level
    edges declare depth=1. Any rule that fixes a single base depth would
    therefore report the reference's own hand-authored XML as ~19% invalid,
    which says more about the rule than about the data.

    Checking the parent relation instead is convention-independent: it holds
    under either numbering, and it is the property that actually matters
    downstream, since depth is consumed relative to the hierarchy.
    """
    elements = pred.elements
    violations: list[dict] = []
    checked = 0

    for e in elements.values():
        if e.parent is None:
            continue
        parent = elements.get(e.parent)
        if parent is None:
            continue          # dangling parent is a different defect (PAA's)
        checked += 1
        if e.depth != parent.depth + 1:
            violations.append({"id": e.id, "declared": e.depth,
                               "expected": parent.depth + 1,
                               "reason": f"parent '{parent.id}' at {parent.depth}"})

    return {
        "depth_violation_rate": (len(violations) / checked) if checked else 0.0,
        "depth_violations": violations,
        "elements_checked": checked,
    }


def parent_assignment(gt: DiagramXml, pred: DiagramXml, alignment: Alignment) -> dict:
    """PAA over matched scorable elements, plus how much was matched at all.

    A ground-truth element counts correct when its matched prediction's parent
    contracts onto the same group as the ground-truth parent. Top-level
    elements must map to top-level elements. The match rate is reported
    separately: PAA over three matched elements out of forty is not the same
    result as PAA over thirty-eight, and a single number would hide that.
    """
    scorable = gt.scorable()
    detail: list[dict] = []
    correct = 0
    matched = 0

    for gt_id, gt_el in scorable.items():
        pred_id = alignment.pred_for(gt_id)
        if pred_id is None or pred_id not in pred.elements:
            detail.append({"gt": gt_id, "status": "unmatched"})
            continue
        matched += 1
        pred_el = pred.elements[pred_id]

        # Contract both parents onto GT groups and compare there. A parent
        # that is itself unmatched cannot be verified and counts wrong: the
        # prediction hung this element under structure the reference lacks.
        gt_parent_group = alignment.group_of_gt(gt_el.parent) if gt_el.parent else None
        pred_parent_group = (alignment.group_of_pred(pred_el.parent)
                             if pred_el.parent else None)

        ok = gt_parent_group == pred_parent_group
        correct += ok
        detail.append({
            "gt": gt_id, "pred": pred_id, "status": "ok" if ok else "wrong_parent",
            "gt_parent": gt_el.parent, "pred_parent": pred_el.parent,
        })

    return {
        "paa": (correct / matched) if matched else None,
        "matched_coverage": (matched / len(scorable)) if scorable else None,
        "matched": matched,
        "scorable_gt_elements": len(scorable),
        "parent_detail": detail,
    }


def edge_topology(gt: DiagramXml, pred: DiagramXml, alignment: Alignment) -> dict:
    """Edge precision/recall/F1 after contracting endpoints onto GT groups.

    An endpoint that does not contract (unmatched element) makes its edge
    uncomparable; those edges are excluded from the numerator and denominator
    of *precision* but still reported, because an edge to a hallucinated
    element is a different defect than a wrong connection between real ones.
    """
    gt_pairs = gt.edge_pairs()
    # GT edges whose endpoints are scorable at the group level.
    gt_set = {(alignment.group_of_gt(s) or s, alignment.group_of_gt(t) or t)
              for s, t in gt_pairs}

    pred_contracted: set[tuple[str, str]] = set()
    uncontractable: list[dict] = []
    for s, t in pred.edge_pairs():
        gs, gt_group = alignment.group_of_pred(s), alignment.group_of_pred(t)
        if gs is None or gt_group is None:
            uncontractable.append({"source": s, "target": t})
            continue
        pred_contracted.add((gs, gt_group))

    true_pos = pred_contracted & gt_set
    precision = (len(true_pos) / len(pred_contracted)) if pred_contracted else None
    recall = (len(true_pos) / len(gt_set)) if gt_set else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else
          (0.0 if precision is not None and recall is not None else None))

    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "gt_edges": len(gt_set),
        "pred_edges_comparable": len(pred_contracted),
        "pred_edges_uncontractable": uncontractable,
        "matched_edges": sorted(true_pos),
        "missed_gt_edges": sorted(gt_set - pred_contracted),
        "spurious_pred_edges": sorted(pred_contracted - gt_set),
    }


def run(gt: DiagramXml, pred: DiagramXml, alignment: Alignment) -> dict:
    record = {"suite": "xml"}
    record.update(depth_consistency(pred))
    record.update(parent_assignment(gt, pred, alignment))
    record.update(edge_topology(gt, pred, alignment))
    return record
