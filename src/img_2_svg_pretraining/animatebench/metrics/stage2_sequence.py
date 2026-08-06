"""Stage-2 sequence metrics: coverage, TOF, SSCR, DOVR.

Coverage compares against the reference sequence through the element
alignment (same artifact the XML metrics use). TOF, SSCR and DOVR are
self-contained: they score the predicted sequence against its own XML and
the style contract, so they run without a judge and without ground truth.
"""
from __future__ import annotations

from img_2_svg_pretraining.pipeline.schema import AnimationSequence
from img_2_svg_pretraining.pipeline.styles import STYLES, check_style

from ..alignment import Alignment
from ..gt import DiagramXml, first_appearance, sequence_elements

# Styles whose contract animates only blocks and nodes; text/arrows are
# excluded from coverage there (evaluation doc §2.1.8) and must be EMPTY in
# every step (§2.1.9 SSCR).
BBOX_STYLES = ("hopping_bounding_box", "sliding_bounding_box")


def _covered_ids(seq: AnimationSequence, style: str) -> set[str]:
    buckets = sequence_elements(seq)
    if style in BBOX_STYLES:
        return buckets["blocks"] | buckets["nodes"]
    return set().union(*buckets.values())


def coverage(gt_seq: AnimationSequence, pred_seq: AnimationSequence,
             alignment: Alignment, style: str) -> dict:
    """Precision/recall of animated elements, matched through the alignment.

    Recall: of the GT-sequence elements that have *any* alignment (are
    matchable at all), how many does the prediction animate somewhere?
    Precision: of the predicted animated elements that are matchable, how
    many correspond to something GT animates? Unmatchable ids on either side
    are reported but kept out of both ratios -- they are structure-XML gaps
    (already scored by matched_coverage), not sequencing mistakes.
    """
    gt_ids = _covered_ids(gt_seq, style)
    pred_ids = _covered_ids(pred_seq, style)

    gt_groups = {alignment.group_of_gt(i) for i in gt_ids} - {None}
    pred_groups = {alignment.group_of_pred(i) for i in pred_ids} - {None}

    hit = gt_groups & pred_groups
    recall = (len(hit) / len(gt_groups)) if gt_groups else None
    precision = (len(hit) / len(pred_groups)) if pred_groups else None

    return {
        "coverage_precision": precision,
        "coverage_recall": recall,
        "gt_animated": len(gt_ids),
        "gt_animated_matchable": len(gt_groups),
        "pred_animated": len(pred_ids),
        "pred_animated_matchable": len(pred_groups),
        "gt_animated_unmatchable": sorted(
            i for i in gt_ids if alignment.group_of_gt(i) is None),
        "pred_animated_unmatchable": sorted(
            i for i in pred_ids if alignment.group_of_pred(i) is None),
        "covered_groups": sorted(hit),
        "missed_groups": sorted(gt_groups - pred_groups),
    }


def traversal_order_fidelity(pred_seq: AnimationSequence, pred_xml: DiagramXml) -> dict:
    """Does the realized order match the declared traversal style? (§2.1.8)

    overview_first: every top-level element's first appearance must precede
    every deeper element's first appearance -- score is the fraction of
    (top, deep) pairs correctly ordered.

    detail_first: elements belonging to one top-level block should be visited
    contiguously -- score is 1 minus the fraction of "block switches" beyond
    the minimum possible (a perfect run switches blocks exactly n_blocks-1
    times; every extra switch is a return to an already-visited block).
    """
    appear = first_appearance(pred_seq)
    if not appear:
        return {"tof": None, "tof_detail": "sequence animates no known elements"}

    def top_ancestor(el_id: str) -> str | None:
        el = pred_xml.elements.get(el_id)
        seen = set()
        while el is not None and el.parent is not None and el.parent not in seen:
            seen.add(el.id)
            el = pred_xml.elements.get(el.parent)
        return el.id if el is not None else None

    style = (pred_seq.traversal_style or "").lower()

    if style == "overview_first":
        tops = [i for i in appear if pred_xml.elements.get(i) is not None
                and pred_xml.elements[i].parent is None]
        deeps = [i for i in appear if pred_xml.elements.get(i) is not None
                 and pred_xml.elements[i].parent is not None]
        pairs = [(t, d) for t in tops for d in deeps]
        if not pairs:
            return {"tof": None, "tof_detail": "no top/deep pairs to order"}
        good = sum(appear[t] <= appear[d] for t, d in pairs)
        return {"tof": good / len(pairs),
                "tof_detail": f"{good}/{len(pairs)} (top-level, deeper) pairs "
                              f"front-loaded correctly"}

    # detail_first (and the default when the sequence doesn't say)
    ordered = sorted(appear.items(), key=lambda kv: kv[1])
    block_walk = []
    for el_id, _ in ordered:
        top = top_ancestor(el_id)
        if top is not None and (not block_walk or block_walk[-1] != top):
            block_walk.append(top)
    distinct = len(set(block_walk))
    if distinct <= 1:
        return {"tof": 1.0, "tof_detail": "single block; trivially contiguous"}
    switches = len(block_walk) - 1
    minimum = distinct - 1
    extra = switches - minimum
    # Each revisit of an already-finished block is one extra switch.
    score = max(0.0, 1.0 - extra / max(1, switches))
    return {"tof": score,
            "tof_detail": f"block walk {block_walk}: {switches} switches for "
                          f"{distinct} blocks (min {minimum})"}


def style_schema_compliance(pred_seq: AnimationSequence, style: str) -> dict:
    """SSCR (§2.1.9): hard pass/fail on the style's schema contract.

    Scored on the JSON contract the benchmark actually defines -- which
    element buckets may be non-empty, and how many targets a step may carry --
    not on our internal `action` vocabulary. The bench dialect has no action
    field at all, so `AnimationSequence` defaults every step to "reveal";
    running `check_style` unfiltered would then fail every bench-format
    bounding-box sequence for using an action the source never specified.
    Action checks are therefore applied only to sequences that genuinely
    declare actions (our native dialect).
    """
    violations: list[str] = []

    native = not any(n.element_classes for n in pred_seq.nodes)
    if native and style in STYLES:
        violations.extend(check_style(pred_seq, style))
    elif style in STYLES:
        # Bench dialect: keep the structural half of the style check (element
        # repetition rules), drop anything phrased in terms of actions.
        violations.extend(v for v in check_style(pred_seq, style)
                          if "action" not in v)

    if style in BBOX_STYLES:
        for node in pred_seq.nodes:
            classes = node.element_classes or {}
            for bucket in ("text", "arrows"):
                if classes.get(bucket):
                    violations.append(
                        f"step '{node.id}': {bucket} must be empty for {style}, "
                        f"has {classes[bucket]}")
            # "EXACTLY ONE element per timestamp" (Ruleset 3, Rule B).
            targets = len(classes.get("blocks", [])) + len(classes.get("nodes", []))
            if targets > 1:
                violations.append(
                    f"step '{node.id}': {style} animates exactly one block or "
                    f"node per timestamp, found {targets}")

    return {"sscr_pass": not violations, "sscr_violations": violations}


def dependence_order(pred_seq: AnimationSequence, pred_xml: DiagramXml) -> dict:
    """DOVR (§2.1.9): arrows revealed before both endpoints exist on screen.

    Endpoints come from the prediction's own XML -- this is a self-consistency
    property, not a GT comparison. Edges whose endpoints never appear in the
    sequence at all cannot be tested and are reported separately.
    """
    appear = first_appearance(pred_seq)
    violations: list[dict] = []
    untestable: list[str] = []
    tested = 0

    for edge_id, edge in pred_xml.edges().items():
        t_edge = appear.get(edge_id)
        if t_edge is None:
            continue   # edge never animated; nothing to order
        t_src = appear.get(edge.source) if edge.source else None
        t_tgt = appear.get(edge.target) if edge.target else None
        if t_src is None or t_tgt is None:
            untestable.append(edge_id)
            continue
        tested += 1
        need = max(t_src, t_tgt)
        if t_edge < need:
            violations.append({"edge": edge_id, "at": t_edge,
                               "source_at": t_src, "target_at": t_tgt})

    return {
        "dovr": (len(violations) / tested) if tested else None,
        "dovr_violations": violations,
        "edges_tested": tested,
        "edges_untestable": untestable,
    }


def run(gt_seq: AnimationSequence | None, pred_seq: AnimationSequence,
        pred_xml: DiagramXml, alignment: Alignment | None, style: str) -> dict:
    record = {"suite": "sequence", "style": style}
    if gt_seq is not None and alignment is not None:
        record.update(coverage(gt_seq, pred_seq, alignment, style))
    else:
        record["coverage_skipped"] = ("no GT sequence for this style"
                                      if gt_seq is None else "no alignment")
    record.update(traversal_order_fidelity(pred_seq, pred_xml))
    record.update(style_schema_compliance(pred_seq, style))
    record.update(dependence_order(pred_seq, pred_xml))
    return record
