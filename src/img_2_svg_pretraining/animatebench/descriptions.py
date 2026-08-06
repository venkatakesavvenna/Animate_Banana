"""Human-readable descriptions of every metric, for display in viewers.

Kept beside the metric implementations rather than in the frontend so the
wording and the computation cannot drift apart: if a metric's meaning changes,
the description sits in the same package and is edited with it.

Each entry carries what the number means, which way is better, and -- where
the metric has a real caveat -- what it cannot tell you. A reader looking at
a dashboard has no other way to know that `paa` over three matched elements
is a different claim from `paa` over forty.
"""
from __future__ import annotations

# metric key -> (label, one-line meaning, direction, caveat)
METRICS: dict[str, dict] = {
    # -- stage 1 -----------------------------------------------------------
    "csr": {
        "label": "Compilation Success Rate",
        "stage": "Stage 1 · diagram code",
        "doc": "§2.3.1",
        "what": "Does the generated diagram code compile at all?",
        "better": "higher",
        "range": "0 or 1 per sample",
        "caveat": "Compiling says nothing about whether the figure is correct — "
                  "a document can compile cleanly and render almost empty.",
    },
    "component_accuracy": {
        "label": "Diagram Component Score",
        "stage": "Stage 1 · diagram code",
        "doc": "§2.1.3",
        "what": "Fraction of elements whose semantic class tag (block, node, "
                "text, raster, edge…) a judge agrees with, read off the code.",
        "better": "higher",
        "range": "0–1",
        "caveat": "Judged on the code rather than the induced XML, because the "
                  "tags originate at the diagram-to-code step.",
    },
    "rendering_fidelity": {
        "label": "Rendering Fidelity",
        "stage": "Stage 1 · diagram code",
        "doc": "§2.3.1 / §2.2.1",
        "what": "How faithfully the compiled render reproduces the source "
                "figure, weighting layout and containment above styling.",
        "better": "higher",
        "range": "0–1",
        "caveat": "Scored 0 when the code does not compile. This is the only "
                  "Stage-1 check that can see a defect the compiler cannot.",
    },
    "code_quality": {
        "label": "Diagram Code Quality",
        "stage": "Stage 1 · diagram code",
        "doc": "§2.3.3",
        "what": "Readability, idiomatic TikZ (relative positioning, \\foreach, "
                "reusable styles) and composite complexity.",
        "better": "higher",
        "range": "0–1",
        "caveat": "Style only — correctness is measured elsewhere.",
    },

    # -- stage 2, XML ------------------------------------------------------
    "paa": {
        "label": "Parent-Assignment Accuracy",
        "stage": "Stage 2 · structure XML",
        "doc": "§2.1.4",
        "what": "Of the elements matched to the reference, the fraction whose "
                "parent is the corresponding reference parent.",
        "better": "higher",
        "range": "0–1",
        "caveat": "Read together with match coverage: a perfect score over "
                  "three matched elements is not the same claim as over forty.",
    },
    "matched_coverage": {
        "label": "Match Coverage",
        "stage": "Stage 2 · structure XML",
        "doc": "§2.1.4",
        "what": "Fraction of reference elements that could be matched to a "
                "predicted element at all.",
        "better": "higher",
        "range": "0–1",
        "caveat": "Low coverage means the two decompositions disagree about "
                  "what the figure contains, which limits every GT metric.",
    },
    "edge_f1": {
        "label": "Edge Topology F1",
        "stage": "Stage 2 · structure XML",
        "doc": "§2.1.4",
        "what": "Do the arrows connect the same things as in the reference? "
                "Endpoints are mapped through the element alignment first.",
        "better": "higher",
        "range": "0–1",
        "caveat": "Edges internal to a merged group become self-loops and "
                  "cannot be matched; those are reported separately.",
    },
    "depth_violation_rate": {
        "label": "Depth Violations",
        "stage": "Stage 2 · structure XML",
        "doc": "§2.1.4",
        "what": "Fraction of nested elements whose declared depth is not one "
                "more than their parent's.",
        "better": "lower",
        "range": "0–1",
        "caveat": "Checks the parent relation only, so it is independent of "
                  "whether a document counts the root as a level.",
    },

    # -- stage 2, sequence -------------------------------------------------
    "coverage_recall": {
        "label": "Element Coverage (recall)",
        "stage": "Stage 2 · animation sequence",
        "doc": "§2.1.8",
        "what": "Of the elements the reference animation touches, how many "
                "does this sequence animate somewhere?",
        "better": "higher",
        "range": "0–1",
        "caveat": "Style-aware: for bounding-box styles only blocks and nodes "
                  "count, since text and arrows are out of contract.",
    },
    "coverage_precision": {
        "label": "Element Coverage (precision)",
        "stage": "Stage 2 · animation sequence",
        "doc": "§2.1.8",
        "what": "Of the elements this sequence animates, how many correspond "
                "to something the reference also animates?",
        "better": "higher",
        "range": "0–1",
        "caveat": "Catches hallucinated entries — steps referencing elements "
                  "the figure does not contain.",
    },
    "tof": {
        "label": "Traversal-Order Fidelity",
        "stage": "Stage 2 · animation sequence",
        "doc": "§2.1.8",
        "what": "Does the realized order match the declared traversal style? "
                "Overview-first should front-load top-level elements; "
                "detail-first should finish one block before starting the next.",
        "better": "higher",
        "range": "0–1",
        "caveat": "Measured against the sequence's own declared style, not "
                  "against the reference's.",
    },
    "dovr": {
        "label": "Dependence-Order Violations",
        "stage": "Stage 2 · animation sequence",
        "doc": "§2.1.9",
        "what": "Fraction of arrows revealed before both the elements they "
                "connect have appeared.",
        "better": "lower",
        "range": "0–1",
        "caveat": "An arrow appearing before its endpoints reads as a line to "
                  "nowhere. Endpoints come from the run's own XML.",
    },
    "sscr_pass": {
        "label": "Style-Schema Compliance",
        "stage": "Stage 2 · animation sequence",
        "doc": "§2.1.9",
        "what": "Does the sequence obey its style's hard contract — e.g. "
                "bounding-box styles animate exactly one element per step and "
                "leave text and arrows empty?",
        "better": "pass",
        "range": "pass / fail",
        "caveat": "A binary contract, reported rather than averaged.",
    },

    # -- stage 3 -----------------------------------------------------------
    "anim_csr": {
        "label": "Animation Compilation Rate",
        "stage": "Stage 3 · animation code",
        "doc": "§2.3.1",
        "what": "Does the animation code compile to a multi-page PDF?",
        "better": "higher",
        "range": "0 or 1 per sample",
        "caveat": "Reported separately from diagram CSR: a diagram that "
                  "compiles can still fail once animation is layered on.",
    },
    "anim_code_quality": {
        "label": "Animation Code Quality",
        "stage": "Stage 3 · animation code",
        "doc": "§2.3.3",
        "what": "Readability of the frame logic and whether the effect is "
                "achieved by the simplest correct implementation.",
        "better": "higher",
        "range": "0–1",
    },
    "aif": {
        "label": "Animation Integration Footprint",
        "stage": "Stage 3 · animation code",
        "doc": "§2.3.3",
        "what": "Diagram lines rewritten or deleted, divided by animation "
                "lines added. Low means animation was a clean additive layer.",
        "better": "lower",
        "range": "0 upward",
        "caveat": "0 means the diagram body was untouched; above 1 means more "
                  "existing code was disturbed than animation code added.",
    },
}

# Display order within each suite, so the viewer reads top-down by stage.
SUITE_ORDER: dict[str, list[str]] = {
    "stage1": ["csr", "rendering_fidelity", "component_accuracy", "code_quality"],
    "xml": ["paa", "matched_coverage", "edge_f1", "depth_violation_rate"],
    "sequence": ["coverage_recall", "coverage_precision", "tof", "dovr", "sscr_pass"],
    "stage3": ["anim_csr", "aif", "anim_code_quality"],
}


def describe(key: str) -> dict:
    """Description for one metric, with a safe fallback for unknown keys."""
    return METRICS.get(key, {"label": key, "what": "", "better": "higher",
                             "stage": "", "doc": "", "range": ""})


def ordered(suite: str) -> list[str]:
    return SUITE_ORDER.get(suite, [])
