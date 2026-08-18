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
        "formula": "CSR = 1 if the code compiles, else 0",
        "terms": ["compiles"],
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
        "formula": "component_accuracy = correct verdicts / judged elements",
        "terms": ["component_judged"],
        "ratio": ("component_accuracy", "component_judged"),
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
        "formula": "judge score over 5 axes (semantic, logical layout, "
                   "geometric layout, style/colour, proportion/scale), with "
                   "layout weighted most heavily; forced to 0 when CSR = 0",
        "judged": True,
        "breakdown": "rendering_fidelity_breakdown",
        "notes_key": "rendering_fidelity_notes",
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
        "formula": "judge score; optional, only runs with --include-quality",
        "judged": True,
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
        "formula": "PAA = elements whose parent-group matches / matched elements",
        "terms": ["matched"],
        "ratio": ("paa", "matched"),
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
        "formula": "match_coverage = matched / scorable reference elements",
        "terms": ["matched", "scorable_gt_elements"],
        "ratio": ("matched", "scorable_gt_elements"),
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
        "formula": "F1 = 2PR / (P + R), where P = matched / comparable "
                   "predicted edges and R = matched / reference edges, all "
                   "after contracting endpoints onto alignment groups",
        "terms": ["edge_precision", "edge_recall", "gt_edges",
                  "pred_edges_comparable"],
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
        "formula": "violations / elements checked, where an element is a "
                   "violation when depth ≠ parent.depth + 1. Root children "
                   "are exempt (see the suite note).",
        "terms": ["elements_checked"],
        "ratio": ("depth_violation_rate", "elements_checked"),
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
        "formula": "recall = |gt_groups ∩ pred_groups| / |gt_groups|, over "
                   "alignment groups rather than raw ids. Ids with no group "
                   "are excluded from the ratio and listed separately.",
        "terms": ["gt_animated_matchable", "gt_animated"],
        "ratio": ("coverage_recall", "gt_animated_matchable"),
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
        "formula": "precision = |gt_groups ∩ pred_groups| / |pred_groups|, "
                   "over alignment groups rather than raw ids.",
        "terms": ["pred_animated_matchable", "pred_animated"],
        "ratio": ("coverage_precision", "pred_animated_matchable"),
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
        "formula": "overview_first: correctly ordered (top-level, deeper) "
                   "pairs / all such pairs.\n"
                   "detail_first: 1 − extra block switches / total switches, "
                   "where a perfect run switches blocks exactly once per block.",
        "detail_key": "tof_detail",
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
        "formula": "DOVR = edges revealed early / edges testable, where an "
                   "edge is early when it appears before max(source, target). "
                   "Null when no edge is testable — normal for bounding-box "
                   "styles, which never animate arrows.",
        "terms": ["edges_tested"],
        "ratio": ("dovr", "edges_tested"),
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
        "formula": "pass when the violation list is empty. Action-phrased "
                   "violations are filtered out on bench-dialect sequences "
                   "(see the suite note).",
        "detail_key": "sscr_violations",
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
        "formula": "1 if the animation compiles, else 0. Measured with the "
                   "exporter's source repairs applied; `raw_compiles` repeats "
                   "it without them, so `repair_rescued` shows what the "
                   "repair layer saved.",
        "terms": ["anim_compiles", "raw_compiles", "repair_rescued"],
    },
    "anim_code_quality": {
        "label": "Animation Code Quality",
        "stage": "Stage 3 · animation code",
        "doc": "§2.3.3",
        "what": "Readability of the frame logic and whether the effect is "
                "achieved by the simplest correct implementation.",
        "better": "higher",
        "range": "0–1",
        "formula": "judge score; optional, only runs with --include-quality",
        "judged": True,
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
        "formula": "AIF = diagram lines touched / animation lines added, from "
                   "a line diff: 'delete' and 'replace' count as touched, "
                   "'insert' and the replacement side of 'replace' as added. "
                   "Null when nothing was added.",
        "terms": ["aif_lines_touched", "aif_lines_added"],
        "ratio": ("aif_lines_touched", "aif_lines_added"),
    },

    # -- the animation evaluation tree -----------------------------------
    "vfs": {
        "label": "Visual Fidelity Score",
        "stage": "Animation · elimination 1",
        "doc": "VFS",
        "what": "Whether what is on screen is still the source figure. The "
                "judge rates six axes -- element alteration, visual layout, "
                "geometric layout, style/colour, proportion, text integrity -- "
                "with the two layout axes weighted heaviest.",
        "better": "higher",
        "range": "0–1",
        "caveat": "The last frame only, except alpha masking, where every "
                  "frame is judged and the score is their mean: intermediate "
                  "alpha frames mask parts of the figure, so only they can "
                  "show a defect the final frame hides.",
        "formula": "VFS = mean(visual_fidelity_score) / 10, one judged call "
                   "per frame. Additive content -- captions, step numbers, "
                   "titles -- is explicitly not a penalty.",
        "judged": True,
        "terms": ["vfs_raw", "vfs_frames_judged"],
    },
    "ascs_pass": {
        "label": "Ani-Style Compliance",
        "stage": "Animation · elimination 2",
        "doc": "ASCS",
        "what": "Whether the declared animation style was actually "
                "implemented, checked frame by frame against that style's own "
                "ruleset rather than against the source figure.",
        "better": "pass",
        "range": "pass / fail",
        "caveat": "The source document defines no number here despite 'Score' "
                  "in its name -- only per-rule booleans and a per-frame "
                  "verdict. Two rules per box style have no field in the "
                  "document's own schema, and one is sequence-level; those are "
                  "listed in `ascs_unenforced_rules` rather than silently "
                  "passing.",
        "formula": "pass when no frame is DISCARDed. The aggregation is ours: "
                   "judging one frame per call leaves no whole-animation turn "
                   "to ask in, and the document never states the rule. "
                   "`ascs_frames_discarded` is kept so a laxer rule can be "
                   "applied without re-running a call.",
        "judged": True,
        "detail_key": "ascs_discarded_frames",
    },
    "omission_rate": {
        "label": "Element Omission Rate",
        "stage": "Animation · elimination 3",
        "doc": "Omitted Elements/Arrows",
        "what": "Of the elements the animation's own sequence scheduled, the "
                "share that never actually appeared on screen.",
        "better": "lower",
        "range": "0–1",
        "caveat": "Self-referential by design. The checklist is built from the "
                  "sequence, not the XML, so an element the sequencer never "
                  "scheduled can never be flagged. This asks whether the "
                  "animation rendered what it promised -- `coverage_recall` in "
                  "the sequence suite is what asks whether it covered the "
                  "figure.",
        "formula": "omission rate = elements never popped / checklist size "
                   "(blocks + nodes). The counts are computed from the fold, "
                   "not asked of the judge: each call is only ever asked what "
                   "appeared in the one frame in front of it.",
        "terms": ["element_omission_count", "omission_checklist_size"],
        "ratio": ("element_omission_count", "omission_checklist_size"),
    },
    "arrow_omission_count": {
        "label": "Arrow Omissions",
        "stage": "Animation · elimination 3",
        "doc": "Omitted Elements/Arrows",
        "what": "Scheduled arrows that never appeared.",
        "better": "lower",
        "range": "0 upward",
        "caveat": "Always 0 for the two bounding-box styles, which never "
                  "highlight arrows -- the document tells the judge to ignore "
                  "the edge list entirely there. A zero means 'not applicable', "
                  "not 'none missing'; `omission_scores_edges` says which.",
        "formula": "count of checklist edges never popped, or 0 where the "
                   "style does not animate arrows",
        "terms": ["omission_arrows_remaining", "omission_scores_edges"],
    },
    "sss": {
        "label": "Selection Sensibility",
        "stage": "Animation · score contributor 1",
        "doc": "SSS",
        "what": "Per timestep, whether the right element or group was animated "
                "at that moment -- element appropriateness (criterion A) and "
                "group coherence (criterion B).",
        "better": "higher",
        "range": "0–1",
        "caveat": "One judged call per timestep, each seeing the figure, the "
                  "previous frame and the current one. The step's element ids "
                  "are deliberately withheld so the judge infers what changed "
                  "visually rather than reading the sequence file.",
        "formula": "SSS = mean(FINAL_BAND) / 4 over parseable timesteps. The "
                   "two criteria collapse by taking the LOWER band, and one "
                   "lower still when both are ≤ 2. Unparseable steps are "
                   "excluded and counted, never scored 0.",
        "judged": True,
        "terms": ["sss_band_mean", "sss_steps_scored", "sss_steps_invalid"],
    },
    "gps": {
        "label": "Granularity & Pacing",
        "stage": "Animation · score contributor 2",
        "doc": "GPS",
        "what": "Per timestep, whether the volume and complexity of what was "
                "animated together forms a digestible unit or overloads the "
                "viewer.",
        "better": "higher",
        "range": "0–1",
        "caveat": "Sparsity is never a violation, and a moving highlight "
                  "visiting one element at a time is exempt from volume "
                  "penalties however dense its surroundings.",
        "formula": "GPS = mean(FINAL_BAND) / 4. The collapse differs from "
                   "SSS: when both sub-criteria are below 4 it anchors on the "
                   "COMPLEXITY band and goes one lower, because complexity is "
                   "weighted above volume.",
        "judged": True,
        "terms": ["gps_band_mean", "gps_steps_scored", "gps_steps_invalid"],
    },
    "repetition_rate": {
        "label": "Unnecessary Repetition",
        "stage": "Animation · score contributor 3",
        "doc": "Unnecessary Repetition",
        "what": "The share of tracked elements re-highlighted without a good "
                "reason. A revisit counts only on a genuine state change, and "
                "only after the judge rules it unjustified.",
        "better": "lower",
        "range": "0–1",
        "caveat": "Defined for alpha masking and the two bounding-box styles "
                  "only. Progressive reveal and colour pop record "
                  "`repetition_status: not_specified` -- absent is not zero.",
        "formula": "rate = unjustified repeated elements / checklist size. "
                   "Justified revisits (cyclic edges, returning to a hub or "
                   "parent, overview-first traversal) are excluded by the "
                   "judge before counting.",
        "terms": ["unnecessary_repetition_count_element",
                  "unnecessary_repetition_count_arrow", "repetition_revisited"],
    },
}

# Display order within each suite, so the viewer reads top-down by stage.
SUITE_ORDER: dict[str, list[str]] = {
    "stage1": ["csr", "rendering_fidelity", "component_accuracy", "code_quality"],
    "xml": ["paa", "matched_coverage", "edge_f1", "depth_violation_rate"],
    "sequence": ["coverage_recall", "coverage_precision", "tof", "dovr", "sscr_pass"],
    "stage3": ["anim_csr", "aif", "anim_code_quality"],
    # Tree order: the three gates, then the three contributors.
    "animation": ["vfs", "ascs_pass", "omission_rate", "arrow_omission_count",
                  "sss", "gps", "repetition_rate"],
}


# Why each suite's metrics are shaped the way they are. Quoted from the
# metric modules' own docstrings and README.md §2-§3 rather than paraphrased,
# for the same reason METRICS lives here: the explanation and the computation
# must not drift apart.
SUITE_NOTES: dict[str, dict] = {
    "stage1": {
        "title": "Stage 1 · image → diagram code",
        "note": "Both judged metrics score the **code**, not the XML induced "
                "from it. The semantic class tags originate at the "
                "diagram-to-code step, so an error there propagates into the "
                "XML — scoring the XML would blame the parser for a D2C "
                "mistake.\n\n"
                "Compilation and fidelity are deliberately separate. "
                "Compiling is mechanical and says nothing about whether the "
                "figure is right; fidelity is the only Stage-1 check that "
                "sees a defect the compiler cannot.",
        "example": {
            "title": "Why one number would not have been enough",
            "body": "On pipe00041 the diagram scored CSR 1.000 and rendering "
                    "fidelity 0.300. The code declared each `fit` block "
                    "*after* the nodes it contained, with an opaque fill — "
                    "and TikZ paints in source order, so every container "
                    "painted over its own children. It compiled with zero "
                    "warnings, embedded all its raster crops, and rendered "
                    "almost empty. A single pass/fail metric would have "
                    "called this a success.",
        },
    },
    "xml": {
        "title": "Stage 2 · structure XML",
        "note": "This is where the GT↔prediction **alignment** does its work. "
                "Ids never match textually and the correspondence is not "
                "1:1 — the reference may draw three thumbnails where the "
                "prediction has one image, and some elements have no "
                "counterpart at all. So a judge proposes a *grouping* once "
                "per sample, it is mechanically validated (invented ids "
                "dropped, duplicates rejected) and cached, and every "
                "GT-dependent metric reads that same artifact. Scores cannot "
                "disagree about element identity.\n\n"
                "**Always read PAA against match coverage.** A perfect parent "
                "score over three matched elements is not the claim it looks "
                "like.\n\n"
                "Depth violations exempt root children on purpose: the "
                "reference bundle mixes both depth conventions inside one "
                "file, and a fixed base-depth rule reported the reference's "
                "own hand-authored XML as ~19% invalid.",
    },
    "sequence": {
        "title": "Stage 2 · animation sequence",
        "note": "Only **coverage** compares against the reference, and it "
                "does so as a set comparison of animated elements contracted "
                "through the element alignment — there is no step-to-step "
                "matching, no edit distance over the traversal.\n\n"
                "TOF, SSCR and DOVR are self-contained: they score the "
                "sequence against its own XML and its declared style "
                "contract, with no judge and no ground truth.\n\n"
                "Ids that cannot be aligned are excluded from both coverage "
                "ratios and reported separately — they are structure-XML "
                "gaps, already scored by match coverage, not sequencing "
                "mistakes.\n\n"
                "SSCR drops action-phrased violations on bench-dialect "
                "sequences: that dialect has no action field, so every step "
                "defaults to \"reveal\", and checking it unfiltered failed "
                "every bench-format bounding-box sequence.",
    },
    "stage3": {
        "title": "Stage 3 · animation code",
        "note": "AIF is a *footprint*, not a quality score: it asks how much "
                "of the diagram the animation had to disturb. 0 means "
                "animation was a purely additive layer; above 1 means more "
                "existing code was rewritten than animation code added.\n\n"
                "Compilation is measured twice — once with the exporter's "
                "source repairs, once without — so the repair layer's "
                "contribution stays visible instead of silently inflating "
                "the headline number.",
    },
    "animation": {
        "title": "Animation · the evaluation tree",
        "note": "A **cascade**, not a weighted sum: three elimination gates "
                "(visual fidelity, style compliance, omitted elements) and "
                "three score contributors (selection, pacing, repetition). "
                "The lower questions are meaningless when the top one fails — "
                "\"was the right group animated here\" says nothing about an "
                "animation whose frames do not depict the source figure.\n\n"
                "**The gates are recorded, not enforced.** No source document "
                "states a threshold for any of them, so every node that can "
                "run does, and `would_eliminate` records what *would* have "
                "fired. Only style compliance can answer today, because its "
                "verdict is categorical. Closing a gate on an invented cutoff "
                "would zero every metric beneath it, and nothing downstream "
                "could tell that from a genuine failure.\n\n"
                "**One judged call per frame.** The gates walk the animation "
                "forward, each call seeing a single frame plus the state the "
                "previous call produced — the checklist still outstanding, the "
                "running frequency table. Every count is then computed here "
                "from those per-frame reports rather than asked of the model: "
                "a judge asked to total thirty frames of its own work is being "
                "asked to do arithmetic nobody can check, while a judge asked "
                "\"what appeared in this frame\" is being asked what it can "
                "see.\n\n"
                "**Omission is self-referential.** Its checklist comes from "
                "the animation's own sequence, because the design document "
                "rules the XML out: \"not all elements in the XML may "
                "explicitly make it to the actual animated list.\" So an "
                "element the sequencer never scheduled can never be flagged "
                "omitted. Read it beside `coverage_recall`, which is the "
                "GT-based question.\n\n"
                "**Two nodes are incompletely specified.** Repetition has no "
                "prompt for progressive reveal or colour pop, and records "
                "`not_specified` there rather than 0. Both bounding-box styles "
                "carry style rules with no field in the document's own output "
                "schema, and sliding carries one that is sequence-level and "
                "unanswerable from a single frame; all are listed in "
                "`ascs_unenforced_rules`.",
        "example": {
            "title": "Why the counts are computed here, not asked for",
            "body": "The source prompts ask the judge to walk every frame and "
                    "report `element_omission_count` itself. Judging one frame "
                    "per call makes that impossible — and better. Each call "
                    "answers only what it can see, the outstanding list "
                    "shrinks in Python, and a name the judge invents or "
                    "re-pops cannot shrink it, because only names actually "
                    "still outstanding are allowed to pop. The headline number "
                    "is then arithmetic over evidence rather than a model's "
                    "own tally of work it cannot re-examine.",
        },
    },
}


def describe(key: str) -> dict:
    """Description for one metric, with a safe fallback for unknown keys."""
    return METRICS.get(key, {"label": key, "what": "", "better": "higher",
                             "stage": "", "doc": "", "range": ""})


def ordered(suite: str) -> list[str]:
    return SUITE_ORDER.get(suite, [])


def suite_note(suite: str) -> dict:
    return SUITE_NOTES.get(suite, {"title": suite, "note": ""})


def _fmt_number(value) -> str:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def instantiate(key: str, record: dict) -> str | None:
    """Render a metric's formula using this record's own numbers.

    Returns e.g. "118 / 232 = 0.509" for a ratio metric, the stored detail
    string where the metric keeps one (TOF states its own pair counts), or
    None for judged metrics, which have no arithmetic to show -- their
    evidence is the breakdown and the judge's prose instead.

    The numerator is reconstructed as `value * denominator` rather than
    recomputed from source data: this displays what was actually recorded, so
    a drift between the stored score and its own terms shows up rather than
    being silently papered over.
    """
    meta = METRICS.get(key)
    if meta is None:
        return None

    detail = meta.get("detail_key")
    if detail and record.get(detail):
        value = record[detail]
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return f"{len(value)} violation(s)"

    # F1 shows both of its inputs rather than a single fraction.
    if key == "edge_f1":
        p, r = record.get("edge_precision"), record.get("edge_recall")
        if p is None or r is None:
            return None
        return (f"P = {_fmt_number(p)}, R = {_fmt_number(r)} → "
                f"2PR/(P+R) = {_fmt_number(record.get(key))}")

    ratio = meta.get("ratio")
    if not ratio:
        return None

    num_key, den_key = ratio
    value, denominator = record.get(num_key), record.get(den_key)
    if value is None or denominator in (None, 0):
        return None

    # `num_key` is either the score itself (reconstruct the count) or an
    # already-counted numerator (use it directly).
    if num_key == key:
        numerator = round(value * denominator)
        return (f"{_fmt_number(numerator)} / {_fmt_number(denominator)} "
                f"= {_fmt_number(value)}")

    result = record.get(key)
    return (f"{_fmt_number(value)} / {_fmt_number(denominator)}"
            + (f" = {_fmt_number(result)}" if result is not None else ""))
