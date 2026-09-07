"""The animation evaluation tree: judged animation quality.

Three elimination gates (visual fidelity, style compliance, omitted elements)
and three score contributors (selection sensibility, granularity/pacing,
unnecessary repetition). A cascade rather than a weighted sum, because the
lower questions are meaningless when the top one fails: "was the right group
animated at this timestep" says nothing about an animation whose frames do not
depict the source figure.

Two structural facts drive the whole module.

**One image per call.** The gates are sequential folds, not batch calls: each
frame is judged on its own, carrying forward the state the previous frame
produced -- the checklist still outstanding, the running frequency table. That
is what the design documents' "popping" describes, and it is why the counts
here are computed in Python from the accumulated per-frame reports rather than
asked of the model. A model asked to total its own work across thirty frames is
being asked to do arithmetic it cannot check; a model asked "what appeared in
this one frame" is being asked what it can see.

**The gates are recorded, not enforced.** No document states a threshold for
any of the three. Running with the gates closed on invented cutoffs would zero
every metric below them, so `run` scores every node it can and records
`would_eliminate` instead. Turning a gate on is then a decision made against an
observed distribution rather than a guess made before one.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from img_2_svg_pretraining.pipeline.prompts import load_and_render

from ..frames import FrameSet
from ..judge import JudgeError, PROMPTS_ROOT

STYLES = ("progressive_reveal", "colour_pop", "alpha_masking",
          "hopping_bounding_box", "sliding_bounding_box")

# Which frames each node looks at, per style. These must agree with what the
# prompt's style adapter tells the judge it is being shown -- the adapter says
# "the LAST frame", this decides that only the last frame is attached. A test
# pins them together.
#
# VFS: the last frame for four styles (it is the completed diagram), every
# frame for alpha masking, whose intermediate states mask parts of the figure
# and so have to be checked individually.
VFS_POLICY = {**{s: "last" for s in STYLES}, "alpha_masking": "all"}
# Style compliance asks "does every frame obey the contract", so: every frame.
ASCS_POLICY = {s: "all" for s in STYLES}
# Omission and repetition walk the animation forward. Sliding bounding box
# samples every fourth frame, which is the design document's own rule and also
# what keeps the 84-frame outlier tractable.
WALK_POLICY = {**{s: "all" for s in STYLES}, "sliding_bounding_box": "every_4"}

# The repetition contributor is specified for three styles only. Progressive
# reveal never hides anything, so a re-reveal is close to impossible; colour pop
# has a real greyscale/colour analogue and its absence looks like an oversight
# in the source document. Either way, absent means unscored -- never zero.
REPETITION_STYLES = ("alpha_masking", "hopping_bounding_box",
                     "sliding_bounding_box")

# Rules the judge is asked to enforce but structurally cannot report on.
# Recorded rather than silently passed: silence from a rule with nowhere to
# put its answer reads exactly like compliance.
#
# "Mobile Boxes, Static Elements" used to be listed here for both box styles,
# because the document's own schema had no key for it. It is enforceable now:
# the 2026-08-19 revision added the missing `mobile_boxes_static_elements`
# field AND started sending the previous frame, without which "no element
# moved" was a question about change asked of a single image.
UNENFORCEABLE_RULES = {
    "hopping_bounding_box": [],
    "sliding_bounding_box": ["Hopping Boxes (HIGHLY CRITICAL): sequence-level, "
                             "unanswerable from one frame"],
}

STAGES = ("vfs", "ascs", "omission", "sss", "gps", "repetition", "nas")

# Opt-in only: named on the command line or it does not run. Deliberately NOT
# folded into STAGES, for three reasons.
#   - Every default full run stays byte-identical to the one that produced the
#     64 existing records, so old and new numbers remain comparable.
#   - `run_eval._run_animation` branches on `set(stages) != set(STAGES)`; adding
#     a member here would silently change that comparison's meaning.
#   - vfs_video is the only node that calls a paid/quota'd external judge for
#     every cell. Accidental spend should require a typo in an explicit flag,
#     not merely running the suite.
EXTRA_STAGES = ("vfs_video", "vfs_band", "ascs_video", "nas")

ALL_STAGES = STAGES + EXTRA_STAGES

# The two banded judges answer in section headers, not JSON. Regexes are the
# design documents' own.
_BAND_RE = {
    "criterion_a_band": re.compile(r"###\s*CRITERION_A_BAND:\s*([0-4])", re.I),
    "criterion_b_band": re.compile(r"###\s*CRITERION_B_BAND:\s*([0-4])", re.I),
    "volume_band": re.compile(r"###\s*VOLUME_BAND:\s*([0-4])", re.I),
    "complexity_band": re.compile(r"###\s*COMPLEXITY_BAND:\s*([0-4])", re.I),
    "relevance_band": re.compile(r"###\s*RELEVANCE_BAND:\s*([0-4])", re.I),
    "final_band": re.compile(r"###\s*FINAL_BAND:\s*([0-4])", re.I),
    "label": re.compile(r"###\s*LABEL:\s*(.+)", re.I),
}
_RATIONALE_RE = re.compile(r"###\s*RATIONALE\s*(.*?)(?=###\s*[A-Z_]+_BAND)",
                           re.I | re.S)

SSS_KEYS = ("criterion_a_band", "criterion_b_band")
# RELEVANCE_BAND joined GPS in the 2026-08-19 prompt revision as the "shield"
# that excuses a cohesive composite reveal. Only FINAL_BAND gates `is_valid`,
# so a reply from an older cached response that predates the header still
# parses -- relevance_band simply lands as None.
GPS_KEYS = ("volume_band", "complexity_band", "relevance_band")

BAND_MAX = 4.0
VFS_MAX = 10.0

# The letter rubric, mapped onto the SAME 0-4 scale the header rubric used, so
# both formats produce a directly comparable `sss`/`gps` number.
#
# NOTE THE DIRECTION. "A" is the BEST band and maps to 4, the top of the old
# integer scale; "E" is the worst and maps to 0. Assigning A->0 because A comes
# first would invert the metric silently -- every well-paced animation would
# score zero and every information dump would score one, and the number would
# still look perfectly plausible in a table.
BAND_LETTER_VALUE = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}

# Sub-scores per metric in the JSON rubric: (json key, stored key).
SSS_JSON_KEYS = (("criterion_a_appropriateness", "criterion_a_band"),
                 ("criterion_b_coherence", "criterion_b_band"))
NAS_JSON_KEYS = (("animation_transition_alignment", "alignment_band"),
                 ("contextual_insight_and_depth", "insight_band"),
                 ("coherence_and_accuracy", "coherence_band"))
GPS_JSON_KEYS = (("volume", "volume_band"),
                 ("complexity", "complexity_band"),
                 ("relevance", "relevance_band"))

# The FINAL prompts (Medha, 2026-08-31: *_final.yaml) changed the output
# contract: per-rule REASONING STRINGS plus one final_score letter, no
# per-criterion letter grades. The reasonings are stored verbatim as text --
# they must NEVER go through `_letter_band`, whose first-standalone-A-E regex
# would happily read a "band" out of uppercased prose. Hence `text_keys`,
# a separate lane through `parse_json_bands`.
SSS_FINAL_TEXT_KEYS = (
    ("rule_1_appropriateness_reasoning", "rule_1_appropriateness_reasoning"),
    ("rule_2_coherence_reasoning", "rule_2_coherence_reasoning"))
GPS_FINAL_TEXT_KEYS = (
    ("rule_1_volume_reasoning", "rule_1_volume_reasoning"),
    ("rule_2_complexity_reasoning", "rule_2_complexity_reasoning"),
    ("rule_3_relevance_reasoning", "rule_3_relevance_reasoning"))
NAS_FINAL_TEXT_KEYS = (
    ("rule_1_alignment_reasoning", "rule_1_alignment_reasoning"),
    ("rule_2_narrative_context_reasoning", "rule_2_narrative_context_reasoning"),
    ("rule_3_coherence_and_factuality_reasoning",
     "rule_3_coherence_and_factuality_reasoning"))


# -- shared helpers --------------------------------------------------------

def fold_frames(frames: FrameSet, step) -> tuple[list[dict], list[str]]:
    """Call `step(position, path, label)` once per frame, in playback order.

    State is threaded through the caller's own closure rather than a tuple
    protocol, so each node keeps its state in the shape it actually needs.

    A frame whose judge call fails is recorded and skipped rather than
    aborting: losing one frame of thirty degrades the evidence, losing the
    fold loses the cell.
    """
    results: list[dict] = []
    errors: list[str] = []
    for position, (path, label) in enumerate(zip(frames.paths, frames.labels)):
        try:
            outcome = step(position, path, label)
        except JudgeError as e:
            errors.append(f"{label}: {e}")
            continue
        if outcome is not None:
            results.append(outcome)
    return results, errors


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _adapter(prompt_file: str, style: str, prefix: str = "adapter") -> str:
    return load_and_render(f"{prompt_file}#{prefix}_{style}", {}, root=PROMPTS_ROOT)


def _names(payload, *buckets) -> list[str]:
    """Flatten a judge's bucketed name lists, dropping anything unusable."""
    out: list[str] = []
    if not isinstance(payload, dict):
        return out
    for bucket in buckets:
        values = payload.get(bucket)
        if isinstance(values, list):
            out.extend(v.strip() for v in values
                       if isinstance(v, str) and v.strip())
    return out


def _instances(payload, bucket: str) -> list[tuple[str, int]]:
    """(label, instances_seen) pairs from one bucket of a judge's pop report.

    Accepts both shapes. The frequency-aware prompt returns
    `{"label": ..., "instances_seen": n}`; a bare string is what the older
    prompt returned and what a model occasionally still emits, and it means
    exactly one sighting. Reading only the new shape would silently score
    every such reply as "nothing appeared".
    """
    out: list[tuple[str, int]] = []
    if not isinstance(payload, dict):
        return out
    values = payload.get(bucket)
    if not isinstance(values, list):
        return out
    for value in values:
        if isinstance(value, str) and value.strip():
            out.append((value.strip(), 1))
        elif isinstance(value, dict):
            label = str(value.get("label") or "").strip()
            if not label:
                continue
            try:
                seen = int(value.get("instances_seen", 1))
            except (TypeError, ValueError):
                seen = 1
            out.append((label, max(seen, 1)))
    return out


def parse_bands(text: str, extra_keys: tuple[str, ...]) -> dict:
    """Recover a banded verdict from the `### HEADER: value` reply.

    `is_valid` is False when FINAL_BAND cannot be read, which keeps an
    unparseable timestep out of the mean instead of letting it land there as a
    zero -- a parse failure is missing data, not a severe violation.
    """
    result = {"rationale": None, "final_band": None, "label": None,
              "is_valid": False}
    for key in (*extra_keys, "final_band"):
        match = _BAND_RE[key].search(text)
        result[key] = int(match.group(1)) if match else None
    label = _BAND_RE["label"].search(text)
    result["label"] = label.group(1).strip() if label else None
    rationale = _RATIONALE_RE.search(text)
    result["rationale"] = rationale.group(1).strip() if rationale else None
    result["is_valid"] = result["final_band"] is not None
    return result


def _has_final_band(text: str) -> bool:
    return bool(_BAND_RE["final_band"].search(text))


# -- E1a: visual fidelity --------------------------------------------------

def visual_fidelity(judge, source_image: Path, frames: FrameSet,
                    style: str) -> dict:
    """VFS: is what is on screen still the source figure? (Elimination 1)"""
    prompt = load_and_render("animation_fidelity.yaml#frames", {
        "style_adapter": _adapter("animation_fidelity.yaml", style,
                                  "adapter_frames"),
    }, root=PROMPTS_ROOT)

    per_frame: list[dict] = []

    def step(position, path, label):
        data = judge.ask_json(f"{prompt}\n\nThe frame provided is: {label}",
                              images=[Path(source_image), path],
                              tag=f"vfs_{style}")
        entries = data.get("frame_evaluations")
        entry = entries[0] if isinstance(entries, list) and entries else data
        score = entry.get("visual_fidelity_score") if isinstance(entry, dict) else None
        record = {
            "frame": label,
            "score_raw": float(score) if isinstance(score, (int, float)) else None,
            "assessment": entry.get("assessment") if isinstance(entry, dict) else None,
        }
        per_frame.append(record)
        return record

    results, errors = fold_frames(frames, step)
    scores = [r["score_raw"] for r in results if r["score_raw"] is not None]
    mean = _mean(scores)

    return {
        "vfs": (mean / VFS_MAX) if mean is not None else None,
        "vfs_raw": mean,
        "vfs_frame_scores": [r["score_raw"] for r in per_frame],
        "vfs_frames": per_frame,
        "vfs_errors": errors,
        **{f"vfs_{k}": v for k, v in frames.manifest().items()},
    }


# -- E1b: visual fidelity, judged from the video ---------------------------

def video_fidelity(judge, source_image: Path, video_path: Path,
                   style: str, meta: dict | None = None) -> dict:
    """VFS judged from the animation as a video rather than frame by frame.

    The same construct as `visual_fidelity`, asked of a different modality. It
    exists to answer whether a frame judge is enough: `VFS_POLICY` is "last" for
    four of the five styles, so for those the frame score is one call about one
    frame and cannot, even in principle, see flicker, a defect that appears and
    resolves, or anything about pacing.

    `temporal_defects_observed` is the field with no frame-judge analogue and
    the reason this node is worth its quota -- it is where a mid-animation
    defect gets named. Every key is namespaced `vfs_video_*` so nothing here can
    overwrite the Qwen-produced `vfs*` keys sitting in the same record.
    """
    prompt = load_and_render("animation_fidelity.yaml#video", {
        "style_adapter": _adapter("animation_fidelity.yaml", style,
                                  "adapter_video"),
    }, root=PROMPTS_ROOT)

    out: dict = {
        "vfs_video": None,
        "vfs_video_raw": None,
        "vfs_video_assessment": None,
        "vfs_video_temporal_defects": None,
        "vfs_video_errors": [],
        **{f"vfs_video_{k}": v for k, v in (meta or {}).items()},
    }

    try:
        data = judge.ask_json(prompt, images=[Path(source_image)],
                              videos=[Path(video_path)], tag=f"vfsv_{style}")
    except JudgeError as exc:
        out["vfs_video_errors"] = [str(exc)]
        return out

    block = data.get("video_evaluation")
    if not isinstance(block, dict):
        block = data
    score = block.get("visual_fidelity_score")
    if isinstance(score, (int, float)):
        out["vfs_video_raw"] = float(score)
        out["vfs_video"] = float(score) / VFS_MAX
    else:
        out["vfs_video_errors"] = ["no numeric visual_fidelity_score in response"]
    out["vfs_video_assessment"] = block.get("assessment")
    out["vfs_video_temporal_defects"] = block.get("temporal_defects_observed")
    return out


# -- E1b/E2 judged from the video, with the band rubric --------------------

# "BAND A".."BAND D", best to worst. Ordinal 0..3 so a mean is meaningful and
# so `_band_pass` is a comparison rather than a set membership test.
FIDELITY_BANDS = ("BAND A", "BAND B", "BAND C", "BAND D")

# A and B pass; C and D fail. Stated by the prompts themselves -- C is labelled
# "Poor (FAIL)" and D "Severe Failure (FAIL)" -- so this is transcription, not
# a threshold invented here. Contrast `_would_eliminate`, which refuses to
# guess cutoffs the source documents never state.
FIDELITY_BAND_PASS_MAX = 1


def parse_fidelity_band(text: str) -> int | None:
    """Ordinal of the band named in `text`, or None if none is.

    Matches on the band token rather than a bare letter: the rationale prose
    routinely contains "A" and "B" as words, and an over-eager letter match
    would read a band out of the explanation instead of the verdict.
    """
    if not isinstance(text, str):
        return None
    upper = text.strip().upper()
    for ordinal, name in enumerate(FIDELITY_BANDS):
        if name in upper:
            return ordinal
    return None


def video_fidelity_bands(judge, source_image: Path, video_path: Path,
                         style: str, meta: dict | None = None) -> dict:
    """VFS as a four-band classification of the whole animation (E1-video).

    Distinct from BOTH existing fidelity nodes, and deliberately additive:
    `vfs` is Qwen judging frames on 0-10, `vfs_video` is Gemini judging the
    video on 0-10, and this is Gemini judging the video into bands. All three
    keep their own key namespace so a record can carry all of them and the
    three can be compared against each other -- which is the entire point of
    the study. Nothing here overwrites anything.

    The numeric scales piled up at their ceiling: 86% of scored cells sat at
    exactly VFS 1.0, which makes a correlation against them undefined rather
    than weak. Bands force a commitment to whether a defect is material.
    """
    prompt = load_and_render(f"animation_fidelity_bands.yaml#{style}", {},
                             root=PROMPTS_ROOT)

    out: dict = {
        "vfs_band": None,
        "vfs_band_ordinal": None,
        "vfs_band_pass": None,
        "vfs_band_summary": None,
        "vfs_band_rationale": None,
        "vfs_band_errors": [],
        **{f"vfs_band_{k}": v for k, v in (meta or {}).items()},
    }

    try:
        data = judge.ask_json(prompt, images=[Path(source_image)],
                             videos=[Path(video_path)], tag=f"vfsband_{style}")
    except JudgeError as exc:
        out["vfs_band_errors"] = [str(exc)]
        return out

    ordinal = parse_fidelity_band(data.get("fidelity_band"))
    if ordinal is None:
        out["vfs_band_errors"] = [
            f"no recognisable fidelity_band in response: "
            f"{str(data.get('fidelity_band'))[:120]!r}"]
    else:
        out["vfs_band"] = FIDELITY_BANDS[ordinal]
        out["vfs_band_ordinal"] = ordinal
        out["vfs_band_pass"] = ordinal <= FIDELITY_BAND_PASS_MAX
    out["vfs_band_summary"] = data.get("summary")
    rationale = data.get("rationale")
    out["vfs_band_rationale"] = rationale if isinstance(rationale, dict) else None
    return out


def style_compliance_video(judge, source_image: Path, video_path: Path,
                           style: str, meta: dict | None = None) -> dict:
    """ASC asked ONCE of the whole video, instead of per frame (E2-video).

    The frame-level `style_compliance` folds per-frame verdicts with a strict
    AND, and `docs/METRIC_RELIABILITY.md` measured the consequence: the
    per-frame judgements are internally consistent (Spearman-Brown 0.809), but
    35% of cell failures rest on a single frame. The aggregation, not the
    judge, was the unreliable half -- so this node removes the aggregation
    entirely by asking the question once.

    It also asks a question the frame judge could not answer even in principle.
    Hopping and sliding differ ONLY in transit; every individual frame of both
    shows a box resting somewhere. See video.py on why a sparse deck can still
    hide the transit from this judge too.
    """
    prompt = load_and_render(f"animation_style_video.yaml#{style}", {},
                             root=PROMPTS_ROOT)

    out: dict = {
        # The scoreboard reads the metric's own name as the column value, the
        # way `ascs_pass` and `vfs_band` do. Without this key the column stays
        # blank on a cell that was in fact judged, which reads as "the metric
        # did not run" rather than "it ran and said DISCARD".
        "ascs_video": None,
        "ascs_video_verdict": None,
        "ascs_video_pass": None,
        "ascs_video_rationale": None,
        "ascs_video_errors": [],
        **{f"ascs_video_{k}": v for k, v in (meta or {}).items()},
    }

    try:
        data = judge.ask_json(prompt, images=[Path(source_image)],
                             videos=[Path(video_path)], tag=f"ascv_{style}")
    except JudgeError as exc:
        out["ascs_video_errors"] = [str(exc)]
        return out

    verdict = data.get("overall_verdict")
    text = verdict.strip().upper() if isinstance(verdict, str) else ""
    # Substring rather than equality: the field arrives as "ACCEPT", but also
    # as "DISCARD - rule 1 failed". DISCARD is checked FIRST because that
    # phrasing can contain both tokens, and reading such a response as ACCEPT
    # would turn a failure into a pass.
    if "DISCARD" in text:
        out["ascs_video_verdict"], out["ascs_video_pass"] = "DISCARD", False
    elif "ACCEPT" in text:
        out["ascs_video_verdict"], out["ascs_video_pass"] = "ACCEPT", True
    else:
        out["ascs_video_errors"] = [
            f"no ACCEPT/DISCARD in overall_verdict: {str(verdict)[:120]!r}"]
    out["ascs_video"] = out["ascs_video_verdict"]
    out["ascs_video_rationale"] = data.get("rationale")
    return out


# -- E2: style compliance --------------------------------------------------

def style_compliance(judge, source_image: Path, frames: FrameSet,
                     style: str) -> dict:
    """ASCS: was the declared style actually implemented? (Elimination 2)

    The overall verdict is aggregated here rather than asked of the judge.
    Judging one frame per call leaves no whole-animation turn to ask in, and
    the document never states the aggregation anyway -- its worked examples
    show an overall DISCARD alongside accepted frames. So the rule is applied
    where it can be seen and changed: any discarded frame discards the
    animation, with the discarded count kept so a laxer rule can be applied
    later without re-running a single call.
    """
    total = len(frames.paths)
    prompt_template = load_and_render("animation_style.yaml#frames", {
        "style_name": style.replace("_", " ").title(),
        "style_adapter": _adapter("animation_style.yaml", style),
        "output_schema": _adapter("animation_style.yaml", style, "schema"),
    }, root=PROMPTS_ROOT)

    # Every style now has at least one rule phrased as a temporal comparison
    # ("compare the current frame to the PREVIOUS frame"), so the previous
    # frame is sent alongside. Asking that question while showing only one
    # frame invites an invented answer: the judge cannot see what changed, but
    # nothing stops it from asserting that something did.
    def step(position, path, label):
        prompt = prompt_template.replace(
            "{frame_context}", f"frame {position + 1} of {total} ({label})")
        if position == 0:
            # No predecessor exists. Say so plainly and excuse the temporal
            # rules rather than passing the original image as a stand-in --
            # frame 1 legitimately shows far less than the original, which
            # would read as a mass persistence violation on every first frame.
            images = [Path(source_image), path]
            # Drop the "2. PREVIOUS frame" line entirely and renumber, so the
            # prompt never promises an image that is not attached. Matched
            # line-wise: the YAML block scalar's exact trailing whitespace is
            # not something to depend on.
            kept = [ln for ln in prompt.split("\n")
                    if "{previous_note}" not in ln]
            prompt = "\n".join(kept).replace(
                "  3. The CURRENT frame under evaluation",
                "  2. The CURRENT frame under evaluation")
            prompt += ("\n\nNOTE: This is the FIRST frame of the sequence, so "
                       "NO previous frame is provided. Every rule labelled "
                       "'Temporal Comparison' or 'Temporal Persistence' cannot "
                       "be evaluated here: mark it `followed: true` and state "
                       "in its reasoning that it is not applicable to the first "
                       "frame. Judge all remaining rules normally.")
        else:
            images = [Path(source_image), frames.paths[position - 1], path]
            prompt = prompt.replace("{previous_note}",
                                    f"This is frame {position} of {total}.")
        data = judge.ask_json(prompt, images=images, tag=f"ascs_{style}")
        verdict = str(data.get("frame_verdict") or "").strip().upper()
        return {
            "frame": label,
            "verdict": verdict if verdict in ("ACCEPT", "DISCARD") else None,
            "generic_quality_checks": data.get("generic_quality_checks"),
            "style_specific_checks": data.get("style_specific_checks"),
        }

    results, errors = fold_frames(frames, step)
    judged = [r for r in results if r["verdict"] is not None]
    discarded = [r["frame"] for r in judged if r["verdict"] == "DISCARD"]

    return {
        # `frames.manifest()` emits a key named `frames_judged`, which becomes
        # `ascs_frames_judged` and COLLIDES with the count of frames that came
        # back with a parseable verdict. The splat is first so the manifest's
        # meaning (frames SENT) is the one that survives: that is what all 64
        # existing records already store -- verified identical on every one --
        # and what `scripts/compare_judges.py` reads it as. The unambiguous
        # count gets its own key rather than silently redefining the old one.
        **{f"ascs_{k}": v for k, v in frames.manifest().items()},
        "ascs_pass": (not discarded) if judged else None,
        "ascs_frames_verdicted": len(judged),
        "ascs_frames_discarded": len(discarded),
        "ascs_discarded_frames": discarded,
        "ascs_frame_detail": results,
        "ascs_errors": errors,
        "ascs_unenforced_rules": UNENFORCEABLE_RULES.get(style, []),
    }


# -- E3: omitted elements --------------------------------------------------

def omitted_elements(judge, source_image: Path, frames: FrameSet, style: str,
                     checklist) -> dict:
    """Did the animation render what its own sequence promised? (Elimination 3)

    The counts are computed here, from the checklist minus everything the fold
    popped. The judge is only ever asked what appeared in the frame in front of
    it -- which is what it can actually see -- so a miscount is impossible
    rather than merely unlikely.
    """
    scores_edges = style not in ("hopping_bounding_box", "sliding_bounding_box")
    # label -> instances still unseen. A label can stand for several physically
    # distinct elements, so this counts down rather than being removed: seeing
    # one "Layer Norm" of three leaves two still owed.
    remaining = {
        "blocks": {e.label: e.frequency for e in checklist.blocks},
        "nodes": {e.label: e.frequency for e in checklist.nodes},
        "edges": ({e.label: e.frequency for e in checklist.edges}
                  if scores_edges else {}),
    }
    prompt_template = load_and_render("animation_omission.yaml#frames", {
        "style_adapter": _adapter("animation_omission.yaml", style),
        "output_schema": _adapter("animation_omission.yaml", style, "schema"),
    }, root=PROMPTS_ROOT)

    def render_remaining() -> str:
        lines = []
        for bucket in ("blocks", "nodes", "edges"):
            if not scores_edges and bucket == "edges":
                continue
            shown = [label if count == 1 else f"{label} (x{count})"
                     for label, count in remaining[bucket].items() if count > 0]
            lines.append(f"  {bucket}: {shown}")
        return "\n".join(lines)

    def step(position, path, label):
        prompt = prompt_template.replace("{remaining}", render_remaining())
        data = judge.ask_json(f"{prompt}\n\nThe frame provided is: {label}",
                              images=[Path(source_image), path],
                              tag=f"omission_{style}")
        popped = data.get("elements_popped_in_this_frame")
        this_frame: dict[str, list[dict]] = {}
        for bucket in ("blocks", "nodes", "edges"):
            if bucket == "edges" and not scores_edges:
                continue
            hits = []
            for name, seen in _instances(popped, bucket):
                # Only what is actually outstanding may be popped, and never
                # more of it than is owed: a judge naming something already
                # gone, or claiming three sightings of a label with one
                # instance left, must not drive the count below zero.
                outstanding = remaining[bucket].get(name, 0)
                if outstanding <= 0:
                    continue
                take = min(seen, outstanding)
                remaining[bucket][name] = outstanding - take
                hits.append({"label": name, "instances_seen": take})
            if hits:
                this_frame[bucket] = hits
        return {"frame": label, "reasoning": data.get("reasoning"),
                "popped": this_frame}

    results, errors = fold_frames(frames, step)

    def unseen(bucket: str) -> list[dict]:
        return [{"label": label, "unseen_count": count}
                for label, count in remaining[bucket].items() if count > 0]

    element_unseen = unseen("blocks") + unseen("nodes")
    arrow_unseen = unseen("edges")
    element_omission_count = sum(e["unseen_count"] for e in element_unseen)
    arrow_omission_count = sum(e["unseen_count"] for e in arrow_unseen)

    return {
        "element_omission_count": element_omission_count,
        "arrow_omission_count": arrow_omission_count if scores_edges else 0,
        "omission_rate": (element_omission_count / checklist.total(False)
                          if checklist.total(False) else None),
        "omission_checklist_size": checklist.total(scores_edges),
        "omission_elements_remaining": element_unseen,
        "omission_arrows_remaining": arrow_unseen,
        "omission_scores_edges": scores_edges,
        "omission_frame_detail": results,
        "omission_errors": errors,
        **{f"omission_{k}": v for k, v in frames.manifest().items()},
    }


# -- SC3: unnecessary repetition -------------------------------------------

def unnecessary_repetition(judge, source_image: Path, frames: FrameSet,
                           style: str, checklist) -> dict:
    """Does the animation re-highlight what it already covered?

    Specified for three styles only; the other two record `not_specified`
    rather than a zero, because "no prompt exists" and "no repetition found"
    are different claims and only one of them is a measurement.
    """
    if style not in REPETITION_STYLES:
        return {"repetition_status": "not_specified",
                "repetition_note": (f"the design document defines no repetition "
                                    f"prompt for {style}")}

    scores_edges = style == "alpha_masking"
    # Keyed by label. Repetition asks how often a label was re-highlighted, so
    # several instances sharing a label collapse to one counter here -- unlike
    # omission, which owes a sighting per instance.
    frequency: dict[str, int] = {e.label: 0 for e in
                                 checklist.items(include_edges=scores_edges)}
    unjustified_elements: list[str] = []
    unjustified_arrows: list[str] = []

    prompt_template = load_and_render("animation_repetition.yaml#frames", {
        "style_adapter": _adapter("animation_repetition.yaml", style),
        "output_schema": _adapter("animation_repetition.yaml", style, "schema"),
    }, root=PROMPTS_ROOT)

    def render_frequencies() -> str:
        return "\n".join(f"  {name}: {count}"
                         for name, count in frequency.items()) or "  (empty)"

    def step(position, path, label):
        prompt = prompt_template.replace("{frequencies}", render_frequencies())
        data = judge.ask_json(f"{prompt}\n\nThe frame provided is: {label}",
                              images=[Path(source_image), path],
                              tag=f"repetition_{style}")
        updates = data.get("frequency_updates_this_frame")
        touched: list[str] = []
        for bucket in ("blocks", "nodes", "edges"):
            values = updates.get(bucket) if isinstance(updates, dict) else None
            for entry in values or []:
                name = (entry.get("name") if isinstance(entry, dict)
                        else entry if isinstance(entry, str) else None)
                if isinstance(name, str) and name.strip() in frequency:
                    frequency[name.strip()] += 1
                    touched.append(name.strip())

        evaluation = data.get("repetition_evaluation") or {}
        flagged_el = _names(evaluation, "unjustified_freq_increase_element")
        flagged_ar = _names(evaluation, "unjustified_freq_increase_arrow")
        unjustified_elements.extend(n for n in flagged_el
                                    if n in frequency and n not in unjustified_elements)
        unjustified_arrows.extend(n for n in flagged_ar
                                  if n in frequency and n not in unjustified_arrows)
        return {"frame": label, "state_changes": touched,
                "justification_reasoning": evaluation.get("justification_reasoning"),
                "unjustified": flagged_el + flagged_ar}

    results, errors = fold_frames(frames, step)
    revisited = [n for n, c in frequency.items() if c >= 2]

    return {
        "repetition_status": "scored",
        "unnecessary_repetition_count_element": len(unjustified_elements),
        "unnecessary_repetition_count_arrow": len(unjustified_arrows),
        "repetition_rate": (len(unjustified_elements) / len(frequency)
                            if frequency else None),
        "repetition_unjustified_elements": unjustified_elements,
        "repetition_unjustified_arrows": unjustified_arrows,
        "repetition_revisited": revisited,
        "repetition_frequencies": frequency,
        "repetition_frame_detail": results,
        "repetition_errors": errors,
        **{f"repetition_{k}": v for k, v in frames.manifest().items()},
    }


# -- SC1 / SC2: the two banded, per-timestep judges -------------------------

def _banded(judge, prompt_file: str, metric: str, extra_keys: tuple[str, ...],
            step_frames_: list[Path], source_image: Path, style: str,
            xml_text: str, prior_summary_at) -> dict:
    """Shared driver for selection sensibility and granularity/pacing.

    One call per timestep, each seeing the figure, the previous frame and the
    current one -- the judge is told to infer what is new from that difference,
    which is why the current step's element ids are deliberately withheld.
    """
    total = len(step_frames_)
    # The rubric belongs to the system turn, not the user turn: the reference
    # `build_system_instruction()` concatenates intro + RUBRIC_TEXT, and only
    # the adapter and the output format are spliced into the user prompt. The
    # standalone `rubric` key is kept in the YAML for readability, but splicing
    # it here too would send the whole rubric twice.
    system = load_and_render(f"{prompt_file}#system", {}, root=PROMPTS_ROOT)
    output_format = load_and_render(f"{prompt_file}#output_format", {},
                                    root=PROMPTS_ROOT)
    adapter = _adapter(prompt_file, style)

    per_step: list[dict] = []
    errors: list[str] = []

    for index, current in enumerate(step_frames_, 1):
        user = load_and_render(f"{prompt_file}#user", {
            "timestep_idx": str(index), "total_steps": str(total),
            "style_name": style.replace("_", " ").title(),
            "style_adapter": adapter,
            "prior_summary": prior_summary_at(index), "xml_text": xml_text,
            "output_format": output_format,
        }, root=PROMPTS_ROOT)
        images = [Path(source_image)]
        if index > 1:
            images.append(step_frames_[index - 2])
        images.append(current)
        try:
            text = judge.ask_text(f"{system}\n\n{user}", images=images,
                                  tag=f"{metric}_{style}",
                                  accept=_has_final_band)
        except JudgeError as e:
            errors.append(f"t{index}: {e}")
            continue
        verdict = parse_bands(text, extra_keys)
        verdict["timestep"] = index
        verdict["frame"] = current.name
        per_step.append(verdict)

    valid = [v["final_band"] for v in per_step if v["is_valid"]]
    mean = _mean([float(b) for b in valid])
    return {
        metric: (mean / BAND_MAX) if mean is not None else None,
        f"{metric}_band_mean": mean,
        f"{metric}_steps_scored": len(valid),
        f"{metric}_steps_total": total,
        f"{metric}_steps_invalid": len(per_step) - len(valid),
        f"{metric}_step_detail": per_step,
        f"{metric}_errors": errors,
    }


def _banded_workers(judge) -> int:
    """How many timesteps to judge at once.

    Taken from the judge's own backend concurrency so one config knob controls
    both, rather than a second number here that can silently disagree with it.
    Falls back to 4 for a judge that does not expose one.
    """
    backend = getattr(judge, "_backend", None)
    return int(getattr(backend, "_max_concurrency", 4) or 4)


def _letter_band(value) -> int | None:
    """Ordinal for a letter band, accepting the shapes a judge actually emits.

    Seen in practice: "A", "a", "BAND A", "A - Fully Sensible". Anchored to the
    FIRST standalone A-E token so a trailing label ("A - ... Coherence") cannot
    be misread, and so prose that merely mentions another letter later does not
    win over the score itself.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # An older prompt's integer, already on the 0-4 scale.
        return int(value) if 0 <= int(value) <= 4 else None
    if not isinstance(value, str):
        return None
    match = re.search(r"\b([A-E])\b", value.strip().upper())
    return BAND_LETTER_VALUE.get(match.group(1)) if match else None


def parse_json_bands(text: str, keys: tuple[tuple[str, str], ...],
                     text_keys: tuple[tuple[str, str], ...] = ()) -> dict:
    """Recover a banded verdict from the JSON reply.

    Mirrors `parse_bands`: `is_valid` is False when `final_score` cannot be
    read, so an unparseable timestep is excluded from the mean rather than
    landing in it as a zero. A parse failure is missing data; zero is the
    worst possible verdict, and conflating them would drag a cell's score down
    for the judge's formatting rather than for the animation.
    """
    result = {"rationale": None, "final_band": None, "label": None,
              "is_valid": False, "newly_targeted_elements": None}
    for _, stored in keys:
        result[stored] = None

    data = _loads_json(text)
    if not isinstance(data, dict):
        return result

    for json_key, stored in keys:
        block = data.get(json_key)
        if isinstance(block, dict):
            result[stored] = _letter_band(block.get("score"))
            # The per-criterion rationale is the reason this metric is
            # inspectable at all; keep it beside its band.
            result[f"{stored}_rationale"] = block.get("rationale")
        else:
            result[stored] = _letter_band(block)

    for json_key, stored in text_keys:
        value = data.get(json_key)
        result[stored] = value if isinstance(value, str) else None

    result["final_band"] = _letter_band(data.get("final_score"))
    result["newly_targeted_elements"] = data.get("newly_targeted_elements")
    result["rationale"] = data.get("tie_breaking_rationale")
    result["is_valid"] = result["final_band"] is not None
    return result


def _loads_json(text: str):
    """Parse a JSON object out of a reply, fenced or not."""
    if not isinstance(text, str):
        return None
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...}, for a reply with prose around it.
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return None


def _has_final_score(text: str) -> bool:
    data = _loads_json(text)
    return isinstance(data, dict) and _letter_band(data.get("final_score")) is not None


def _banded_json(judge, prompt_file: str, metric: str,
                 keys: tuple[tuple[str, str], ...],
                 step_frames_: list[Path], source_image: Path, style: str,
                 xml_text: str,
                 text_keys: tuple[tuple[str, str], ...] = ()) -> dict:
    """`_banded` for the JSON/letter rubric.

    Kept separate rather than branching inside `_banded`: the two rubrics
    differ in their prompt layout, their reply format AND their scale
    direction, and a single function carrying all three forks is how the
    header version would eventually get parsed with the letter mapping.

    `prior_summary` is deliberately absent. The header prompt spliced a summary
    of earlier timesteps into every call; the JSON prompt names its four inputs
    explicitly and a running summary is not among them, so sending one would be
    scoring a prompt nobody wrote.
    """
    total = len(step_frames_)
    system = load_and_render(f"{prompt_file}#system", {}, root=PROMPTS_ROOT)
    adapter = _adapter(prompt_file, style)

    def judge_step(index: int) -> tuple[int, dict | None, str | None]:
        current = step_frames_[index - 1]
        user = load_and_render(f"{prompt_file}#user", {
            "style_adapter": adapter, "xml_text": xml_text,
            "timestep_idx": str(index), "total_steps": str(total),
        }, root=PROMPTS_ROOT)
        images = [Path(source_image)]
        if index > 1:
            images.append(step_frames_[index - 2])
        images.append(current)
        try:
            text = judge.ask_text(f"{system}\n\n{user}", images=images,
                                  tag=f"{metric}json_{style}",
                                  accept=_has_final_score)
        except JudgeError as e:
            return index, None, f"t{index}: {e}"
        verdict = parse_json_bands(text, keys, text_keys)
        verdict["timestep"] = index
        verdict["frame"] = current.name
        return index, verdict, None

    # TIMESTEPS ARE JUDGED CONCURRENTLY.
    #
    # Each call is independent: the prompt names the previous and current
    # frames explicitly and nothing carries state between steps, unlike the
    # header rubric which spliced in a running prior_summary. Sequentially this
    # was the study's wall-clock bottleneck -- measured at ~34s median per
    # call, a 19-timestep cell is ~22 minutes for one metric.
    #
    # The backend already guards itself with `Semaphore(max_concurrency)`, so
    # this cannot exceed the provider concurrency the config asked for; the
    # pool here just stops the driver from idling between calls.
    #
    # Results are reassembled BY TIMESTEP, not by completion order -- otherwise
    # `*_step_detail` would be shuffled, and the per-step detail is exactly
    # what makes a banded score auditable.
    per_step: list[dict] = []
    errors: list[str] = []
    workers = max(1, min(_banded_workers(judge), total))
    if workers == 1:
        results = [judge_step(i) for i in range(1, total + 1)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(judge_step, range(1, total + 1)))
    for index, verdict, error in sorted(results, key=lambda r: r[0]):
        if error:
            errors.append(error)
        else:
            per_step.append(verdict)

    valid = [v["final_band"] for v in per_step if v["is_valid"]]
    mean = _mean([float(b) for b in valid])
    return {
        metric: (mean / BAND_MAX) if mean is not None else None,
        f"{metric}_band_mean": mean,
        f"{metric}_steps_scored": len(valid),
        f"{metric}_steps_total": total,
        f"{metric}_steps_invalid": len(per_step) - len(valid),
        f"{metric}_step_detail": per_step,
        f"{metric}_errors": errors,
        f"{metric}_rubric": "letters",
    }



def narration_alignment(judge, source_image, step_frames_, style,
                        narrations: list[str], context_dump: str,
                        xml_text: str = "") -> dict:
    """NAS: does each caption describe the transition it accompanies? (Contributor 4)

    Unlike SSS and GPS, this node needs two inputs the banded driver does not
    carry, so it does not reuse `_banded_json`:

      * the per-timestep NARRATION, which is the thing under evaluation rather
        than context for it;
      * a SCIENTIFIC CONTEXT DUMP -- the paper's title, abstract, method and
        figure caption -- without which "does this caption add insight beyond
        the picture?" cannot be asked at all.

    A timestep with no narration is SKIPPED rather than scored: there is no
    caption to judge, and scoring it as a failure would conflate "the narrator
    said nothing here" with "the narrator said something wrong", which are
    different defects and only one of them is this metric's business.

    The prompt carries two ceilings the judge applies itself -- a purely visual
    caption caps at C, and the final band cannot exceed the alignment band --
    and returns the reasoning under `tie_breaking_rationale`, so a score can be
    audited against the rule that produced it.
    """
    total = len(step_frames_)
    # The FINAL prompt lists the XML schema as input 7 and asks the judge to
    # map targeted elements onto XML ids -- promptv1 never sent it.
    system = load_and_render("animation_narration_final.yaml#system", {}, root=PROMPTS_ROOT)
    adapter = _adapter("animation_narration_final.yaml", style)

    def judge_step(index: int):
        narration = (narrations[index - 1] if index - 1 < len(narrations) else "") or ""
        if not narration.strip():
            return index, None, None          # nothing to score, not a failure
        current = step_frames_[index - 1]
        user = load_and_render("animation_narration_final.yaml#user", {
            "style_adapter": adapter, "narration": narration,
            "context_dump": context_dump or "(no paper context available)",
            "xml_text": xml_text or "(no XML schema available)",
            "timestep_idx": str(index), "total_steps": str(total),
        }, root=PROMPTS_ROOT)
        images = [Path(source_image)]
        if index > 1:
            images.append(step_frames_[index - 2])
        images.append(current)
        try:
            text = judge.ask_text(f"{system}\n\n{user}", images=images,
                                  tag=f"nas_{style}", accept=_has_final_score)
        except JudgeError as e:
            return index, None, f"t{index}: {e}"
        verdict = parse_json_bands(text, (), NAS_FINAL_TEXT_KEYS)
        verdict["timestep"] = index
        verdict["frame"] = current.name
        verdict["narration"] = narration
        return index, verdict, None

    workers = max(1, min(_banded_workers(judge), total))
    if workers == 1:
        results = [judge_step(i) for i in range(1, total + 1)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(judge_step, range(1, total + 1)))

    per_step, errors, skipped = [], [], 0
    for index, verdict, error in sorted(results, key=lambda r: r[0]):
        if error:
            errors.append(error)
        elif verdict is None:
            skipped += 1
        else:
            per_step.append(verdict)

    valid = [v["final_band"] for v in per_step if v["is_valid"]]
    mean = _mean([float(b) for b in valid])
    return {
        "nas": (mean / BAND_MAX) if mean is not None else None,
        "nas_band_mean": mean,
        "nas_steps_scored": len(valid),
        "nas_steps_total": total,
        "nas_steps_unnarrated": skipped,
        "nas_steps_invalid": len(per_step) - len(valid),
        "nas_step_detail": per_step,
        "nas_errors": errors,
        "nas_rubric": "letters",
    }




def selection_sensibility_bands(judge, source_image, step_frames_, style,
                                xml_text) -> dict:
    # The FINAL prompt: no per-criterion letter grades, so the band-key map
    # is empty and the per-rule reasonings ride the text lane instead.
    return _banded_json(judge, "animation_selection_final.yaml", "sss",
                        (), step_frames_, source_image, style,
                        xml_text, text_keys=SSS_FINAL_TEXT_KEYS)


def granularity_pacing_bands(judge, source_image, step_frames_, style,
                             xml_text) -> dict:
    return _banded_json(judge, "animation_pacing_final.yaml", "gps",
                        (), step_frames_, source_image, style,
                        xml_text, text_keys=GPS_FINAL_TEXT_KEYS)


def selection_sensibility(judge, source_image, step_frames_, style, xml_text,
                          prior_summary_at) -> dict:
    return _banded(judge, "animation_selection.yaml", "sss", SSS_KEYS,
                   step_frames_, source_image, style, xml_text, prior_summary_at)


def granularity_pacing(judge, source_image, step_frames_, style, xml_text,
                       prior_summary_at) -> dict:
    return _banded(judge, "animation_pacing.yaml", "gps", GPS_KEYS,
                   step_frames_, source_image, style, xml_text, prior_summary_at)


# -- orchestration ---------------------------------------------------------

def prior_summary_builder(seq, xml):
    """A `step -> text` function naming everything targeted before that step.

    The two banded judges are given the cumulative history but deliberately
    NOT the current step's element ids -- they are meant to infer what is new
    from the frame difference, and handing them the answer would score the
    sequence file rather than the animation. So this stops one step short by
    construction.
    """
    from ..checklist import BUCKETS, sequence_view  # noqa: F401  (shared bucketing)

    by_id = {n.id: n for n in seq.nodes}
    per_step: list[list[str]] = []
    for node_id in seq.traversal:
        node = by_id.get(node_id)
        if node is None:
            continue
        if node.element_classes:
            ids = [i for ids in node.element_classes.values() for i in ids]
        else:
            ids = list(node.focus)
        per_step.append(ids)

    def at(step_index: int) -> str:
        seen: list[str] = []
        for ids in per_step[:max(0, step_index - 1)]:
            seen.extend(i for i in ids if i not in seen)
        return "\n".join(f"  - {i}" for i in seen) or "  (nothing yet -- this is the first step)"

    return at


def run(judge, *, source_image: Path, frames_dir: Path, style: str,
        checklist=None, xml_text: str = "", step_frames_: list[Path] | None = None,
        prior_summary_at=None, stages: tuple[str, ...] = STAGES,
        frame_px: int | None = None, cache_dir: Path | None = None,
        export_video: Path | None = None,
        video_source: str = "frames",
        rubric: str = "headers",
        narrations: list[str] | None = None,
        context_dump: str = "") -> dict:
    """Score one (sample, style) cell across the requested stages.

    Every stage runs that can; nothing is gated. The gates' *inputs* are
    recorded so a threshold can be chosen later from the distribution, which is
    the only honest way to pick one when no document states it.
    """
    from ..frames import DEFAULT_LONG_EDGE, FrameError, frame_set

    long_edge = frame_px or DEFAULT_LONG_EDGE
    record: dict = {"suite": "animation", "style": style,
                    "stages_requested": list(stages), "stages_run": [],
                    "stages_skipped": {}}

    def prepared(policy_map, tag):
        return frame_set(frames_dir, policy_map[style], long_edge,
                         cache_dir=(Path(cache_dir) / tag) if cache_dir else None)

    def attempt(name, fn):
        if name not in stages:
            return
        try:
            record.update(fn())
            record["stages_run"].append(name)
        except FrameError as e:
            record["stages_skipped"][name] = str(e)
        except Exception as e:                      # noqa: BLE001 - one stage
            record["stages_skipped"][name] = f"{type(e).__name__}: {e}"

    attempt("vfs", lambda: visual_fidelity(
        judge, source_image, prepared(VFS_POLICY, "vfs"), style))
    attempt("ascs", lambda: style_compliance(
        judge, source_image, prepared(ASCS_POLICY, "ascs"), style))

    def _video_fidelity() -> dict:
        """Resolve which mp4 to judge, then judge it.

        `frames` (default) re-times the ASCS deck to 1 fps -- ASCS_POLICY, not
        VFS_POLICY, because the video must show the whole animation and
        VFS_POLICY is "last" for four of the five styles. See video.py for why
        the export is not used by default.
        """
        from ..video import JUDGE_VIDEO_FPS, judge_video

        if video_source == "export":
            if not export_video or not Path(export_video).exists():
                raise FileNotFoundError(f"no exported video at {export_video}")
            path = Path(export_video)
            meta = {"source": "export", "path": str(path),
                    "bytes": path.stat().st_size, "fps": None}
        else:
            deck = prepared(ASCS_POLICY, "ascs")
            out_dir = (Path(cache_dir) / "video") if cache_dir else Path(frames_dir)
            path = judge_video(deck, out_dir / f"judge_{JUDGE_VIDEO_FPS}fps.mp4")
            meta = {"source": "frames", "path": str(path),
                    "bytes": path.stat().st_size, "fps": JUDGE_VIDEO_FPS,
                    "frame_count": len(deck.paths)}
        return video_fidelity(judge, source_image, path, style, meta)

    attempt("vfs_video", _video_fidelity)

    def _slow_video() -> tuple[Path, dict]:
        """The 4x-slowed mp4 both band judges see.

        Two choices are load-bearing:

        DENSE DECK IF ONE EXISTS. `frames_dense/` is written by
        scripts/render_dense_decks.py at a sampling rate high enough to contain
        the motion between stages. The default `frames/` deck is a 2 Hz
        sampling in which a slide's transit is simply absent -- measured, a
        sliding animation holds 26 of its 194 distinct states there -- and the
        bounding-box style prompts turn entirely on seeing that transit.

        0.5 FPS. Four times slower than the authored `exporter.fps: 2`, so each
        frame occupies two seconds and Gemini's 1 fps inline sampler lands on
        every one of them at least twice. Slowing playback cannot conjure a
        frame the deck never had, which is why the dense deck comes first.
        """
        from ..video import SLOWDOWN_VIDEO_FPS, judge_video
        from ..frames import frame_set

        dense = Path(frames_dir).parent / "frames_dense"
        use_dense = dense.is_dir() and any(dense.glob("*.png"))
        source_dir = dense if use_dense else Path(frames_dir)
        deck = frame_set(source_dir, "all", long_edge,
                         cache_dir=(Path(cache_dir) / "slowdeck") if cache_dir else None)
        out_dir = (Path(cache_dir) / "video") if cache_dir else Path(frames_dir)
        path = judge_video(deck, out_dir / f"judge_slow_{SLOWDOWN_VIDEO_FPS}fps.mp4",
                           fps=SLOWDOWN_VIDEO_FPS)
        return path, {"source": "frames_dense" if use_dense else "frames",
                      "path": str(path), "bytes": path.stat().st_size,
                      "fps": SLOWDOWN_VIDEO_FPS, "frame_count": len(deck.paths)}

    def _vfs_band() -> dict:
        path, meta = _slow_video()
        return video_fidelity_bands(judge, source_image, path, style, meta)

    def _ascs_video() -> dict:
        path, meta = _slow_video()
        return style_compliance_video(judge, source_image, path, style, meta)

    attempt("vfs_band", _vfs_band)
    attempt("ascs_video", _ascs_video)

    if checklist is not None:
        attempt("omission", lambda: omitted_elements(
            judge, source_image, prepared(WALK_POLICY, "walk"), style, checklist))
        attempt("repetition", lambda: unnecessary_repetition(
            judge, source_image, prepared(WALK_POLICY, "walk"), style, checklist))
    else:
        for name in ("omission", "repetition"):
            if name in stages:
                record["stages_skipped"][name] = "no element checklist"

    # `rubric="letters"` selects the JSON/A-E prompts; "headers" keeps the
    # `### FINAL_BAND: <0-4>` ones every stored score was produced by. The
    # letter path needs no prior_summary -- its prompt does not take one -- so
    # it is not gated on prior_summary_at being supplied.
    # NAS needs the captions themselves plus the paper's own words. Gated on
    # narrations rather than on `rubric`, because a run can have frames and a
    # sequence but no narration stage -- in which case there is nothing to
    # score and saying so is better than reporting zeros.
    if step_frames_ and narrations:
        attempt("nas", lambda: narration_alignment(
            judge, source_image, step_frames_, style, narrations, context_dump,
            xml_text=xml_text))
    elif "nas" in stages:
        record["stages_skipped"]["nas"] = (
            "no per-timestep narration for this cell; run the narrate stage"
            if step_frames_ else
            "frame count does not match the sequence's timestep count")

    if step_frames_ and rubric == "letters":
        attempt("sss", lambda: selection_sensibility_bands(
            judge, source_image, step_frames_, style, xml_text))
        attempt("gps", lambda: granularity_pacing_bands(
            judge, source_image, step_frames_, style, xml_text))
    elif step_frames_ and prior_summary_at is not None:
        attempt("sss", lambda: selection_sensibility(
            judge, source_image, step_frames_, style, xml_text, prior_summary_at))
        attempt("gps", lambda: granularity_pacing(
            judge, source_image, step_frames_, style, xml_text, prior_summary_at))
    else:
        for name in ("sss", "gps"):
            if name in stages:
                record["stages_skipped"][name] = (
                    "frame count does not match the sequence's timestep count, "
                    "so no frame can be attributed to a step")

    record["would_eliminate"] = _would_eliminate(record)
    return record


def _would_eliminate(record: dict) -> list[str]:
    """Which gates would have fired, had a threshold been set.

    Only ASCS can answer today: its verdict is categorical, so "any frame
    discarded" is a rule the document itself implies. VFS and omission produce
    numbers with no stated cutoff, so they are listed as undecided rather than
    guessed -- a fabricated threshold here would zero every metric below it.
    """
    fired: list[str] = []
    if record.get("ascs_pass") is False:
        fired.append("ascs")
    return fired
