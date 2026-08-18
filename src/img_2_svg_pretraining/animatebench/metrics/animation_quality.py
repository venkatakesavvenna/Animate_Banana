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

import re
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

# Style rules that no single-frame call can answer, recorded so the gap stays
# visible instead of reading as a pass. The first two have no key in their own
# schema at all -- the document asks the judge to enforce a rule it gave it no
# field to report. The third is inherently sequence-level: "there must be at
# least one frame where the box is between elements" is a property of the whole
# animation, and belongs in an aggregation nobody has specified yet.
UNENFORCEABLE_RULES = {
    "hopping_bounding_box": ["Mobile Boxes, Static Elements (CRITICAL): no key "
                             "in the document's own schema"],
    "sliding_bounding_box": ["Mobile Boxes, Static Elements (CRITICAL): no key "
                             "in the document's own schema",
                             "Hopping Boxes (HIGHLY CRITICAL): sequence-level, "
                             "unanswerable from one frame"],
}

STAGES = ("vfs", "ascs", "omission", "sss", "gps", "repetition")

# The two banded judges answer in section headers, not JSON. Regexes are the
# design documents' own.
_BAND_RE = {
    "criterion_a_band": re.compile(r"###\s*CRITERION_A_BAND:\s*([0-4])", re.I),
    "criterion_b_band": re.compile(r"###\s*CRITERION_B_BAND:\s*([0-4])", re.I),
    "volume_band": re.compile(r"###\s*VOLUME_BAND:\s*([0-4])", re.I),
    "complexity_band": re.compile(r"###\s*COMPLEXITY_BAND:\s*([0-4])", re.I),
    "final_band": re.compile(r"###\s*FINAL_BAND:\s*([0-4])", re.I),
    "label": re.compile(r"###\s*LABEL:\s*(.+)", re.I),
}
_RATIONALE_RE = re.compile(r"###\s*RATIONALE\s*(.*?)(?=###\s*[A-Z_]+_BAND)",
                           re.I | re.S)

SSS_KEYS = ("criterion_a_band", "criterion_b_band")
GPS_KEYS = ("volume_band", "complexity_band")

BAND_MAX = 4.0
VFS_MAX = 10.0


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

    def step(position, path, label):
        prompt = prompt_template.replace(
            "{frame_context}", f"frame {position + 1} of {total} ({label})")
        data = judge.ask_json(prompt, images=[Path(source_image), path],
                              tag=f"ascs_{style}")
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
        "ascs_pass": (not discarded) if judged else None,
        "ascs_frames_judged": len(judged),
        "ascs_frames_discarded": len(discarded),
        "ascs_discarded_frames": discarded,
        "ascs_frame_detail": results,
        "ascs_errors": errors,
        "ascs_unenforced_rules": UNENFORCEABLE_RULES.get(style, []),
        **{f"ascs_{k}": v for k, v in frames.manifest().items()},
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
    remaining = {
        "blocks": list(checklist.blocks),
        "nodes": list(checklist.nodes),
        "edges": list(checklist.edges) if scores_edges else [],
    }
    prompt_template = load_and_render("animation_omission.yaml#frames", {
        "style_adapter": _adapter("animation_omission.yaml", style),
        "output_schema": _adapter("animation_omission.yaml", style, "schema"),
    }, root=PROMPTS_ROOT)

    def render_remaining() -> str:
        lines = [f"  {b}: {remaining[b]}" for b in ("blocks", "nodes", "edges")
                 if scores_edges or b != "edges"]
        return "\n".join(lines)

    def step(position, path, label):
        prompt = prompt_template.replace("{remaining}", render_remaining())
        data = judge.ask_json(f"{prompt}\n\nThe frame provided is: {label}",
                              images=[Path(source_image), path],
                              tag=f"omission_{style}")
        popped = data.get("elements_popped_in_this_frame")
        this_frame: dict[str, list[str]] = {}
        for bucket in ("blocks", "nodes", "edges"):
            if bucket == "edges" and not scores_edges:
                continue
            # Only names actually outstanding may be popped: a judge naming
            # something already gone, or never on the list, must not shrink it.
            hits = [n for n in _names(popped, bucket) if n in remaining[bucket]]
            for name in hits:
                remaining[bucket].remove(name)
            if hits:
                this_frame[bucket] = hits
        return {"frame": label, "reasoning": data.get("reasoning"),
                "popped": this_frame}

    results, errors = fold_frames(frames, step)
    element_omissions = remaining["blocks"] + remaining["nodes"]

    return {
        "element_omission_count": len(element_omissions),
        "arrow_omission_count": len(remaining["edges"]) if scores_edges else 0,
        "omission_rate": (len(element_omissions) / checklist.total(False)
                          if checklist.total(False) else None),
        "omission_checklist_size": checklist.total(scores_edges),
        "omission_elements_remaining": element_omissions,
        "omission_arrows_remaining": remaining["edges"],
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
    frequency: dict[str, int] = {name: 0 for name in
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
        frame_px: int | None = None, cache_dir: Path | None = None) -> dict:
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

    if checklist is not None:
        attempt("omission", lambda: omitted_elements(
            judge, source_image, prepared(WALK_POLICY, "walk"), style, checklist))
        attempt("repetition", lambda: unnecessary_repetition(
            judge, source_image, prepared(WALK_POLICY, "walk"), style, checklist))
    else:
        for name in ("omission", "repetition"):
            if name in stages:
                record["stages_skipped"][name] = "no element checklist"

    if step_frames_ and prior_summary_at is not None:
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
