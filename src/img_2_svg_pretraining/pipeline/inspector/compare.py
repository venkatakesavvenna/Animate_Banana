"""Side-by-side comparison viewer for an AnimateBench run.

Shows, for one sample and one animation style, four things at once:

  1. the source figure (ground truth input),
  2. the benchmark's own reference animation, shipped in the sample bundle,
  3-5. our pipeline's animation for each configured model.

The point is qualitative: play them together and see which pipeline tells the
better story. Numeric metrics come later.

Each model is a separate pipeline config, so its artifacts live under its own
cache lineage. This resolves them by loading every config and asking its
`CachePaths` where things are, rather than guessing at directory names -- when
a lineage component changes, the paths move, and only the config knows where
to.

Run inside the container:
    python -m img_2_svg_pretraining.pipeline.inspector.compare --port 7861

Port 7861 keeps this separate from the single-run inspector on 7860, so both
can be open at once. Both ports must be published at container creation.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

from flask import Flask, abort, jsonify, request, send_file

# Module level, unlike the lazy `descriptions, results` import inside
# `_metrics`: `_TABLE_SUITES` below is evaluated at import time. Safe to hoist
# because `descriptions` imports nothing itself -- it is pure metric metadata.
from img_2_svg_pretraining.animatebench import descriptions

from img_2_svg_pretraining.pipeline.cache import CachePaths
from img_2_svg_pretraining.pipeline.config import load_config
from img_2_svg_pretraining.pipeline.samples import discover_samples
from img_2_svg_pretraining.pipeline.schema import AnimationSequence
from img_2_svg_pretraining.pipeline.styles import STYLES

app = Flask(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "configs"

# The three comparison configs, in display order. Label -> config file.
DEFAULT_CONFIGS = {
    "Gemini 3.6 Flash": "bench_gemini.yaml",
    "Gemma 4 31B": "bench_gemma4.yaml",
    "Qwen3.6 27B": "bench_qwen.yaml",
}

# Styles the benchmark ships, in the order its own outputs list them.
BENCH_STYLES = ["progressive_reveal", "colour_pop", "alpha_masking",
                "hopping_bounding_box", "sliding_bounding_box"]

STATE: dict = {"models": {}, "samples": [], "by_id": {},
               "styles": BENCH_STYLES, "cells": None,
               "suites": None, "metrics": None}


def _init(config_files: dict[str, str], dataset_root: str | None,
          cells: list[dict] | None = None,
          extra_roots: list[str] | None = None) -> None:
    """Load each label's config(s).

    A label may name SEVERAL configs separated by `|`, which is how one model
    carries more than one target:

        --configs "Gemini 3.7=bench_v3_or.yaml|bench_v3_or_svg.yaml"

    They become `variants` keyed by each config's own `target`, so the metrics
    view can put TikZ and SVG side by side in one table instead of forcing a
    reader to flip between two models that differ only in representation. The
    first config listed stays the label's primary, which is what every
    single-target code path already reads.
    """
    models = {}
    samples = None
    for label, spec in config_files.items():
        variants, primary = {}, None
        for filename in [f.strip() for f in str(spec).split("|") if f.strip()]:
            path = CONFIG_DIR / filename
            if not path.exists():
                print(f"  ! {label}: no config at {path}, skipping")
                continue
            cfg = load_config(path)
            entry = {"cfg": cfg, "file": filename, "target": cfg.target,
                     "model": cfg.backend_model(cfg.agent("designer").backend)}
            # Two configs with the same target would silently shadow each
            # other; say so rather than dropping one.
            if cfg.target in variants:
                print(f"  ! {label}: two configs both target '{cfg.target}' "
                      f"({variants[cfg.target]['file']}, {filename}); keeping the first")
                continue
            variants[cfg.target] = entry
            primary = primary or entry
            if samples is None:
                samples = discover_samples(dataset_root or cfg.dataset_root)

        if not primary:
            continue
        models[label] = {**primary, "variants": variants}

    if not models:
        raise SystemExit("no comparison configs could be loaded")

    samples = samples or []
    # A study can span dataset ROOTS. Six samples here exist only under
    # animatebench_v2 (dropped from v3). Their configs cannot be attached as
    # `|` variants because variants are keyed by TARGET and both are 'svg', so
    # the second is discarded before discovery. Hence an explicit list.
    for root in (extra_roots or []):
        known = {s.id for s in samples}
        samples = samples + [s for s in discover_samples(root)
                             if s.id not in known]
    if cells:
        # A STUDY VIEW: a fixed list of (sample, style) cells rather than the
        # whole dataset crossed with every style. Each sample is pinned to the
        # one style it was chosen for, so the UI cannot land on a (sample,
        # style) pair that was never generated and read the resulting empty
        # panel as a pipeline failure.
        #
        # Order follows `cells`, not `discover_samples`: the list is the
        # study's own, and its order is meaningful to whoever wrote it.
        known = {s.id: s for s in samples}
        missing = [c["id"] for c in cells if c["id"] not in known]
        if missing:
            raise SystemExit(
                "--cells names sample(s) not in the dataset: "
                + ", ".join(missing))
        samples = [known[c["id"]] for c in cells]
        STATE["styles"] = sorted({c["style"] for c in cells})

    STATE.update(models=models, samples=samples,
                 by_id={s.id: s for s in samples}, cells=cells)


def _paths_for(label: str, style: str) -> CachePaths:
    """CachePaths for one model at one style.

    The style is part of the sequence lineage, so it must be set on the config
    before paths are resolved -- otherwise every style would resolve to the
    config's default and the viewer would show the same video five times.
    """
    cfg = STATE["models"][label]["cfg"]
    cfg.style = style
    cfg.raw["animation_style"] = style
    return CachePaths.from_config(cfg)


def _paths_for_target(label: str, style: str, target: str | None) -> CachePaths:
    """CachePaths for one model at one style and one TARGET.

    `?target=` selects between a label's tikz and svg variants. Without it the
    primary is used, which is what every pre-variant caller expects. An
    unknown target falls back to the primary rather than 404ing: a stale
    bookmark should show the default animation, not an error.
    """
    model = STATE["models"][label]
    variants = model.get("variants") or {}
    entry = variants.get(target) if target else None
    cfg = (entry or model)["cfg"]
    cfg.style = style
    cfg.raw["animation_style"] = style
    return CachePaths.from_config(cfg)


def _reference(sample_id: str) -> dict:
    """The benchmark's own outputs for this sample, as imported."""
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        return {}
    ref = Path(sample.directory) / "reference"
    index = ref / "index.json"
    if not index.exists():
        return {}
    try:
        return json.loads(index.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _metrics(cfg, config_file: str, style: str, sample_id: str) -> dict:
    """AnimateBench scores for one panel, with each metric's description.

    Reads the eval records the metrics CLI writes; a panel that has not been
    scored yet simply reports nothing rather than blocking the videos.
    """
    from img_2_svg_pretraining.animatebench import descriptions, results

    root = _evals_root(cfg)
    video_root = _video_evals_root(cfg)
    qwen_root = _qwen_evals_root(cfg)
    config_name = Path(config_file).stem

    suites: list[dict] = []
    for suite in ("stage1", "xml", "sequence", "stage3", "animation"):
        record = results.read_record(
            results.suite_path(root, config_name, style, sample_id, suite))
        # The video judge writes to its own root, so the animation suite is
        # the union of two records. A cell scored ONLY by the video judge
        # still gets a row -- otherwise the one metric it has would be
        # invisible.
        if suite == "animation" and video_root is not None:
            extra = results.read_record(results.suite_path(
                video_root, config_name, style, sample_id, suite))
            if extra is not None:
                merged = dict(record or {})
                for key, value in extra.items():
                    if key.startswith("vfs_video"):
                        merged[key] = value
                record = merged
        if record is None:
            continue
        # The second-opinion judge scored the SAME metrics on the same cells,
        # so its numbers ride alongside as `second` rather than replacing
        # anything. Two judges measurably disagree here (component_accuracy
        # r=0.43, rendering_fidelity 0.38), which is the finding -- collapsing
        # them to one column would hide exactly what is worth seeing.
        second = None
        if qwen_root is not None:
            second = results.read_record(results.suite_path(
                qwen_root, config_name, style, sample_id, suite))

        entries = []
        for key in _suite_columns(suite):
            if key not in record or record[key] is None:
                continue
            entry = {"key": key, "value": record[key],
                     "sort": _sort_key(key, record[key], record),
                     # Distinguishes "judged, and this is the number" from
                     # "never judged" -- which `ZERO_WHEN_ABSENT` otherwise
                     # renders identically as 0 for omission_rate.
                     "measured": True,
                     **descriptions.describe(key)}
            if second is not None and second.get(key) is not None:
                entry["second"] = second[key]
            entries.append(entry)
        if entries or record.get("error"):
            row = {"suite": suite, "metrics": entries,
                   "error": record.get("error")}
            if second is not None:
                row["second_judge"] = (second.get("provenance") or {}).get(
                    "judge_model")
            suites.append(row)
    return {"available": bool(suites), "suites": suites}


def _sequence_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        seq = AnimationSequence.load(path)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {
        "steps": len(seq.nodes),
        "traversal_style": seq.traversal_style,
        "narrated": sum(1 for n in seq.nodes if n.narrative),
        "elements": len(seq.focus_ids()),
        "bench_schema": any(n.element_classes for n in seq.nodes),
    }


# Suite -> column group label, matching the four pipeline stages. Defined in
# `descriptions` so this view and `animatebench.aggregate` cannot disagree
# about what the scoreboard's columns are.
_TABLE_SUITES = descriptions.TABLE_SUITES


# Metrics rendered as 0 rather than blank when the record has no value.
#
# Requested explicitly for the omission rate. Worth stating the cost plainly:
# this CONFLATES "measured, and nothing was omitted" with "never measured",
# and the table cannot tell the reader which it is looking at. It is safe here
# only because the study's question is about the other four metrics; do not
# extend this to a metric anyone intends to draw a conclusion from without
# also rendering the distinction.
ZERO_WHEN_ABSENT = {"omission_rate": 0}


def _table_suites():
    """The suites this viewer shows, in order.

    A study viewer restricted with `--suites` shows only the suites it is
    about. Restricting HERE rather than in `descriptions.TABLE_SUITES` is
    deliberate: that module is shared with every other viewer on this machine,
    and narrowing it would silently strip columns from tables nobody asked to
    change.
    """
    only = STATE.get("suites")
    if not only:
        return _TABLE_SUITES
    return [(s, label) for s, label in _TABLE_SUITES if s in only]


def _suite_columns(suite: str) -> list[str]:
    """The metric keys shown for one suite, honouring `--metrics`."""
    keys = descriptions.ordered(suite)
    only = STATE.get("metrics")
    return [k for k in keys if k in only] if only else keys

# Metric -> the record keys holding the evidence behind its score. These are
# the per-element detail lists every metric writes alongside its number and
# that the panel/scoreboard views drop; the explain view is the one place
# they surface, so a reviewer can see *which* elements produced the score.
_EVIDENCE: dict[str, list[str]] = {
    "csr": ["compile_log"],
    "component_accuracy": ["component_per_class", "component_verdicts",
                           "component_unjudged", "component_ignored_ids",
                           "confusion_text_vs_child",
                           "confusion_raster_over_trigger"],
    "rendering_fidelity": ["rendering_fidelity_breakdown",
                           "rendering_fidelity_notes",
                           "rendering_fidelity_note"],
    "paa": ["parent_detail"],
    "matched_coverage": ["matched", "scorable_gt_elements"],
    "edge_f1": ["edge_precision", "edge_recall", "gt_edges",
                "pred_edges_comparable", "matched_edges", "missed_gt_edges",
                "spurious_pred_edges", "pred_edges_uncontractable"],
    "depth_violation_rate": ["depth_violations", "elements_checked"],
    "vfs": ["vfs_raw", "vfs_frame_scores", "vfs_frames", "vfs_errors",
            "vfs_frame_policy", "vfs_frames_judged", "vfs_frames_available"],
    "ascs_pass": ["ascs_frames_judged", "ascs_frames_discarded",
                  "ascs_discarded_frames", "ascs_unenforced_rules",
                  "ascs_frame_detail", "ascs_errors"],
    "omission_rate": ["element_omission_count", "omission_checklist_size",
                      "omission_elements_remaining", "omission_frame_detail",
                      "omission_errors"],
    "arrow_omission_count": ["omission_arrows_remaining",
                             "omission_scores_edges"],
    # The video judges' whole value is the prose: a band or an ACCEPT/DISCARD
    # with nothing behind it is unfalsifiable. `vfs_band_rationale` carries one
    # observation per fidelity criterion and `ascs_video_rationale` carries the
    # timestamped violation, which is the part a reader can go and check
    # against the video itself.
    #
    # `*_source` is here too, and matters more than it looks: it says whether
    # the judge saw the dense deck or the sparse one. A hop/slide verdict from
    # a sparse deck was reached without the evidence that distinguishes them.
    "vfs_band": ["vfs_band_ordinal", "vfs_band_pass", "vfs_band_summary",
                 "vfs_band_rationale", "vfs_band_source",
                 "vfs_band_frame_count", "vfs_band_fps", "vfs_band_errors"],
    "ascs_video": ["ascs_video_verdict", "ascs_video_pass",
                   "ascs_video_rationale", "ascs_video_source",
                   "ascs_video_frame_count", "ascs_video_fps",
                   "ascs_video_errors"],
    "sss": ["sss_band_mean", "sss_steps_scored", "sss_steps_total",
            "sss_steps_invalid", "sss_step_detail", "sss_errors",
            "sss_rubric"],
    "gps": ["gps_band_mean", "gps_steps_scored", "gps_steps_total",
            "gps_steps_invalid", "gps_step_detail", "gps_errors",
            "gps_rubric"],
    "nas": ["nas_band_mean", "nas_steps_scored", "nas_steps_total",
            "nas_steps_unnarrated", "nas_steps_invalid", "nas_step_detail",
            "nas_errors", "nas_rubric"],
    "repetition_rate": ["repetition_status", "repetition_note",
                        "unnecessary_repetition_count_element",
                        "unnecessary_repetition_count_arrow",
                        "repetition_unjustified_elements",
                        "repetition_revisited", "repetition_frame_detail",
                        "repetition_errors"],
    "coverage_recall": ["missed_groups", "gt_animated_unmatchable",
                        "gt_animated", "gt_animated_matchable"],
    "coverage_precision": ["covered_groups", "pred_animated_unmatchable",
                           "pred_animated", "pred_animated_matchable"],
    "tof": ["tof_detail"],
    "sscr_pass": ["sscr_violations"],
    "dovr": ["dovr_violations", "edges_tested", "edges_untestable"],
    "anim_csr": ["anim_compiles", "raw_compiles", "repair_rescued",
                 "anim_compile_log"],
    "aif": ["aif_lines_touched", "aif_lines_added", "aif_note"],
}


def _evals_root(cfg) -> Path:
    """Where eval records live.

    Normally beside the artifacts, but the pipeline cache is root-owned
    wherever a root container wrote it first, so `run_eval --evals-root` can
    put records somewhere an ordinary user can write. `ANIMATEBENCH_EVALS_ROOT`
    points this viewer at the same place, and must match whatever that run
    used -- otherwise the tab silently shows no scores for a sample that has
    them.
    """
    from img_2_svg_pretraining.animatebench import results

    override = os.environ.get("ANIMATEBENCH_EVALS_ROOT")
    if override:
        return Path(override)
    return results.evals_root(cfg.cache_root, cfg.dataset_root.name)


def _video_evals_root(cfg) -> Path | None:
    """Where the video-judge records live, if they were written separately.

    `vfs_video` is scored by a different judge on a different modality, into
    its own `--evals-root` so a partial re-run can never relabel the frame
    judge's provenance on the records it did not produce. That isolation is
    deliberate and stays -- but the viewer still has to show both numbers in
    one row, so it reads the second root here and merges on the way out.

    Returns None when no such root exists, which is the normal state for a
    dataset the video judge has not been run over. The animation suite then
    renders exactly as it did before.
    """
    override = os.environ.get("ANIMATEBENCH_VIDEO_EVALS_ROOT")
    if override:
        root = Path(override)
        return root if root.is_dir() else None
    default = _evals_root(cfg).parent / "evals_video_judge"
    return default if default.is_dir() else None


def _qwen_evals_root(cfg) -> Path | None:
    """Where the second-opinion (Qwen) intermediate records live, if scored.

    DIFFERENT IN KIND FROM `_video_evals_root`, and the difference decides how
    it may be used. The video judge scores a metric nothing else produces
    (`vfs_video`), so merging its keys into the animation record adds
    information and cannot overwrite anything. Qwen scored the SAME stage1 and
    xml metrics Gemini already scored -- `csr`, `component_accuracy`,
    `rendering_fidelity`, `edge_f1` -- on the same cells.

    So this root is exposed for COMPARISON, never merged over the primary. The
    two judges were measured to disagree substantially on exactly these
    metrics (edge_f1 r=0.76 but component_accuracy 0.43, rendering_fidelity
    0.38, matched_coverage 0.14), so silently letting one overwrite the other
    would replace a published number with a differently-derived one and leave
    the provenance line claiming the wrong model.

    Returns None when no such root exists, which is the normal state.
    """
    override = os.environ.get("ANIMATEBENCH_QWEN_EVALS_ROOT")
    if override:
        root = Path(override)
        return root if root.is_dir() else None
    default = _evals_root(cfg).parent / "evals_qwen_judge"
    return default if default.is_dir() else None


def _local_candidates(stored: str | None, cache_root: Path) -> list[Path]:
    """Where a recorded artifact path might actually live on this machine.

    Eval records written inside the container hold `/code/...` paths, and
    `/code` does not exist outside it. Rather than guess how many segments to
    strip, re-anchor on the local cache root: the tail of the stored path from
    `cache/` onward is stable across both layouts.
    """
    if not stored:
        return []
    candidates = [Path(stored)]
    parts = Path(stored).parts
    if "cache" in parts:
        tail = parts[parts.index("cache") + 1:]
        candidates.append(Path(cache_root).joinpath(*tail))
    return candidates


# -- sort keys -------------------------------------------------------------
#
# ONE NUMBER PER METRIC, ALWAYS HIGHER-IS-BETTER. Sorting must not be derived
# from `descriptions.describe(key)["better"]` plus the raw stored value, because
# for the banded metrics those two disagree and the disagreement is silent:
#
#   `vfs_band_ordinal` is 0=BAND A (best) .. 3=BAND D (worst)  -- LOWER is better
#   `descriptions.py` declares vfs_band  better="higher"
#
# A comparator trusting `better` therefore sorts BAND D to the top of the "best"
# list. No error, no crash, and the table looks entirely normal -- you click
# "sort by fidelity", read the top rows expecting the best animations, and are
# shown the worst. To make it worse, a DIFFERENT metric in the same codebase uses
# the opposite convention on purpose (`BAND_LETTER_VALUE`, A=4, higher better,
# for the SSS/GPS letter rubric), so there is no single rule to apply globally.
#
# Hence: normalise here, next to the code that owns the band definitions, and
# ship the result alongside the display value. `metricClass` still colours from
# the raw value; only ordering uses this.
def _sort_key(key: str, value, record: dict):
    """A higher-is-better float for `key`, or None if it cannot be ordered."""
    if key == "vfs_band":
        ordinal = record.get("vfs_band_ordinal")
        # 3 - ordinal: BAND A (0) -> 3 best, BAND D (3) -> 0 worst.
        return None if ordinal is None else float(3 - ordinal)
    if key == "ascs_video":
        passed = record.get("ascs_video_pass")
        return None if passed is None else (1.0 if passed else 0.0)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        # `better: "lower"` metrics (omission_rate) are negated so that every
        # sort key in the table points the same way.
        better = descriptions.describe(key).get("better")
        return float(-value) if better == "lower" else float(value)
    return None


def _evidence_for(key: str, record: dict) -> dict:
    """The stored evidence behind one metric, empties dropped."""
    out = {}
    for name in _EVIDENCE.get(key, []):
        value = record.get(name)
        if value is None or value == "" or value == [] or value == {}:
            continue
        out[name] = value
    return out


@app.get("/api/table/<label>/<style>")
def api_table(label, style):
    """Paper-style scoreboard: rows = samples, columns = stage -> metrics.

    One table per (model, style), since a style change is a different
    pipeline run end to end. Reuses the same eval records _metrics() reads
    for the per-panel view, just pivoted into one row per sample.
    """
    from img_2_svg_pretraining.animatebench import descriptions

    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    cfg = STATE["models"][label]["cfg"]
    config_file = STATE["models"][label]["file"]

    groups = []
    for suite, suite_label in _table_suites():
        keys = _suite_columns(suite)
        groups.append({
            "suite": suite, "label": suite_label,
            "columns": [{"key": k, **descriptions.describe(k)} for k in keys],
        })

    # One row per (sample, TARGET). A label carrying both a tikz and an svg
    # config puts both in THIS table, adjacent, rather than making a reader
    # open a second table to compare representations.
    variants = STATE["models"][label].get("variants") or {
        cfg.target: {"cfg": cfg, "file": config_file, "target": cfg.target}}
    targets = sorted(variants, key=lambda tg: (tg != "tikz", tg))

    # In a study view every sample has exactly ONE style, so a per-style table
    # would split five cells across three tables that can never be compared
    # side by side. Here the whole cell list goes into one table and the style
    # travels with the row instead of with the table.
    cells_spec = STATE.get("cells")
    if cells_spec:
        pairs = [(STATE["by_id"][c["id"]], c["style"]) for c in cells_spec
                 if c["id"] in STATE["by_id"]]
    else:
        pairs = [(s, style) for s in STATE["samples"]]

    rows = []
    for sample, row_style in pairs:
        for target in targets:
            entry = variants[target]
            m = _metrics(entry["cfg"], entry["file"], row_style, sample.id)
            by_suite = {s["suite"]: s for s in m["suites"]}
            cells = {}
            for suite, _ in _table_suites():
                s_entry = by_suite.get(suite)
                if s_entry is None:
                    cells[suite] = {"error": None,
                                    "values": dict(ZERO_WHEN_ABSENT),
                                    "sort": {},
                                    "measured": {k: False for k in ZERO_WHEN_ABSENT}}
                    continue
                # `second` rides in its own map. Flattening both into
                # `values` would make a second judge's figure indistinguishable
                # from the primary's in a table whose whole purpose is to be
                # read as one model's scoreboard.
                second = {e["key"]: e["second"] for e in s_entry["metrics"]
                          if e.get("second") is not None}
                values = {e["key"]: e["value"] for e in s_entry["metrics"]}
                sorts = {e["key"]: e["sort"] for e in s_entry["metrics"]
                         if e.get("sort") is not None}
                measured = {e["key"]: True for e in s_entry["metrics"]}
                for key, default in ZERO_WHEN_ABSENT.items():
                    if values.get(key) is None:
                        values[key] = default
                        # Defaulted, not judged. Without this the sort would
                        # rank every unscored cell as a perfect zero.
                        measured[key] = False
                cells[suite] = {
                    "error": s_entry.get("error"),
                    "values": values,
                    "sort": sorts,
                    "measured": measured,
                    **({"second": second} if second else {}),
                    **({"second_judge": s_entry["second_judge"]}
                       if s_entry.get("second_judge") else {}),
                }
            rows.append({"id": sample.id, "title": sample.title,
                         "target": target, "cells": cells,
                         # Carried per row, so one table can hold cells of
                         # different styles without the reader guessing.
                         "style": row_style,
                         # Only the first target of a sample prints its id, so
                         # the eye groups the pair instead of reading two
                         # unrelated rows that happen to share a name.
                         "first_of_sample": target == targets[0]})

    return jsonify({"style": style, "label": label, "groups": groups,
                    "rows": rows, "targets": targets,
                    "unified": bool(cells_spec)})


def _explain_media(label: str, target: str, entry: dict, sample_id: str,
                   style: str, ref_videos: dict) -> dict:
    """Render / animation / frame count for one target variant."""
    vcfg = entry["cfg"]
    vcfg.style = style
    vcfg.raw["animation_style"] = style
    paths = CachePaths.from_config(vcfg)
    exports = paths.exports(sample_id)
    frames_dir = exports / "frames"
    ref_key = f"{target}|{style}|full"
    return {
        "render": (f"/api/render/{quote(label)}/{quote(sample_id)}"
                   f"/{quote(style)}?target={quote(target)}"),
        "pipeline_video": (
            f"/api/video/{quote(label)}/{quote(sample_id)}/{quote(style)}"
            f"?target={quote(target)}"
            if (exports / "animation.mp4").is_file() else None),
        "reference_video": (
            f"/api/reference/{quote(sample_id)}/{quote(ref_videos[ref_key])}"
            if ref_key in ref_videos else None),
        "frames": len(list(frames_dir.glob("*.png"))) if frames_dir.is_dir() else 0,
    }


@app.get("/api/explain/<label>/<sample_id>/<style>")
def api_explain(label, sample_id, style):
    """One sample, one style: what it looked like, what it scored, and why.

    Every number is joined to the formula that produced it, instantiated on
    this record's own terms, and to the evidence the metric stored alongside
    it. Nothing here is recomputed -- this reads what the eval wrote, so the
    view cannot disagree with the score.
    """
    from img_2_svg_pretraining.animatebench import descriptions, results

    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404, f"unknown sample '{sample_id}'")

    cfg = STATE["models"][label]["cfg"]
    config_name = Path(STATE["models"][label]["file"]).stem
    paths = _paths_for(label, style)
    root = _evals_root(cfg)

    # Every target this label carries (tikz, svg, ...). The primary target
    # owns the prose -- formula, caveats, evidence -- and the others
    # contribute their number under `by_target`, so one row of the table can
    # be read across representations instead of across two models.
    variants = STATE["models"][label].get("variants") or {
        cfg.target: {"cfg": cfg, "file": STATE["models"][label]["file"],
                     "target": cfg.target}}
    primary_target = cfg.target
    targets = ([primary_target]
               + [tg for tg in sorted(variants) if tg != primary_target])

    def records_for(entry) -> dict:
        """suite -> record, for one target variant."""
        vcfg = entry["cfg"]
        vcfg.style = style
        vcfg.raw["animation_style"] = style
        vroot = _evals_root(vcfg)
        vvroot = _video_evals_root(vcfg)
        vqroot = _qwen_evals_root(vcfg)
        vname = Path(entry["file"]).stem
        out = {}
        for suite, _ in _table_suites():
            rec = results.read_record(
                results.suite_path(vroot, vname, style, sample_id, suite))
            # `vfs_video` is written to its own evals root (see
            # `_video_evals_root`), so the animation record this view shows is
            # the union of the two. Merged per target, which is what lets the
            # by_target comparison carry the video score as well.
            if suite == "animation" and vvroot is not None:
                extra = results.read_record(
                    results.suite_path(vvroot, vname, style, sample_id, suite))
                if extra is not None:
                    rec = dict(rec or {})
                    for k, v in extra.items():
                        if k.startswith("vfs_video"):
                            rec[k] = v
            if rec is not None:
                out[suite] = rec
        return out

    def second_records_for(entry) -> dict:
        """suite -> the second judge's record, for one target variant.

        Kept in a PARALLEL map rather than merged into `records_for`. Qwen
        re-scored the same stage1/xml metrics Gemini already scored, so a merge
        would overwrite published numbers with differently-derived ones while
        the provenance line still named the wrong model.
        """
        vcfg = entry["cfg"]
        vcfg.style = style
        vcfg.raw["animation_style"] = style
        vqroot = _qwen_evals_root(vcfg)
        if vqroot is None:
            return {}
        vname = Path(entry["file"]).stem
        out = {}
        for suite, _ in _table_suites():
            rec = results.read_record(
                results.suite_path(vqroot, vname, style, sample_id, suite))
            if rec is not None:
                out[suite] = rec
        return out

    by_target = {tg: records_for(variants[tg]) for tg in targets}
    second_by_target = {tg: second_records_for(variants[tg]) for tg in targets}

    suites, provenance = [], None
    for suite, suite_label in _table_suites():
        record = by_target.get(primary_target, {}).get(suite)
        # A suite the primary target never scored can still exist on another
        # target; fall back so an SVG-only result is not invisible.
        source = record or next(
            (by_target[tg][suite] for tg in targets if suite in by_target[tg]), None)
        if source is None:
            continue
        record = record if record is not None else source
        provenance = provenance or record.get("provenance")
        metrics = []
        for key in _suite_columns(suite):
            values = {tg: by_target[tg].get(suite, {}).get(key)
                      for tg in targets}
            if all(v is None for v in values.values()):
                continue
            second = {tg: second_by_target[tg].get(suite, {}).get(key)
                      for tg in targets}
            metrics.append({
                "key": key, "value": record.get(key),
                "by_target": values,
                # Present only where a second judge scored this metric, so a
                # cell with one judge renders exactly as it did before.
                **({"second_by_target": second}
                   if any(v is not None for v in second.values()) else {}),
                **descriptions.describe(key),
                "instantiated": descriptions.instantiate(key, record),
                "evidence": _evidence_for(key, record),
            })
        if metrics or record.get("error"):
            second_rec = next(
                (second_by_target[tg][suite] for tg in targets
                 if suite in second_by_target.get(tg, {})), None)
            suites.append({
                "suite": suite, "label": suite_label, "metrics": metrics,
                "error": record.get("error"),
                "skipped": record.get("coverage_skipped"),
                **({"second_judge": (second_rec.get("provenance") or {}).get(
                    "judge_model")} if second_rec else {}),
                **descriptions.suite_note(suite),
            })

    alignment = results.read_record(
        results.alignment_path(root, config_name, sample_id)) or {}

    exports = paths.exports(sample_id)
    frames_dir = exports / "frames"
    reference = _reference(sample_id)
    ref_videos = reference.get("videos", {})
    # Keyed on this model's own target: a model run with `target: svg` has
    # nothing in common with the tikz reference, and pairing it with one
    # anyway is a silent wrong-comparison rather than a missing one.
    ref_key = f"{cfg.target}|{style}|full"

    return jsonify({
        "id": sample_id, "title": sample.title, "style": style,
        "label": label, "model": STATE["models"][label]["model"],
        "targets": targets, "primary_target": primary_target,
        # Media per target, so the Metrics sub-tabs switch the render, the
        # animation and the frame deck together with the numbers -- reading
        # SVG scores beside a TikZ render would be worse than showing neither.
        "media_by_target": {
            tg: _explain_media(label, tg, variants[tg], sample_id, style,
                               ref_videos)
            for tg in targets},
        "scored": bool(suites),
        "figure": f"/api/figure/{quote(sample_id)}",
        "render": f"/api/render/{quote(label)}/{quote(sample_id)}/{quote(style)}",
        "reference_video": (
            f"/api/reference/{quote(sample_id)}/{quote(ref_videos[ref_key])}"
            if ref_key in ref_videos else None),
        "pipeline_video": (
            f"/api/video/{quote(label)}/{quote(sample_id)}/{quote(style)}"
            f"?target={quote(primary_target)}"
            if (exports / "animation.mp4").is_file() else None),
        "frames": len(list(frames_dir.glob("*.png"))) if frames_dir.is_dir() else 0,
        "suites": suites,
        "alignment": {
            "groups": alignment.get("groups", {}),
            "gt_unmatched": alignment.get("gt_unmatched", []),
            "pred_unmatched": alignment.get("pred_unmatched", []),
            "notes": alignment.get("notes", []),
        },
        "provenance": provenance,
    })


# Which record key holds the per-call trace for each node of the animation
# tree, and how that node's prompt is assembled. The prompt is rendered here
# through the metric module's own `_adapter` + `load_and_render`, never a copy
# pasted from the YAML: a reviewer tuning a prompt has to be looking at the
# text the judge actually received, or the tuning is guesswork.
_ANIM_NODES = [
    ("vfs", "Elimination 1 · Visual Fidelity", "animation_fidelity.yaml",
     "frames", "adapter_frames", "vfs_frames"),
    ("vfs_video", "Elimination 1b · Visual Fidelity (video)",
     "animation_fidelity.yaml", "video", "adapter_video", None),
    ("vfs_band", "Elimination 1 · Visual Fidelity (video, banded)",
     "animation_fidelity_bands.yaml", None, None, None),
    ("ascs", "Elimination 2 · Style Compliance", "animation_style.yaml",
     "frames", "adapter", "ascs_frame_detail"),
    ("ascs_video", "Elimination 2 · Style Compliance (video)",
     "animation_style_video.yaml", None, None, None),
    ("omission", "Elimination 3 · Omitted Elements", "animation_omission.yaml",
     "frames", "adapter", "omission_frame_detail"),
    ("sss", "Contributor 1 · Selection Sensibility",
     "animation_selection_bands.yaml", "user", "adapter", "sss_step_detail"),
    ("gps", "Contributor 2 · Granularity & Pacing",
     "animation_pacing_bands.yaml", "user", "adapter", "gps_step_detail"),
    ("nas", "Contributor 3 · Narration Alignment",
     "animation_narration.yaml", "user", "adapter", "nas_step_detail"),
    ("repetition", "Contributor 4 · Unnecessary Repetition",
     "animation_repetition.yaml", "frames", "adapter", "repetition_frame_detail"),
]

# Tree node key -> the scoreboard metric key it corresponds to, where the two
# names differ. Used to apply `--metrics` to the tree as well: without it the
# tree keeps rendering nodes for metrics the study never ran, which show
# identically empty for every model and read as "switching model changed
# nothing".
_NODE_METRIC_KEY = {"ascs": "ascs_pass", "omission": "omission_rate",
                    "repetition": "repetition_rate"}


def _anim_nodes():
    """The tree nodes this viewer shows, honouring `--metrics`."""
    only = STATE.get("metrics")
    if not only:
        return _ANIM_NODES
    return [n for n in _ANIM_NODES
            if _NODE_METRIC_KEY.get(n[0], n[0]) in only]


def _whole_video_prompt(prompt_file: str, style: str) -> dict:
    """The prompt for a node whose YAML holds one complete block per style."""
    from img_2_svg_pretraining.animatebench.metrics.animation_quality import (
        PROMPTS_ROOT)
    from img_2_svg_pretraining.pipeline.prompts import load_and_render
    try:
        return {"file": f"{prompt_file}#{style}",
                "text": load_and_render(f"{prompt_file}#{style}", {},
                                        root=PROMPTS_ROOT)}
    except Exception as exc:                          # noqa: BLE001
        return {"file": prompt_file, "error": str(exc)}


def _anim_prompt(prompt_file: str, body_key: str, adapter_prefix: str,
                 style: str) -> dict:
    """Render one node's prompt exactly as the metric would send it."""
    from img_2_svg_pretraining.animatebench.judge import PROMPTS_ROOT
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq
    from img_2_svg_pretraining.pipeline.prompts import load_and_render

    try:
        values = {"style_adapter": aq._adapter(prompt_file, style, adapter_prefix)}
        for extra, key in (("output_schema", "schema"),):
            try:
                values[extra] = aq._adapter(prompt_file, style, key)
            except Exception:
                pass
        text = load_and_render(f"{prompt_file}#{body_key}", values,
                               root=PROMPTS_ROOT)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "file": prompt_file}
    return {"file": prompt_file, "key": body_key, "text": text,
            "adapter": values.get("style_adapter", ""),
            "chars": len(text)}


def _checklist_prompt() -> dict:
    """The shared first call. It takes no style adapter -- one prompt builds the
    checklist that both omission and repetition consume."""
    from img_2_svg_pretraining.animatebench.judge import PROMPTS_ROOT
    from img_2_svg_pretraining.pipeline.prompts import load_and_render
    try:
        text = load_and_render("animation_checklist.yaml#prompt",
                               {"diagram_code": "<diagram code appended at call time>",
                                "sequence_view": "<bucketed sequence appended at call time>"},
                               root=PROMPTS_ROOT)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"file": "animation_checklist.yaml", "key": "prompt",
            "text": text, "chars": len(text)}


@app.get("/api/animation/<label>/<sample_id>/<style>")
def api_animation(label, sample_id, style):
    """The animation tree for one cell: prompts, call counts, per-call trace.

    Exists so the judged half can be audited the way the programmatic half
    already can. A score from a model is only reviewable if you can see the
    instruction it was given and the answer it gave for each frame.

    TARGET-AWARE. `?target=svg` resolves the SVG variant's own config, evals
    root and config name, so the tree shown is the one that was actually run
    for that representation. Without it this endpoint always answered for the
    primary target while the sub-tab above it said "svg" -- every per-frame
    verdict, prompt and call count in the block belonged to the TikZ run, and
    nothing on screen said so. The metrics *rows* switched (they carry
    `by_target`), which made the mismatch harder to spot, not easier.
    """
    from img_2_svg_pretraining.animatebench import results

    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    entry = STATE["models"][label]
    target = request.args.get("target")
    variants = entry.get("variants") or {}
    variant = variants.get(target) if target else None
    if variant is not None:
        cfg = variant["cfg"]
        cfg.style = style
        cfg.raw["animation_style"] = style
        config_name = Path(variant["file"]).stem
    else:
        cfg = entry["cfg"]
        config_name = Path(entry["file"]).stem
    _paths_for_target(label, style, target)
    root = _evals_root(cfg)
    record = results.read_record(results.suite_path(
        root, config_name, style, sample_id, "animation"))
    video_root = _video_evals_root(cfg)
    if video_root is not None:
        extra = results.read_record(results.suite_path(
            video_root, config_name, style, sample_id, "animation"))
        if extra is not None:
            record = dict(record or {})
            for k, v in extra.items():
                if k.startswith("vfs_video"):
                    record[k] = v
            # `stages_run` decides whether a node renders as "ran" rather
            # than "not scored", and it lives in the video root's own record.
            if "vfs_video" in (extra.get("stages_run") or []):
                record["stages_run"] = list(
                    dict.fromkeys((record.get("stages_run") or [])
                                  + ["vfs_video"]))
    if record is None:
        return jsonify({"scored": False, "nodes": []})

    nodes = []
    for key, title, prompt_file, body_key, adapter_prefix, detail_key in _anim_nodes():
        detail = record.get(detail_key) or []
        errors = record.get(f"{key}_errors") or []
        manifest = {k: record.get(f"{key}_{k}") for k in
                    ("frame_policy", "frames_available", "frames_judged",
                     "frame_long_edge")
                    if record.get(f"{key}_{k}") is not None}
        # The video node has no per-frame trace -- one call over the whole
        # animation. Its evidence is the prose it wrote, so synthesise a
        # single detail row rather than showing an empty node.
        if key == "vfs_band" and record.get("vfs_band") is not None:
            rationale = record.get("vfs_band_rationale")
            detail = [{
                "frame": "whole animation",
                "band": record.get("vfs_band"),
                "pass": record.get("vfs_band_pass"),
                "deck": record.get("vfs_band_source"),
                "frames_in_video": record.get("vfs_band_frame_count"),
                "summary": record.get("vfs_band_summary"),
                **(rationale if isinstance(rationale, dict) else {}),
            }]
        if key == "ascs_video" and record.get("ascs_video") is not None:
            detail = [{
                "frame": "whole animation",
                "verdict": record.get("ascs_video_verdict"),
                "pass": record.get("ascs_video_pass"),
                "deck": record.get("ascs_video_source"),
                "frames_in_video": record.get("ascs_video_frame_count"),
                "rationale": record.get("ascs_video_rationale"),
            }]
        if key == "vfs_video" and record.get("vfs_video") is not None:
            assessment = record.get("vfs_video_assessment")
            detail = [{
                "frame": "whole animation",
                "score": record.get("vfs_video_raw"),
                "temporal_defects_observed":
                    record.get("vfs_video_temporal_defects"),
                **(assessment if isinstance(assessment, dict) else {}),
            }]

        nodes.append({
            "key": key, "title": title,
            "ran": key in (record.get("stages_run") or []),
            "skipped": (record.get("stages_skipped") or {}).get(key),
            "status": record.get("repetition_status") if key == "repetition" else None,
            "note": record.get("repetition_note") if key == "repetition" else None,
            "calls": len(detail), "errors": errors,
            "manifest": manifest,
            "prompt": ({"unspecified": True}
                       if key == "repetition"
                       and record.get("repetition_status") == "not_specified"
                       else _anim_prompt(prompt_file, body_key, adapter_prefix, style)
                       if body_key is not None
                       # The banded video prompts are one self-contained block
                       # per style, with no body/adapter split to reassemble.
                       else _whole_video_prompt(prompt_file, style)),
            "trace": detail,
        })

    checklist = None
    cl_path = root / "checklist" / config_name / style / f"{sample_id}.json"
    if cl_path.exists():
        try:
            checklist = json.loads(cl_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checklist = None

    total = sum(n["calls"] for n in nodes) + (1 if checklist else 0)
    return jsonify({
        "scored": True, "style": style, "sample": sample_id,
        "total_calls": total,
        "would_eliminate": record.get("would_eliminate") or [],
        "unenforced_rules": record.get("ascs_unenforced_rules") or [],
        "checklist": checklist,
        "checklist_prompt": _checklist_prompt() if checklist else None,
        "nodes": nodes,
    })


@app.get("/api/render/<label>/<sample_id>/<style>")
def api_render(label, sample_id, style):
    """The compiled Stage-1 diagram, as rendering fidelity saw it.

    `stage1.json` records the render it scored, but as a container path that
    does not exist outside the container. Try it, then the same path with the
    `/code` prefix stripped, and only then recompile -- so the image shown is
    the one that was actually judged whenever it is still on disk.
    """
    from img_2_svg_pretraining.animatebench import results

    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    target = request.args.get("target")
    variants = STATE["models"][label].get("variants") or {}
    entry = variants.get(target) or STATE["models"][label]
    cfg = entry["cfg"]
    config_name = Path(entry["file"]).stem
    paths = _paths_for_target(label, style, target)
    root = _evals_root(cfg)

    record = results.read_record(
        results.suite_path(root, config_name, style, sample_id, "stage1")) or {}
    stored = record.get("render_path")
    for candidate in _local_candidates(stored, Path(cfg.cache_root)):
        if candidate.is_file():
            return send_file(candidate, mimetype="image/png")

    try:
        code_path = paths.resolve_code(sample_id)
    except FileNotFoundError:
        abort(404, "no diagram code to render")
    source = code_path.read_text(encoding="utf-8")
    if cfg.target == "svg":
        # Same CompileResult contract as compile_tikz by design (see
        # svg_render.py's docstring), so the dispatch is the only thing that
        # needs to know the target exists.
        from img_2_svg_pretraining.pipeline.svg_render import render_svg
        result = render_svg(source, paths.compile_cache())
    else:
        from img_2_svg_pretraining.viewer.compile import compile_tikz
        result = compile_tikz(source, paths.compile_cache())
    if not result.ok or not result.png_path:
        abort(404, result.log or "diagram does not compile")
    return send_file(result.png_path, mimetype="image/png")


@app.get("/api/frame/<label>/<sample_id>/<style>/<int:index>")
def api_frame(label, sample_id, style, index):
    """One exported frame, by playback index.

    Ordered by `animatebench.frames.list_frames`, which is the same numeric
    sort the judges walk -- so index N here is the frame the per-frame
    metrics recorded as N. A plain `sorted(glob(...))` would agree only while
    frame numbers stay zero-padded to the same width: frame-100.png sorts
    between frame-09 and frame-10, silently pairing every verdict with the
    wrong picture.
    """
    from img_2_svg_pretraining.animatebench.frames import list_frames

    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    frames = list_frames(
        _paths_for_target(label, style, request.args.get("target"))
        .exports(sample_id) / "frames")
    if not 0 <= index < len(frames):
        abort(404)
    return send_file(frames[index], mimetype="image/png")


@app.get("/api/index")
def api_index():
    return jsonify({
        "models": [
            {"label": label, "model": m["model"], "config": m["file"]}
            for label, m in STATE["models"].items()
        ],
        "styles": STATE["styles"],
        # Present only in a study view. The UI reads it as "this sample has
        # exactly one style", which is what pins the style selector.
        "cells": STATE.get("cells"),
        "samples": [
            {"id": s.id, "title": s.title,
             "has_reference": bool(_reference(s.id).get("videos"))}
            for s in STATE["samples"]
        ],
    })


@app.get("/api/compare/<sample_id>/<style>")
def api_compare(sample_id, style):
    if style not in STYLES:
        abort(404, f"unknown style '{style}'")
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404, f"unknown sample '{sample_id}'")

    reference = _reference(sample_id)
    ref_videos = reference.get("videos", {})

    # One panel per (label, TARGET). A label carrying both a tikz and an svg
    # config produces two panels, because they are two different animations
    # of the same figure -- showing only the primary would hide an SVG run
    # that exists on disk and is fully scored.
    panels = []
    for label in STATE["models"]:
        variants = STATE["models"][label].get("variants") or {
            STATE["models"][label]["cfg"].target: STATE["models"][label]}
        for target in sorted(variants, key=lambda tg: (tg != "tikz", tg)):
            entry = variants[target]
            vcfg = entry["cfg"]
            vcfg.style = style
            vcfg.raw["animation_style"] = style
            paths = CachePaths.from_config(vcfg)
            exports = paths.exports(sample_id)
            mp4 = exports / "animation.mp4"
            frames_dir = exports / "frames"
            panels.append({
                "label": label if len(variants) == 1 else f"{label} · {target}",
                # The bare label, for endpoints keyed by it -- the display
                # label above may carry a " · svg" suffix that no route knows.
                "model_label": label,
                "model": entry["model"],
                "target": target,
                # Percent-encoded: labels carry spaces and parentheses
                # ("Gemini 3.7 Flash (OpenRouter)"), and a raw space in a
                # <video src> is not a URL the browser will fetch -- it fails
                # as "no video with supported format and MIME type found",
                # which reads like a codec problem and is not one.
                "video": (f"/api/video/{quote(label)}/{quote(sample_id)}"
                          f"/{quote(style)}?target={quote(target)}"
                          if mp4.exists() else None),
                "frames": len(list(frames_dir.glob("*.png"))) if frames_dir.is_dir() else 0,
                "sequence": _sequence_summary(paths.sequence_narrated(sample_id))
                            or _sequence_summary(paths.sequence(sample_id)),
                "code": paths.animation(sample_id).exists(),
                "dir": str(exports),
                "metrics": _metrics(vcfg, entry["file"], style, sample_id),
            })

    # The bundle ships one reference video per (target, style, tier), and the
    # models being compared are not necessarily all the same target -- a tikz
    # baseline and an svg baseline can sit in the same grid. One reference per
    # target actually in play, so each panel is paired against the ground
    # truth for what it actually produced rather than always against tikz.
    targets_shown = sorted({p["target"] for p in panels}) or ["tikz"]
    references = {}
    for target in targets_shown:
        ref_key = f"{target}|{style}|full"
        references[target] = {
            "video": (
                f"/api/reference/{quote(sample_id)}/{quote(ref_videos[ref_key])}"
                if ref_key in ref_videos else None),
            "target": target,
        }
    talk = reference.get("original_presentation")
    talk_dir = Path(sample.directory) / "reference" / "original_presentation"
    return jsonify({
        "id": sample_id,
        "title": sample.title,
        "style": style,
        "figure": f"/api/figure/{quote(sample_id)}",
        "reference": {
            # Back-compat single field: the first (alphabetically, so "svg"
            # before "tikz") target actually shown, for callers that have not
            # moved to `references` yet.
            "video": references[targets_shown[0]]["video"],
            "available_keys": sorted(ref_videos),
            "reviews": reference.get("reviews", []),
        },
        "references": references,
        "presentation": ({
            "title": talk.get("title"),
            "uploader": talk.get("uploader"),
            "url": talk.get("webpage_url"),
            "duration": talk.get("duration"),
            "video": (f"/api/presentation/{sample_id}/video"
                      if (talk_dir / "video.mp4").is_file() else None),
            "transcript": ((talk_dir / "transcript.txt").read_text(encoding="utf-8")
                           if (talk_dir / "transcript.txt").is_file() else None),
        } if talk else None),
        "panels": panels,
    })



@app.get("/api/leaderboard/<metric>")
def api_leaderboard(metric):
    """One row per (model, cell) for ONE metric, ranked best-first.

    The scoreboard is one table per model, so sorting inside it can never put
    model A's worst cell next to model B's -- which is the actual question this
    study asks. This endpoint flattens every model over every pinned cell so the
    best and worst rows ARE the first and last rows.

    Ordering uses the normalised `sort` key from `_sort_key`, never the raw
    value, so a banded metric cannot be ranked upside-down (see that function).
    Rows with no sort key are unrankable and are returned last, flagged, rather
    than being silently treated as zero.
    """
    cells_spec = STATE.get("cells")
    if cells_spec:
        pairs = [(STATE["by_id"][c["id"]], c["style"]) for c in cells_spec
                 if c["id"] in STATE["by_id"]]
    else:
        pairs = [(s, STATE["styles"][0]) for s in STATE["samples"]]

    rows = []
    for label, model in STATE["models"].items():
        variants = model.get("variants") or {model["cfg"].target: model}
        for target, entry in variants.items():
            for sample, style in pairs:
                m = _metrics(entry["cfg"], entry["file"], style, sample.id)
                found = None
                for suite in m["suites"]:
                    for e in suite["metrics"]:
                        if e["key"] == metric:
                            found = e
                            break
                if found is None:
                    continue
                rows.append({
                    "model": label, "target": target, "sample": sample.id,
                    "style": style, "value": found["value"],
                    "sort": found.get("sort"),
                    "measured": found.get("measured", True),
                    "better": found.get("better"),
                })

    # Unrankable rows last, regardless of direction, so they never occupy the
    # "best" end of the table by accident.
    rows.sort(key=lambda r: (r["sort"] is None, -(r["sort"] or 0.0)))
    return jsonify({
        "metric": metric, "rows": rows,
        "label": descriptions.describe(metric).get("label", metric),
        "what": descriptions.describe(metric).get("what"),
        "unrankable": sum(1 for r in rows if r["sort"] is None),
    })



@app.get("/api/overview")
def api_overview():
    """Every model x every cell in ONE table -- the all-models scoreboard.

    `api_table` is one table per model, so comparing models means flipping the
    selector and holding numbers in your head. This returns the whole matrix at
    once: one row per (model, sample), every metric as a column, so a sort by
    any metric ranks across ALL models together and the best and worst rows are
    the extremes of the entire study rather than of one arm.

    Each metric ships both its display `value` and a normalised `sort` key from
    `_sort_key` -- always higher-is-better, so the client never has to know that
    `vfs_band_ordinal` counts the other way (0 = BAND A = best) while
    `descriptions.py` declares vfs_band better="higher". Sorting on the raw
    value there would rank BAND D first and look entirely normal.
    """
    cells_spec = STATE.get("cells")
    if cells_spec:
        pairs = [(STATE["by_id"][c["id"]], c["style"]) for c in cells_spec
                 if c["id"] in STATE["by_id"]]
    else:
        pairs = [(s, STATE["styles"][0]) for s in STATE["samples"]]

    keys, seen = [], set()
    for suite, _ in _table_suites():
        for k in _suite_columns(suite):
            if k not in seen:
                seen.add(k); keys.append(k)
    columns = [{"key": k, **descriptions.describe(k)} for k in keys]

    rows = []
    for label, model in STATE["models"].items():
        variants = model.get("variants") or {model["cfg"].target: model}
        for target, entry in variants.items():
            for sample, style in pairs:
                m = _metrics(entry["cfg"], entry["file"], style, sample.id)
                flat = {}
                for suite in m["suites"]:
                    for e in suite["metrics"]:
                        flat[e["key"]] = e
                if not flat:
                    continue        # nothing scored for this cell yet
                rows.append({
                    "model": label, "sample": sample.id, "style": style,
                    "target": target,
                    "values": {k: flat[k]["value"] for k in flat},
                    "sort": {k: flat[k].get("sort") for k in flat
                             if flat[k].get("sort") is not None},
                    "measured": {k: True for k in flat},
                })
    return jsonify({"columns": columns, "rows": rows,
                    "models": list(STATE["models"]), "cells": len(pairs)})


@app.get("/steps.csv")
def steps_csv():
    """Framewise judge scores as one long-format CSV, for external analysis.

    The cell-level sss/gps/nas numbers are means over per-timestep bands, and
    a mean hides exactly what someone choosing an aggregation needs to see:
    the distribution. One row per (model, sample, metric, timestep), carrying
    the final band AND every sub-band the judge scored on the way there, so a
    spreadsheet can test alternative aggregations (min, trimmed mean, per-
    criterion) directly against the shipped mean.

    Long format rather than one-column-per-metric because the three metrics
    have disjoint sub-bands; metric-specific columns are simply empty on the
    other metrics' rows. Rationales are deliberately omitted -- they are
    prose, and this file is for arithmetic; the viewer shows them per cell.

    Reads records through the same `results.suite_path` resolution as
    `_metrics`, so the CSV can never disagree with what the viewer shows.
    """
    import csv
    import io

    from img_2_svg_pretraining.animatebench import results

    detail_fields = {
        "sss": ["criterion_a_band", "criterion_b_band"],
        "gps": ["volume_band", "complexity_band", "relevance_band"],
        "nas": ["alignment_band", "insight_band", "coherence_band"],
    }
    sub_cols = [f for fields in detail_fields.values() for f in fields]
    header = (["model", "config", "sample", "style", "metric", "cell_score",
               "cell_band_mean", "timestep", "frame", "is_valid",
               "final_band", "band_label"] + sub_cols)

    cells_spec = STATE.get("cells")
    if cells_spec:
        pairs = [(STATE["by_id"][c["id"]], c["style"]) for c in cells_spec
                 if c["id"] in STATE["by_id"]]
    else:
        pairs = [(s, sty) for s in STATE["samples"] for sty in STATE["styles"]]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for label, model in STATE["models"].items():
        config_name = Path(model["file"]).stem
        root = _evals_root(model["cfg"])
        for sample, style in pairs:
            record = results.read_record(results.suite_path(
                root, config_name, style, sample.id, "animation"))
            if not record:
                continue
            for metric, fields in detail_fields.items():
                for step in (record.get(f"{metric}_step_detail") or []):
                    writer.writerow(
                        [label, config_name, sample.id, style, metric,
                         record.get(metric),
                         record.get(f"{metric}_band_mean"),
                         step.get("timestep"), step.get("frame"),
                         step.get("is_valid"), step.get("final_band"),
                         step.get("label")]
                        + [step.get(f) if f in fields else "" for f in sub_cols])
    return app.response_class(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=animatebench_v5_steps.csv"})


@app.get("/api/figure/<sample_id>")
def api_figure(sample_id):
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404)
    return send_file(sample.image_path)


# Reference videos the bundle ships in a codec no browser plays, transcoded
# once and cached here. Keyed by source path + mtime so a re-imported bundle
# does not serve a stale transcode.
_TRANSCODE_CACHE: dict[str, Path] = {}


def _browser_playable(path: Path) -> bool:
    """Is this mp4 something a <video> element can actually decode?

    Half the v3 reference videos are MPEG-4 Part 2 (`mp4v`, Simple Profile),
    which no browser supports -- every SVG-target reference is, while every
    TikZ one is h264. The failure surfaces as
    `DEMUXER_ERROR_NO_SUPPORTED_STREAMS`, which reads like a broken file and
    is really an unsupported codec.
    """
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return True          # cannot check; serve it and let the client decide
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                             capture_output=True, text=True, timeout=30)
    except Exception:
        return True
    return "Video: h264" in (out.stdout + out.stderr)


def _playable_copy(path: Path) -> Path:
    """`path` if the browser can play it, else a cached h264 transcode."""
    key = f"{path}:{path.stat().st_mtime_ns}"
    cached = _TRANSCODE_CACHE.get(key)
    if cached and cached.is_file():
        return cached
    if _browser_playable(path):
        _TRANSCODE_CACHE[key] = path
        return path

    import hashlib
    import subprocess
    import imageio_ffmpeg
    out_dir = Path(tempfile.gettempdir()) / "animatebench_transcodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.mp4"
    if not out.is_file():
        # `-movflags +faststart` puts the moov atom first so playback can
        # begin before the whole file has arrived.
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(out)],
            capture_output=True, timeout=300)
    if out.is_file():
        _TRANSCODE_CACHE[key] = out
        return out
    return path              # transcode failed; serve the original


@app.get("/api/reference/<sample_id>/<name>")
def api_reference(sample_id, name):
    sample = STATE["by_id"].get(sample_id)
    if sample is None or ".." in name or "/" in name:
        abort(404)
    path = Path(sample.directory) / "reference" / "videos" / name
    if not path.is_file():
        abort(404)
    return send_file(_playable_copy(path), mimetype="video/mp4")


@app.get("/api/presentation/<sample_id>/video")
def api_presentation_video(sample_id):
    """The author's real conference talk, shipped as reference/original_presentation."""
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404)
    path = Path(sample.directory) / "reference" / "original_presentation" / "video.mp4"
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="video/mp4")


@app.get("/api/video/<label>/<sample_id>/<style>")
def api_video(label, sample_id, style):
    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    path = (_paths_for_target(label, style, request.args.get("target"))
            .exports(sample_id) / "animation.mp4")
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="video/mp4")


@app.get("/api/sequence/<label>/<sample_id>/<style>")
def api_sequence(label, sample_id, style):
    """Full narrated sequence, in the benchmark's own dialect for comparison.

    `?target=` picks the variant, so a tikz panel and an svg panel show their
    own sequences rather than both showing the primary's.
    """
    if label not in STATE["models"] or style not in STYLES:
        abort(404)
    paths = _paths_for_target(label, style, request.args.get("target"))
    path = paths.sequence_narrated(sample_id)
    if not path.exists():
        path = paths.sequence(sample_id)
    if not path.exists():
        abort(404, "no sequence")
    seq = AnimationSequence.load(path)
    return jsonify({"native": seq.to_dict(), "bench": seq.to_bench_dict()})


@app.get("/api/reference-sequence/<sample_id>/<style>")
def api_reference_sequence(sample_id, style):
    """The benchmark's own narrated sequence for this style.

    `?target=svg` selects the SVG-dialect reference; default `tikz` is the
    pre-existing behaviour, kept as the default rather than made mandatory so
    every caller from before targets were distinguished still resolves.
    """
    target = request.args.get("target", "tikz")
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404)
    directory = Path(sample.directory) / "reference" / "narration"
    # The two bundle generations name these differently. v2 ships
    # `<style>_<id>_<target>.json` (the convention gt.py already uses); v1
    # ships `<target>_<style>_img_and_context_<id>.json`. Try v2 first, then
    # fall back, so one viewer serves both datasets. The v1 fallback has no
    # SVG bundles, so it stays tikz-only by construction.
    matches = sorted(directory.glob(f"{style}_{sample_id}_{target}.json")) \
        or (sorted(directory.glob(f"tikz_{style}_img_and_context_*.json"))
            if target == "tikz" else [])
    if not matches:
        abort(404, "no reference narration")
    try:
        return jsonify(json.loads(matches[0].read_text()))
    except json.JSONDecodeError as e:
        # One file in the shipped bundle is malformed (an element missing its
        # "id" key). Report it rather than 500ing.
        return jsonify({"error": f"reference file is not valid JSON: {e}"}), 200


@app.get("/")
def index():
    return (Path(__file__).parent / "compare.html").read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--dataset", default=None,
                        help="override the dataset root from the configs")
    parser.add_argument("--configs", default=None,
                        help="comma-separated label=config_file.yaml pairs, "
                             "replacing DEFAULT_CONFIGS (e.g. for a different bench)")
    parser.add_argument("--extra-roots", default=None,
                        help="comma-separated additional dataset roots to "
                             "discover samples from, for a study spanning "
                             "more than one bench version")
    parser.add_argument("--suites", default=None,
                        help="comma-separated suite ids to show in the "
                             "scoreboard and metrics views (e.g. 'animation'). "
                             "Default: every suite.")
    parser.add_argument("--metrics", default=None,
                        help="comma-separated metric keys to show. Default: "
                             "every metric each shown suite defines.")
    parser.add_argument("--cells", default=None,
                        help="comma-separated sample_id:style pairs. Restricts "
                             "the viewer to exactly those cells and pins each "
                             "sample to its own style -- for a study over a "
                             "chosen subset rather than the whole bench.")
    args = parser.parse_args()

    config_files = DEFAULT_CONFIGS
    if args.configs:
        config_files = {}
        for pair in args.configs.split(","):
            label, _, filename = pair.partition("=")
            if not filename:
                raise SystemExit(f"--configs: bad pair '{pair}', expected label=file.yaml")
            config_files[label.strip()] = filename.strip()

    cells = None
    if args.cells:
        cells = []
        for pair in args.cells.split(","):
            sample_id, _, style = pair.partition(":")
            if not style:
                raise SystemExit(
                    f"--cells: bad pair '{pair}', expected sample_id:style")
            if style.strip() not in STYLES:
                raise SystemExit(f"--cells: unknown style '{style.strip()}'")
            cells.append({"id": sample_id.strip(), "style": style.strip()})

    if args.suites:
        STATE["suites"] = {s.strip() for s in args.suites.split(",") if s.strip()}
    if args.metrics:
        STATE["metrics"] = {m.strip() for m in args.metrics.split(",") if m.strip()}

    extra = [r.strip() for r in (args.extra_roots or "").split(",") if r.strip()]
    _init(config_files, args.dataset, cells, extra)
    print(f"{len(STATE['models'])} model(s), {len(STATE['samples'])} sample(s)"
          + (f", {len(cells)} pinned cell(s)" if cells else ""))
    print(f"open http://<host>:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
