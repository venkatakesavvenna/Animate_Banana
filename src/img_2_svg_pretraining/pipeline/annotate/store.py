"""Per-sample annotation records: one JSON file per sample, covering every
Stage-1 screen.

Filelock-guarded read-modify-write, matching `annotation_tool/store.py`'s
pattern -- a single Flask process can still receive two rapid requests for
the same sample (e.g. a double-click), and across multiple annotators the
lock is cheap insurance even though disjoint sharding should make collisions
impossible by construction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from filelock import FileLock

SCHEMA_VERSION = 3

# Stages a reviewer signs off on. Narration (2e) is deliberately absent: it is
# written into an already-valid sequence and only touches the `narrative`
# field, so there is nothing to approve and nothing it can invalidate.
APPROVABLE = ("stage1a", "stage1b", "stage2b", "stage2c", "stage2d", "stage3a")


def _with_approval(record: dict) -> dict:
    """Give every approvable stage its sign-off fields.

    Kept out of `_empty` so the field set is defined once rather than repeated
    in six blocks that would drift apart.
    """
    for stage in APPROVABLE:
        block = record.get(stage)
        if isinstance(block, dict):
            block.setdefault("approved", False)
            block.setdefault("approved_at", None)
            block.setdefault("approved_by", None)
    return record


def annotations_dir(cache_root: Path) -> Path:
    d = cache_root / "annotations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_path(cache_root: Path, sample_id: str) -> Path:
    return annotations_dir(cache_root) / f"{sample_id}.json"


def _empty(sample_id: str) -> dict:
    return _with_approval({
        "sample_id": sample_id,
        "schema_version": SCHEMA_VERSION,
        # Which animation style this sample is annotated against. Recorded
        # rather than re-rolled, because a sample judged against a different
        # style each visit is not comparable with itself.
        "style": {
            "assigned": None,
            "source": None,          # "random" | "manual"
            "assigned_at": None,
            "history": [],
        },
        # tikz | svg. Unlike style there is nothing to randomise -- it defaults
        # to whatever the config targets and only changes when a reviewer
        # switches it. Recorded per sample so one instance can annotate both,
        # and because the target decides the artifact suffix and therefore
        # every path the human-edit swaps back up.
        "target": {
            "assigned": None,
            "source": None,          # "config" | "manual"
            "assigned_at": None,
            "history": [],
        },
        "stage1a": {
            "status": "unreviewed",
            "code_lineage_at_review": None,
            "comments": [],
            "human_code_path": None,
            "human_code_source": None,
            "human_code_updated_at": None,
            "promoted_to_code_final": False,
        },
        "stage1b": {
            "status": "unreviewed",
            "code_lineage_at_review": None,
            "boxes": [],
            "code_final_human_path": None,
            "promoted_to_code_final": False,
        },
        # Stage 2b -- the parsed structure XML.
        "stage2b": {
            "status": "unreviewed",
            "xml_lineage_at_review": None,
            "comments": [],
            "human_xml_path": None,
            "human_xml_source": None,     # "pasted" | "tree_edit"
            "human_xml_updated_at": None,
            # Against the machine XML at save time. Load-bearing: these ids are
            # what every downstream `focus` entry is validated against, so a
            # removal here breaks sequence steps elsewhere.
            "ids_added": [],
            "ids_removed": [],
            "promoted_to_xml": False,
        },
        # Stage 2c -- the sequencer's own output, before the critic sees it.
        # Distinct from 2d: this one swaps over `sequence()`, so the critic
        # still runs on the correction. 2d swaps over `sequence_final()` and
        # freezes it. A reviewer picks depending on whether they want the
        # repair refined or taken verbatim.
        "stage2c": {
            "status": "unreviewed",
            "sequence_lineage_at_review": None,
            "style_at_review": None,
            "comments": [],
            "human_sequence_path": None,
            "human_sequence_updated_at": None,
            "validation_at_save": [],
            "promoted_to_sequence": False,
        },
        # Stage 2d -- the animation sequence.
        "stage2d": {
            "status": "unreviewed",
            "sequence_lineage_at_review": None,
            # Not redundant with `style.assigned`: the record is per-sample but
            # the artifact is per-(sample, style), so this says which style the
            # saved sequence was actually written for.
            "style_at_review": None,
            "comments": [],
            "human_sequence_path": None,
            "human_sequence_updated_at": None,
            "validation_at_save": [],
            "promoted_to_sequence_final": False,
            "video_rendered_at": None,
        },
        # Stage 3a -- the animation code the designer wrote. The third and last
        # human-code override point, after 1a and 2c.
        "stage3a": {
            "status": "unreviewed",
            "animation_lineage_at_review": None,
            # Load-bearing for discard: the promoted file lives under a
            # style-keyed lineage, so removing it needs the style it was saved
            # under, not whatever the sample is set to now.
            "style_at_review": None,
            "comments": [],
            "human_animation_path": None,
            "human_animation_updated_at": None,
            # Whether the saved code compiled, recorded at save time. Saved
            # even when it does not: a work-in-progress paste is still worth
            # keeping, and the 3a gate is what refuses to advance on it.
            "compiles_at_save": None,
            "compile_log_at_save": None,
            "promoted_to_animation_final": False,
            "frames_rendered_at": None,
        },
    })


def _migrate(record: dict, sample_id: str) -> dict:
    """Fill in keys added since this record was written.

    Additive merge rather than a version check: Stage-1 records already exist
    on disk from live annotation runs, and resetting them to defaults would
    throw away real human work. This is what lets the 2c/3a blocks and the
    approval fields land on existing records without a migration script.
    """
    template = _empty(sample_id)
    for key, default in template.items():
        if key not in record:
            record[key] = default
        elif isinstance(default, dict) and isinstance(record[key], dict):
            for sub, sub_default in default.items():
                record[key].setdefault(sub, sub_default)
    record["schema_version"] = SCHEMA_VERSION
    return _with_approval(record)


def load(cache_root: Path, sample_id: str) -> dict:
    """The annotation record for one sample, or a fresh empty one."""
    path = record_path(cache_root, sample_id)
    if not path.exists():
        return _empty(sample_id)
    lock = FileLock(str(path) + ".lock")
    with lock:
        try:
            return _migrate(json.loads(path.read_text(encoding="utf-8")), sample_id)
        except (OSError, json.JSONDecodeError):
            return _empty(sample_id)


def _save(cache_root: Path, sample_id: str, record: dict) -> Path:
    path = record_path(cache_root, sample_id)
    lock = FileLock(str(path) + ".lock")
    with lock:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        tmp.replace(path)
    return path


def update(cache_root: Path, sample_id: str, mutate) -> dict:
    """Locked read-modify-write. `mutate(record)` edits the dict in place."""
    path = record_path(cache_root, sample_id)
    lock = FileLock(str(path) + ".lock")
    with lock:
        if path.exists():
            try:
                record = _migrate(json.loads(path.read_text(encoding="utf-8")), sample_id)
            except (OSError, json.JSONDecodeError):
                record = _empty(sample_id)
        else:
            record = _empty(sample_id)
        mutate(record)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        tmp.replace(path)
    return record


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
