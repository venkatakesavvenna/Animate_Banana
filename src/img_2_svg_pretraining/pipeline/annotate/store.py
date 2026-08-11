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

SCHEMA_VERSION = 2


def annotations_dir(cache_root: Path) -> Path:
    d = cache_root / "annotations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_path(cache_root: Path, sample_id: str) -> Path:
    return annotations_dir(cache_root) / f"{sample_id}.json"


def _empty(sample_id: str) -> dict:
    return {
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
    }


def _migrate(record: dict, sample_id: str) -> dict:
    """Fill in keys added since this record was written.

    Additive merge rather than a version check: Stage-1 records already exist
    on disk from live annotation runs, and resetting them to defaults would
    throw away real human work.
    """
    template = _empty(sample_id)
    for key, default in template.items():
        if key not in record:
            record[key] = default
        elif isinstance(default, dict) and isinstance(record[key], dict):
            for sub, sub_default in default.items():
                record[key].setdefault(sub, sub_default)
    record["schema_version"] = SCHEMA_VERSION
    return record


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
