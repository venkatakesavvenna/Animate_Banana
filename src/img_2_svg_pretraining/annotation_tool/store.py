"""One-JSON-per-image persistence with file locking (spec section 12).

Writes happen on every accept/reject/merge -- not batched. `filelock` guards
against two reviewer processes (separate Streamlit servers, spec's
one-process-per-reviewer model) writing the same image's file at once.

Default annotations directory is ``data/annotations`` under the repo root;
override with the ``ANNOTATIONS_DIR`` env var.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from filelock import FileLock

from .datamodel import ImageRecord, Instance, deserialize, serialize

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANNOTATIONS_DIR = _REPO_ROOT / "data" / "annotations"


def annotations_dir() -> Path:
    d = Path(os.environ.get("ANNOTATIONS_DIR", DEFAULT_ANNOTATIONS_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_path(image_id: str) -> Path:
    return annotations_dir() / f"{image_id}.json"


def save_image_record(record: ImageRecord, instances: dict[str, Instance]) -> Path:
    path = record_path(record.id)
    lock = FileLock(str(path) + ".lock")
    with lock:
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(serialize(record, instances), f)
        os.replace(tmp, path)  # atomic: readers never see a half-written file
    return path


def load_image_record(image_id: str) -> tuple[ImageRecord, dict[str, Instance]] | None:
    path = record_path(image_id)
    if not path.exists():
        return None
    lock = FileLock(str(path) + ".lock")
    with lock:
        with open(path) as f:
            data = json.load(f)
    return deserialize(data)


def list_annotated_image_ids() -> list[str]:
    return sorted(p.stem for p in annotations_dir().glob("*.json"))
