"""Data model for the raster-region annotation tool.

Plain dataclasses serialized with ``dataclasses.asdict()`` + ``json.dump()``,
one JSON file per image (see store.py). No OCR fields anywhere -- the
duplicate-text check from an earlier draft is removed, so there is no
``ocr_text`` and no ``flags`` field. Don't pre-add extensibility fields.

All state changes go through ``transition_instance()`` -- never assign
``instance.state`` directly anywhere else.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

import numpy as np
from pycocotools import mask as _mask_utils

PointSource = Literal["molmo", "human_added", "human_moved"]
InstanceState = Literal[
    "proposed", "needs_point_review", "needs_mask_review",
    "accepted", "rejected", "merged",
]

# ---------------------------------------------------------------------------
# Dataclasses (spec section 5)
# ---------------------------------------------------------------------------


@dataclass
class Point:
    id: str
    x: float
    y: float
    label: Literal[0, 1]                       # 1 = positive, 0 = negative
    source: PointSource
    original_xy: Optional[tuple[float, float]] = None  # pre-move loc for molmo points


@dataclass
class EditEvent:
    timestamp: str
    actor: str
    action: Literal[
        "point_add", "point_move", "point_delete", "point_label_toggle",
        "mask_accept", "mask_reject_manual_box", "state_change",
    ]
    detail: dict


@dataclass
class Instance:
    id: str
    image_id: str
    points: list[Point]
    mask_rle: Optional[str]                    # json string: {"size":[h,w],"counts":...}
    mask_source: Optional[Literal["sam3_auto", "human_box"]]
    state: InstanceState
    merged_into_id: Optional[str]
    created_at: str = ""
    updated_at: str = ""
    edit_log: list[EditEvent] = field(default_factory=list)


@dataclass
class SessionStats:
    points_added: int = 0
    points_moved: int = 0
    points_deleted: int = 0
    masks_manually_fixed: int = 0
    time_seconds: float = 0.0


@dataclass
class ImageRecord:
    id: str
    file_path: str
    width: int
    height: int
    instances: list[str]                       # instance ids, display order
    session_stats: SessionStats


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def make_point(x: float, y: float, label: int, source: PointSource) -> Point:
    return Point(id=new_id(), x=float(x), y=float(y), label=label, source=source)


def make_proposed_instance(image_id: str, x: float, y: float) -> Instance:
    """A fresh Molmo-proposed instance: one positive point, no mask yet."""
    ts = now_iso()
    return Instance(
        id=new_id(), image_id=image_id,
        points=[make_point(x, y, label=1, source="molmo")],
        mask_rle=None, mask_source=None,
        state="proposed", merged_into_id=None,
        created_at=ts, updated_at=ts,
    )


def make_empty_instance(image_id: str) -> Instance:
    """A human-created instance with no points yet (reviewer clicks to add
    them). Starts `proposed` like everything else so the state machine has a
    single entry path."""
    ts = now_iso()
    return Instance(
        id=new_id(), image_id=image_id, points=[],
        mask_rle=None, mask_source=None,
        state="proposed", merged_into_id=None,
        created_at=ts, updated_at=ts,
    )


def log_event(instance: Instance, actor: str, action: str, detail: dict) -> None:
    instance.edit_log.append(
        EditEvent(timestamp=now_iso(), actor=actor, action=action, detail=detail)
    )
    instance.updated_at = now_iso()


def has_human_point_action(instance: Instance) -> bool:
    """Acceptance criterion 6: no instance reaches `accepted` on raw Molmo
    output with zero logged human interaction on its point set."""
    return any(
        e.action in ("point_add", "point_move", "point_delete",
                     "point_label_toggle", "mask_reject_manual_box")
        or (e.action == "state_change" and e.detail.get("action") == "confirm_points")
        for e in instance.edit_log
    )


# ---------------------------------------------------------------------------
# State machine (spec section 6) -- the ONLY place instance.state is assigned
# ---------------------------------------------------------------------------

_TRANSITIONS: dict[tuple[str, str], InstanceState] = {
    # proposed -> needs_point_review | needs_mask_review   (on first open)
    ("proposed", "open_points"): "needs_point_review",
    ("proposed", "open_mask"): "needs_mask_review",
    # needs_point_review -> needs_mask_review | rejected | merged
    ("needs_point_review", "confirm_points"): "needs_mask_review",
    ("needs_point_review", "reject"): "rejected",
    ("needs_point_review", "merge"): "merged",
    # needs_mask_review -> accepted | needs_point_review
    ("needs_mask_review", "accept"): "accepted",
    ("needs_mask_review", "back_to_points"): "needs_point_review",
    # accepted/rejected/merged -> needs_point_review   (on reopen)
    ("accepted", "reopen"): "needs_point_review",
    ("rejected", "reopen"): "needs_point_review",
    ("merged", "reopen"): "needs_point_review",
}


def transition_instance(instance: Instance, action: str,
                        actor: str = "human") -> InstanceState:
    """Apply `action` to `instance`, mutating its state. Raises ValueError on
    an illegal transition so a UI bug can't silently corrupt annotation state."""
    key = (instance.state, action)
    if key not in _TRANSITIONS:
        raise ValueError(
            f"illegal transition: state={instance.state!r} action={action!r}"
        )
    new_state = _TRANSITIONS[key]
    log_event(instance, actor, "state_change",
              {"from": instance.state, "to": new_state, "action": action})
    instance.state = new_state
    if action == "reopen":
        instance.merged_into_id = None
    return new_state


# ---------------------------------------------------------------------------
# RLE mask helpers (pycocotools)
# ---------------------------------------------------------------------------


def mask_to_rle(mask: np.ndarray) -> str:
    """HxW bool array -> compact JSON string of the COCO compressed RLE."""
    rle = _mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return json.dumps({"size": rle["size"], "counts": rle["counts"].decode("ascii")})


def rle_to_mask(rle_str: str) -> np.ndarray:
    d = json.loads(rle_str)
    rle = {"size": d["size"], "counts": d["counts"].encode("ascii")}
    return _mask_utils.decode(rle).astype(bool)


# ---------------------------------------------------------------------------
# (De)serialization -- dataclasses.asdict + json, spec section 5
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1


def serialize(record: ImageRecord, instances: dict[str, Instance]) -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "image": asdict(record),
        "instances": [asdict(instances[iid]) for iid in record.instances
                      if iid in instances],
    }


def deserialize(data: dict) -> tuple[ImageRecord, dict[str, Instance]]:
    img = dict(data["image"])
    img["session_stats"] = SessionStats(**img["session_stats"])
    record = ImageRecord(**img)

    instances: dict[str, Instance] = {}
    for raw in data["instances"]:
        raw = dict(raw)
        raw["points"] = [
            Point(**{**p, "original_xy": tuple(p["original_xy"])
                     if p.get("original_xy") else None})
            for p in raw["points"]
        ]
        raw["edit_log"] = [EditEvent(**e) for e in raw["edit_log"]]
        inst = Instance(**raw)
        instances[inst.id] = inst
    return record, instances
