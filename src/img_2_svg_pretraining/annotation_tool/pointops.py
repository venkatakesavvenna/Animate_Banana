"""Pure point-interaction logic (spec section 10), kept out of app.py so the
most failure-prone part of the tool -- click resolution: select vs two-click
move vs add -- is unit-testable without Streamlit.

Click resolution order, given a click at full-image coords (x, y) on the
active instance:
1. Within SELECT_RADIUS of one of the instance's points -> SELECT it
   (nearest wins). Clicking the already-selected point deselects it.
2. Else, if some point is currently selected -> MOVE it to (x, y)
   (this is the second click of the two-click move).
3. Else, if add mode is on -> ADD a new point (negative if negative mode).
4. Else -> no-op.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from .datamodel import Instance, Point, SessionStats, log_event, make_point

SELECT_RADIUS = 10.0        # px in full-image coordinates


@dataclass
class ClickAction:
    kind: Literal["select", "deselect", "move", "add", "none"]
    point_id: Optional[str] = None      # select/deselect/move target


def resolve_click(
    instance: Instance | None,
    selected_point_id: str | None,
    x: float, y: float,
    add_mode: bool,
    select_radius: float = SELECT_RADIUS,
) -> ClickAction:
    if instance is None:
        return ClickAction("none")

    nearest, nearest_d = None, float("inf")
    for p in instance.points:
        d = math.hypot(p.x - x, p.y - y)
        if d < nearest_d:
            nearest, nearest_d = p, d
    if nearest is not None and nearest_d <= select_radius:
        if nearest.id == selected_point_id:
            return ClickAction("deselect", point_id=nearest.id)
        return ClickAction("select", point_id=nearest.id)

    if selected_point_id is not None and any(
            p.id == selected_point_id for p in instance.points):
        return ClickAction("move", point_id=selected_point_id)

    if add_mode:
        return ClickAction("add")
    return ClickAction("none")


# ---------------------------------------------------------------------------
# Mutations -- each logs an EditEvent and updates session stats
# ---------------------------------------------------------------------------


def apply_add(instance: Instance, x: float, y: float, negative: bool,
              stats: SessionStats, actor: str = "human") -> Point:
    point = make_point(x, y, label=0 if negative else 1, source="human_added")
    instance.points.append(point)
    log_event(instance, actor, "point_add",
              {"point_id": point.id, "x": x, "y": y, "label": point.label})
    stats.points_added += 1
    return point


def apply_move(instance: Instance, point_id: str, x: float, y: float,
               stats: SessionStats, actor: str = "human") -> Point:
    point = next(p for p in instance.points if p.id == point_id)
    old = (point.x, point.y)
    if point.source == "molmo":
        point.original_xy = old
        point.source = "human_moved"
    point.x, point.y = float(x), float(y)
    log_event(instance, actor, "point_move",
              {"point_id": point.id, "from": list(old), "to": [x, y]})
    stats.points_moved += 1
    return point


def apply_delete(instance: Instance, point_id: str,
                 stats: SessionStats, actor: str = "human") -> None:
    point = next(p for p in instance.points if p.id == point_id)
    instance.points.remove(point)
    log_event(instance, actor, "point_delete",
              {"point_id": point.id, "x": point.x, "y": point.y,
               "label": point.label})
    stats.points_deleted += 1


def apply_label_toggle(instance: Instance, point_id: str,
                       actor: str = "human") -> Point:
    point = next(p for p in instance.points if p.id == point_id)
    point.label = 0 if point.label == 1 else 1
    log_event(instance, actor, "point_label_toggle",
              {"point_id": point.id, "label": point.label})
    return point


def positive_points(instance: Instance) -> list[Point]:
    return [p for p in instance.points if p.label == 1]


def can_accept(instance: Instance) -> bool:
    """Acceptance criterion 4: never acceptable with zero positive points --
    including the has-negatives-but-no-positives case. Recomputed every
    rerun by the caller; never cached."""
    return len(positive_points(instance)) > 0 and instance.mask_rle is not None
