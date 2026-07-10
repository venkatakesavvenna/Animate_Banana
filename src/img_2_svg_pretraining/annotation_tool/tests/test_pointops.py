"""Click-resolution + point-mutation tests -- the two-click move pattern is
the most failure-prone part of the tool (spec build-order step 4)."""
import numpy as np
import pytest

from img_2_svg_pretraining.annotation_tool.datamodel import (
    SessionStats, make_proposed_instance, mask_to_rle,
)
from img_2_svg_pretraining.annotation_tool.pointops import (
    apply_add, apply_delete, apply_label_toggle, apply_move, can_accept,
    resolve_click,
)


@pytest.fixture
def inst():
    return make_proposed_instance("img", 100.0, 100.0)


@pytest.fixture
def stats():
    return SessionStats()


# --- resolve_click -----------------------------------------------------------

def test_click_near_point_selects(inst):
    a = resolve_click(inst, None, 105, 103, add_mode=True)
    assert a.kind == "select" and a.point_id == inst.points[0].id


def test_click_on_selected_point_deselects(inst):
    pid = inst.points[0].id
    a = resolve_click(inst, pid, 101, 99, add_mode=True)
    assert a.kind == "deselect"


def test_second_click_moves_selected(inst):
    pid = inst.points[0].id
    a = resolve_click(inst, pid, 300, 250, add_mode=True)
    assert a.kind == "move" and a.point_id == pid


def test_click_far_adds_when_add_mode(inst):
    assert resolve_click(inst, None, 300, 250, add_mode=True).kind == "add"
    assert resolve_click(inst, None, 300, 250, add_mode=False).kind == "none"


def test_stale_selection_falls_through_to_add(inst, stats):
    """Selected point deleted meanwhile -> click should NOT 'move' a ghost."""
    pid = inst.points[0].id
    apply_delete(inst, pid, stats)
    a = resolve_click(inst, pid, 300, 250, add_mode=True)
    assert a.kind == "add"


def test_nearest_point_wins(inst, stats):
    p2 = apply_add(inst, 120.0, 100.0, negative=False, stats=stats)
    a = resolve_click(inst, None, 112, 100, add_mode=True)  # 12px vs 8px
    assert a.kind == "select" and a.point_id == p2.id


def test_no_instance_is_noop():
    assert resolve_click(None, None, 10, 10, add_mode=True).kind == "none"


# --- mutations ---------------------------------------------------------------

def test_move_molmo_point_records_provenance(inst, stats):
    pid = inst.points[0].id
    apply_move(inst, pid, 140.0, 90.0, stats)
    p = inst.points[0]
    assert p.source == "human_moved"
    assert p.original_xy == (100.0, 100.0)
    assert (p.x, p.y) == (140.0, 90.0)
    # second move keeps the ORIGINAL molmo location
    apply_move(inst, pid, 150.0, 95.0, stats)
    assert inst.points[0].original_xy == (100.0, 100.0)
    assert stats.points_moved == 2
    assert [e.action for e in inst.edit_log] == ["point_move", "point_move"]


def test_add_negative_and_toggle(inst, stats):
    p = apply_add(inst, 50.0, 50.0, negative=True, stats=stats)
    assert p.label == 0 and p.source == "human_added"
    apply_label_toggle(inst, p.id)
    assert inst.points[-1].label == 1
    assert stats.points_added == 1


# --- can_accept (criterion 4) -----------------------------------------------

def test_can_accept_rules(inst, stats):
    assert not can_accept(inst)                      # positive point, no mask
    inst.mask_rle = mask_to_rle(np.ones((4, 4), bool))
    assert can_accept(inst)
    # negatives-but-no-positives: never acceptable
    apply_add(inst, 5, 5, negative=True, stats=stats)
    apply_delete(inst, inst.points[0].id, stats)
    assert [p.label for p in inst.points] == [0]
    assert not can_accept(inst)
