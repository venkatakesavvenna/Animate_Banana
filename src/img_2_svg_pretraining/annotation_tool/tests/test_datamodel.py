"""Standalone data-model tests (build-order step 1): fixture data only,
no Streamlit, no SAM3."""
import json
import multiprocessing as mp
import os

import numpy as np
import pytest

from img_2_svg_pretraining.annotation_tool import store
from img_2_svg_pretraining.annotation_tool.datamodel import (
    ImageRecord, SessionStats, deserialize, has_human_point_action, log_event,
    make_point, make_proposed_instance, mask_to_rle, rle_to_mask, serialize,
    transition_instance,
)


def _fixture(image_id="img_001"):
    inst_a = make_proposed_instance(image_id, 10.0, 20.0)
    inst_b = make_proposed_instance(image_id, 100.0, 200.0)
    record = ImageRecord(
        id=image_id, file_path=f"/data/{image_id}.png", width=640, height=480,
        instances=[inst_a.id, inst_b.id], session_stats=SessionStats(),
    )
    return record, {inst_a.id: inst_a, inst_b.id: inst_b}


# --- state machine ---------------------------------------------------------

def test_state_machine_happy_path():
    _, instances = _fixture()
    inst = next(iter(instances.values()))
    assert inst.state == "proposed"
    assert transition_instance(inst, "open_points") == "needs_point_review"
    assert transition_instance(inst, "confirm_points") == "needs_mask_review"
    assert transition_instance(inst, "accept") == "accepted"
    # reopen always re-enters at needs_point_review
    assert transition_instance(inst, "reopen") == "needs_point_review"
    # every transition logged as a state_change event
    actions = [e.action for e in inst.edit_log]
    assert actions.count("state_change") == 4


def test_state_machine_rejects_illegal_transitions():
    _, instances = _fixture()
    inst = next(iter(instances.values()))
    with pytest.raises(ValueError):
        transition_instance(inst, "accept")          # proposed -/-> accepted
    transition_instance(inst, "open_points")
    with pytest.raises(ValueError):
        transition_instance(inst, "accept")          # must pass mask review
    transition_instance(inst, "reject")
    assert inst.state == "rejected"
    with pytest.raises(ValueError):
        transition_instance(inst, "accept")          # rejected -/-> accepted


def test_merge_and_reopen_clears_merged_into():
    _, instances = _fixture()
    a, b = list(instances.values())
    transition_instance(a, "open_points")
    a.merged_into_id = b.id
    transition_instance(a, "merge")
    assert a.state == "merged"
    transition_instance(a, "reopen")
    assert a.merged_into_id is None


# --- RLE -------------------------------------------------------------------

def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    mask = rng.random((48, 64)) > 0.7
    rle = mask_to_rle(mask)
    assert isinstance(rle, str)
    json.loads(rle)  # valid json
    back = rle_to_mask(rle)
    assert back.dtype == bool and back.shape == (48, 64)
    assert np.array_equal(mask, back)


# --- serialization ---------------------------------------------------------

def test_serialize_roundtrip():
    record, instances = _fixture()
    inst = instances[record.instances[0]]
    transition_instance(inst, "open_points")
    inst.points.append(make_point(30, 40, label=0, source="human_added"))
    log_event(inst, "human", "point_add", {"point_id": inst.points[-1].id})
    inst.points[0].original_xy = (9.0, 19.0)
    inst.mask_rle = mask_to_rle(np.ones((4, 4), dtype=bool))
    inst.mask_source = "sam3_auto"

    data = json.loads(json.dumps(serialize(record, instances)))
    record2, instances2 = deserialize(data)
    assert record2 == record
    assert instances2 == instances
    assert instances2[inst.id].points[0].original_xy == (9.0, 19.0)


def test_human_action_guard():
    _, instances = _fixture()
    inst = next(iter(instances.values()))
    assert not has_human_point_action(inst)          # raw Molmo output
    transition_instance(inst, "open_points")
    assert not has_human_point_action(inst)          # opening isn't human review
    transition_instance(inst, "confirm_points")      # explicit confirm counts
    assert has_human_point_action(inst)


# --- persistence -----------------------------------------------------------

def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNOTATIONS_DIR", str(tmp_path))
    record, instances = _fixture()
    store.save_image_record(record, instances)
    loaded = store.load_image_record(record.id)
    assert loaded is not None
    record2, instances2 = loaded
    assert record2 == record and instances2 == instances
    assert store.load_image_record("nonexistent") is None
    assert store.list_annotated_image_ids() == [record.id]


def _writer_proc(ann_dir: str, image_id: str, n_writes: int):
    os.environ["ANNOTATIONS_DIR"] = ann_dir
    record, instances = _fixture(image_id)
    for _ in range(n_writes):
        store.save_image_record(record, instances)


def test_concurrent_writes_do_not_corrupt(tmp_path, monkeypatch):
    """Acceptance criterion 8: two processes writing the same annotations dir
    (same AND different images) leave every file valid JSON."""
    monkeypatch.setenv("ANNOTATIONS_DIR", str(tmp_path))
    procs = [
        mp.Process(target=_writer_proc, args=(str(tmp_path), "img_A", 25)),
        mp.Process(target=_writer_proc, args=(str(tmp_path), "img_A", 25)),
        mp.Process(target=_writer_proc, args=(str(tmp_path), "img_B", 25)),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert all(p.exitcode == 0 for p in procs)
    for image_id in ("img_A", "img_B"):
        loaded = store.load_image_record(image_id)
        assert loaded is not None
        record, instances = loaded
        assert len(instances) == 2 and record.id == image_id
