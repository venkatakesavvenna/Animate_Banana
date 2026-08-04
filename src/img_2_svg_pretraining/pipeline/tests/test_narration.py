"""Stage 2e: grafting narration, and aligning frames to narration clips.

Both units under test exist to absorb a mismatch that really happens:

- `_graft` defends against a model that edits the sequence it was told only to
  annotate. The prompt forbids it; this is the enforcement.
- `align_frames_to_clips` absorbs the frame/step ratio, which is per-sample.
  Real runs produced 4 steps/17 frames and 7 steps/41 as well as exact
  matches, so any 1:1 assumption would break most samples.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from img_2_svg_pretraining.pipeline.export.narration import Clip, align_frames_to_clips
from img_2_svg_pretraining.pipeline.planner.narrative_writer import _graft
from img_2_svg_pretraining.pipeline.schema import AnimationSequence


def make_sequence(n: int = 3) -> AnimationSequence:
    return AnimationSequence.from_dict({
        "style": "progressive_reveal",
        "traversal_style": "OVERVIEW_FIRST",
        "nodes": [
            {"id": f"node_{i:02d}", "parent": None if i == 1 else "node_01",
             "depth": 1 if i == 1 else 2, "focus": [f"elem_{i}"],
             "action": "reveal", "duration": 3.0}
            for i in range(1, n + 1)
        ],
        "traversal": [f"node_{i:02d}" for i in range(1, n + 1)],
    })


# -- grafting --------------------------------------------------------------

def test_graft_fills_narrative_by_id():
    seq = make_sequence(2)
    out, notes = _graft(seq, {"nodes": [
        {"id": "node_01", "narrative": "First the encoder."},
        {"id": "node_02", "narrative": "Then the decoder."},
    ]})
    assert [n.narrative for n in out.nodes] == ["First the encoder.", "Then the decoder."]
    assert notes == []


def test_graft_matches_by_id_not_position():
    """Order in the response is irrelevant; ids are the join key."""
    seq = make_sequence(2)
    out, _ = _graft(seq, {"nodes": [
        {"id": "node_02", "narrative": "second"},
        {"id": "node_01", "narrative": "first"},
    ]})
    assert out.node_by_id("node_01").narrative == "first"
    assert out.node_by_id("node_02").narrative == "second"


def test_graft_discards_structural_edits():
    """A response that rewrites focus/depth/traversal changes nothing but text."""
    seq = make_sequence(2)
    original_focus = [list(n.focus) for n in seq.nodes]
    out, _ = _graft(seq, {
        "traversal_style": "DETAIL_FIRST",
        "traversal": ["node_02"],
        "nodes": [
            {"id": "node_01", "narrative": "kept",
             "focus": ["something_invented"], "depth": 9, "action": "zoom"},
        ],
    })
    assert [list(n.focus) for n in out.nodes] == original_focus
    assert out.traversal == ["node_01", "node_02"]
    assert out.traversal_style == "OVERVIEW_FIRST"
    assert out.nodes[0].depth == 1
    assert out.nodes[0].action == "reveal"
    assert out.nodes[0].narrative == "kept"


def test_graft_reports_invented_ids():
    seq = make_sequence(1)
    _, notes = _graft(seq, {"nodes": [
        {"id": "node_01", "narrative": "ok"},
        {"id": "node_99", "narrative": "hallucinated"},
    ]})
    assert any("unknown node id" in n for n in notes)


def test_graft_reports_missing_nodes():
    seq = make_sequence(3)
    _, notes = _graft(seq, {"nodes": [{"id": "node_01", "narrative": "only one"}]})
    assert any("absent from the response" in n for n in notes)


def test_graft_accepts_narration_key_as_fallback():
    """Models sometimes echo the sequencer's field name instead."""
    seq = make_sequence(1)
    out, _ = _graft(seq, {"nodes": [{"id": "node_01", "narration": "via old key"}]})
    assert out.nodes[0].narrative == "via old key"


def test_graft_treats_blank_as_no_narration():
    seq = make_sequence(2)
    out, notes = _graft(seq, {"nodes": [
        {"id": "node_01", "narrative": "   "},
        {"id": "node_02", "narrative": None},
    ]})
    assert [n.narrative for n in out.nodes] == [None, None]
    assert any("no narration produced" in n for n in notes)


# -- script / jsonl --------------------------------------------------------

def test_narration_script_follows_traversal_order():
    seq = make_sequence(3)
    seq.traversal = ["node_03", "node_01", "node_02"]
    for node, text in zip(seq.nodes, ["one", "two", "three"]):
        node.narrative = text
    script = seq.narration_script()
    assert [e["timestamp"] for e in script] == [1, 2, 3]
    assert [e["text"] for e in script] == ["three", "one", "two"]


def test_narration_jsonl_keeps_silent_steps():
    """A null step must keep its slot or every later timestamp shifts."""
    seq = make_sequence(3)
    seq.nodes[1].narrative = "middle only"
    lines = seq.narration_jsonl().strip().splitlines()
    assert len(lines) == 3
    import json
    assert [json.loads(x)["text"] for x in lines] == [None, "middle only", None]


# -- frame alignment -------------------------------------------------------

def clips(durations: list[float]) -> list[Clip]:
    return [Clip(i, Path(f"a{i}.wav"), d, "t")
            for i, d in enumerate(durations, 1)]


def frames(n: int) -> list[Path]:
    return [Path(f"frame_{i:04d}.png") for i in range(1, n + 1)]


def test_align_one_to_one():
    timeline = align_frames_to_clips(frames(3), clips([2.0, 3.0, 4.0]))
    assert [round(h, 3) for _, h in timeline] == [2.0, 3.0, 4.0]


def test_align_more_frames_than_steps_preserves_total_duration():
    """The 4-step/17-frame case seen in a real run."""
    durations = [2.0, 3.0, 4.0, 5.0]
    timeline = align_frames_to_clips(frames(17), clips(durations))
    assert len(timeline) == 17
    assert round(sum(h for _, h in timeline), 6) == round(sum(durations), 6)


def test_align_distributes_frames_across_steps():
    timeline = align_frames_to_clips(frames(6), clips([3.0, 3.0]))
    # Three frames per step, each holding a third of that step's clip.
    assert [round(h, 3) for _, h in timeline] == [1.0] * 6


def test_align_fewer_frames_than_steps_keeps_audio_length():
    """Audio must never outrun the video, even when steps render no page."""
    durations = [2.0, 3.0, 4.0, 5.0]
    timeline = align_frames_to_clips(frames(2), clips(durations))
    assert round(sum(h for _, h in timeline), 6) == round(sum(durations), 6)
    assert len(timeline) <= 2


def test_align_every_frame_used_once_when_counts_match_ratio():
    timeline = align_frames_to_clips(frames(41), clips([1.0] * 7))
    assert len(timeline) == 41
    assert len({p for p, _ in timeline}) == 41


def test_align_requires_both_sides():
    with pytest.raises(Exception):
        align_frames_to_clips([], clips([1.0]))
    with pytest.raises(Exception):
        align_frames_to_clips(frames(1), [])
