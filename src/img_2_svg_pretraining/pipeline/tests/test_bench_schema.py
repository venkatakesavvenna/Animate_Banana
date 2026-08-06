"""Reading and writing the AnimateBench sequence dialect.

The benchmark, and the `tikz_sequencer`/`svg_sequencer` prompts, express a
sequence as a flat time-indexed list with elements split into four buckets.
Ours is a hierarchy plus a traversal. `AnimationSequence` reads both so the
bench prompts work unchanged and our output stays directly comparable with the
reference.

The bucket split is load-bearing rather than cosmetic: the bounding-box styles
are defined partly by `text` and `arrows` being empty at every timestamp, so a
conversion that flattened everything into `focus` would destroy the very thing
those styles are checked against.
"""
from __future__ import annotations

from img_2_svg_pretraining.pipeline.schema import AnimationSequence

BENCH = {
    "metadata": {"traversal_order": "detail_first", "animation_style": "progressive_reveal"},
    "sequence": [
        {
            "timestamp": 1,
            "narrative": "The pipeline begins with a 3D smooth convex.",
            "to_be_animated": {
                "blocks": [{"id": "block_step1", "depth": 2}],
                "nodes": [],
                "text": [{"id": "label_step1", "depth": 2, "attached_to": "block_step1"}],
                "arrows": [{"id": "arrow1", "depth": 2, "source": "block_step1",
                            "target": "block_step2"}],
            },
        },
        {
            "timestamp": 2,
            "narrative": None,
            "to_be_animated": {
                "blocks": [{"id": "block_step2", "depth": 2}],
                "nodes": [], "text": [], "arrows": [],
            },
        },
    ],
}


def test_detects_bench_dialect():
    seq = AnimationSequence.from_dict(BENCH)
    assert len(seq.nodes) == 2
    assert seq.style == "progressive_reveal"
    assert seq.traversal_style == "DETAIL_FIRST"


def test_focus_is_union_of_buckets():
    seq = AnimationSequence.from_dict(BENCH)
    assert seq.nodes[0].focus == ["block_step1", "label_step1", "arrow1"]


def test_buckets_are_preserved_separately():
    """Without this, the bounding-box styles cannot be checked at all."""
    seq = AnimationSequence.from_dict(BENCH)
    assert seq.nodes[0].element_classes == {
        "blocks": ["block_step1"], "nodes": [],
        "text": ["label_step1"], "arrows": ["arrow1"],
    }


def test_narrative_and_timestamp_survive():
    seq = AnimationSequence.from_dict(BENCH)
    assert seq.nodes[0].narrative == "The pipeline begins with a 3D smooth convex."
    assert seq.nodes[1].narrative is None
    assert [n.timestamp for n in seq.nodes] == [1, 2]


def test_depth_is_shallowest_element():
    seq = AnimationSequence.from_dict(BENCH)
    assert seq.nodes[0].depth == 2


def test_traversal_follows_timestamp_order():
    seq = AnimationSequence.from_dict(BENCH)
    assert seq.traversal == ["t1", "t2"]


def test_round_trip_is_lossless():
    back = AnimationSequence.from_dict(BENCH).to_bench_dict()
    assert back["metadata"] == BENCH["metadata"]
    for original, produced in zip(BENCH["sequence"], back["sequence"]):
        assert produced["timestamp"] == original["timestamp"]
        assert produced["narrative"] == original["narrative"]
        for bucket in ("blocks", "nodes", "text", "arrows"):
            assert ([e["id"] for e in produced["to_be_animated"][bucket]]
                    == [e["id"] for e in original["to_be_animated"][bucket]])


def test_bench_survives_a_save_load_cycle():
    """bench -> native JSON -> native JSON must keep the bucket split.

    Every stage hands off through disk in our own dialect, so a sequence read
    from a bench-format model response is written natively and re-read by the
    next agent. An earlier version serialized `element_classes` but never read
    it back, so the buckets survived exactly until the first save and the
    output stopped being convertible to bench format at all.
    """
    once = AnimationSequence.from_dict(BENCH)
    twice = AnimationSequence.from_json(once.to_json())

    assert twice.nodes[0].element_classes == once.nodes[0].element_classes
    assert [n.timestamp for n in twice.nodes] == [1, 2]

    # And it must still convert back to the bench dialect after the cycle.
    back = twice.to_bench_dict()
    assert [e["id"] for e in back["sequence"][0]["to_be_animated"]["text"]] == ["label_step1"]
    assert [e["id"] for e in back["sequence"][0]["to_be_animated"]["arrows"]] == ["arrow1"]


def test_our_dialect_still_parses():
    """The native schema must keep working -- every cached artifact uses it."""
    native = {
        "style": "progressive_reveal",
        "traversal_style": "OVERVIEW_FIRST",
        "nodes": [{"id": "n1", "parent": None, "depth": 1, "focus": ["a"],
                   "action": "reveal", "duration": 3.0}],
        "traversal": ["n1"],
    }
    seq = AnimationSequence.from_dict(native)
    assert seq.nodes[0].id == "n1"
    assert seq.nodes[0].element_classes == {}
    assert seq.traversal_style == "OVERVIEW_FIRST"


def test_native_sequence_exports_to_bench_format():
    """A sequence with no bucket info still converts, with focus as nodes."""
    seq = AnimationSequence.from_dict({
        "style": "colour_pop",
        "traversal_style": "DETAIL_FIRST",
        "nodes": [{"id": "n1", "depth": 1, "focus": ["a", "b"], "action": "reveal"}],
        "traversal": ["n1"],
    })
    out = seq.to_bench_dict()
    assert out["metadata"]["animation_style"] == "colour_pop"
    assert out["metadata"]["traversal_order"] == "detail_first"
    step = out["sequence"][0]
    assert [e["id"] for e in step["to_be_animated"]["nodes"]] == ["a", "b"]
    assert step["to_be_animated"]["arrows"] == []


def test_string_entries_tolerated():
    """Some outputs give bare ids instead of {"id": ...} objects."""
    seq = AnimationSequence.from_dict({
        "metadata": {"animation_style": "alpha_masking"},
        "sequence": [{"timestamp": 1,
                      "to_be_animated": {"blocks": ["b1"], "nodes": [], "text": [],
                                         "arrows": []}}],
    })
    assert seq.nodes[0].focus == ["b1"]


def test_malformed_entries_skipped_not_fatal():
    """An element with no id is dropped rather than aborting the sequence.

    The bench's own bundle contains exactly this defect (a `{"bp_arrow",
    "depth": 3}` entry missing its `"id":` key), so tolerating it is what
    lets a run proceed on real data.
    """
    seq = AnimationSequence.from_dict({
        "metadata": {"animation_style": "progressive_reveal"},
        "sequence": [{"timestamp": 1,
                      "to_be_animated": {"blocks": [{"depth": 3}, {"id": "ok"}],
                                         "nodes": [], "text": [], "arrows": []}}],
    })
    assert seq.nodes[0].focus == ["ok"]
