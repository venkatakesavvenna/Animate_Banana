"""AnimateBench metrics, on fixtures with hand-computed answers.

Every metric here is checked against a case small enough to verify by eye,
including the ones that motivated the design: many-to-one alignment (the
reference draws three thumbnails where the candidate draws one), and depth
conventions that disagree between documents.

No API calls -- the alignment is constructed directly, which is exactly what
the judge's output becomes after validation.
"""
from __future__ import annotations

from img_2_svg_pretraining.animatebench.alignment import Alignment, validate
from img_2_svg_pretraining.animatebench.gt import DiagramXml, first_appearance
from img_2_svg_pretraining.animatebench.metrics.stage2_sequence import (
    coverage, dependence_order, style_schema_compliance, traversal_order_fidelity,
)
from img_2_svg_pretraining.animatebench.metrics.stage2_xml import (
    depth_consistency, edge_topology, parent_assignment,
)
from img_2_svg_pretraining.animatebench.metrics.stage3_anim import integration_footprint
from img_2_svg_pretraining.pipeline.schema import AnimationSequence

# Reference: a block holding three thumbnails, one node, one edge between them.
GT_XML = """<Diagram depth="1">
  <block id="input_block" depth="2">
    <raster_node id="rgb1" depth="3" />
    <raster_node id="rgb2" depth="3" />
    <raster_node id="rgb3" depth="3" />
  </block>
  <block id="proc_block" depth="2">
    <child_node id="encoder" depth="3" />
  </block>
  <edge id="e_in_to_proc" source="rgb3" target="encoder" depth="1" />
</Diagram>"""

# Candidate: same figure, different granularity and different depth convention.
PRED_XML = """<Diagram depth="1">
  <block id="block_in" depth="1">
    <raster_node id="img_rgb" depth="2" />
  </block>
  <block id="block_proc" depth="1">
    <child_node id="box_encoder" depth="2" />
  </block>
  <edge id="edge_a" source="img_rgb" target="box_encoder" depth="1" />
</Diagram>"""

ALIGNMENT = Alignment(groups={
    "g0": {"gt": ["input_block"], "pred": ["block_in"]},
    "g1": {"gt": ["rgb1", "rgb2", "rgb3"], "pred": ["img_rgb"]},
    "g2": {"gt": ["proc_block"], "pred": ["block_proc"]},
    "g3": {"gt": ["encoder"], "pred": ["box_encoder"]},
})


def gt_xml() -> DiagramXml:
    return DiagramXml.parse(GT_XML)


def pred_xml() -> DiagramXml:
    return DiagramXml.parse(PRED_XML)


# -- depth consistency -----------------------------------------------------

def test_depth_consistency_ignores_root_convention():
    """The two documents number their roots differently; neither is wrong.

    GT declares top-level blocks at depth=2 and top-level edges at depth=1 in
    the same file. Only the parent relation is checked, so both score clean.
    """
    assert depth_consistency(gt_xml())["depth_violation_rate"] == 0.0
    assert depth_consistency(pred_xml())["depth_violation_rate"] == 0.0


def test_depth_consistency_flags_real_violation():
    bad = DiagramXml.parse("""<Diagram depth="1">
      <block id="b" depth="1"><child_node id="c" depth="1" /></block>
    </Diagram>""")
    result = depth_consistency(bad)
    assert result["depth_violation_rate"] == 1.0
    assert result["depth_violations"][0]["id"] == "c"
    assert result["depth_violations"][0]["expected"] == 2


# -- parent assignment -----------------------------------------------------

def test_paa_perfect_under_many_to_one():
    """rgb1..rgb3 all map to img_rgb, whose parent maps to input_block."""
    result = parent_assignment(gt_xml(), pred_xml(), ALIGNMENT)
    assert result["paa"] == 1.0
    assert result["matched_coverage"] == 1.0
    assert result["matched"] == 6     # 2 blocks + 3 thumbnails + encoder
    assert all(d["status"] == "ok" for d in result["parent_detail"])


def test_paa_detects_wrong_parent():
    """Candidate hangs the encoder under the input block instead."""
    wrong = DiagramXml.parse("""<Diagram depth="1">
      <block id="block_in" depth="1">
        <raster_node id="img_rgb" depth="2" />
        <child_node id="box_encoder" depth="2" />
      </block>
      <block id="block_proc" depth="1" />
    </Diagram>""")
    result = parent_assignment(gt_xml(), wrong, ALIGNMENT)
    bad = [d for d in result["parent_detail"] if d["status"] == "wrong_parent"]
    assert [d["gt"] for d in bad] == ["encoder"]
    assert result["paa"] == 5 / 6


def test_paa_reports_unmatched_separately():
    """A high PAA over few matches must not look like a good result."""
    thin = Alignment(groups={"g0": {"gt": ["input_block"], "pred": ["block_in"]}})
    result = parent_assignment(gt_xml(), pred_xml(), thin)
    assert result["paa"] == 1.0            # the one match is correct
    assert result["matched_coverage"] == 1 / 6   # but only one of six matched


# -- edge topology ---------------------------------------------------------

def test_edge_matches_after_contraction():
    """GT rgb3->encoder and pred img_rgb->box_encoder are the same edge."""
    result = edge_topology(gt_xml(), pred_xml(), ALIGNMENT)
    assert result["edge_precision"] == 1.0
    assert result["edge_recall"] == 1.0
    assert result["edge_f1"] == 1.0


def test_edge_spurious_is_caught():
    extra = DiagramXml.parse(PRED_XML.replace(
        '<edge id="edge_a" source="img_rgb" target="box_encoder" depth="1" />',
        '<edge id="edge_a" source="img_rgb" target="box_encoder" depth="1" />\n'
        '  <edge id="edge_b" source="box_encoder" target="img_rgb" depth="1" />'))
    result = edge_topology(gt_xml(), extra, ALIGNMENT)
    assert result["edge_recall"] == 1.0
    assert result["edge_precision"] == 0.5      # one of two is real
    assert result["spurious_pred_edges"] == [("g3", "g1")]


# -- sequence --------------------------------------------------------------

def bench_seq(steps: list[dict], traversal_order: str = "overview_first",
              style: str = "progressive_reveal") -> AnimationSequence:
    return AnimationSequence.from_dict({
        "metadata": {"traversal_order": traversal_order, "animation_style": style},
        "sequence": [{"timestamp": i, "to_be_animated": s}
                     for i, s in enumerate(steps, 1)],
    })


def test_first_appearance_uses_traversal_order():
    seq = bench_seq([
        {"blocks": [{"id": "block_in"}], "nodes": [], "text": [], "arrows": []},
        {"blocks": [], "nodes": [{"id": "img_rgb"}], "text": [], "arrows": []},
    ])
    assert first_appearance(seq) == {"block_in": 1, "img_rgb": 2}


def test_tof_overview_first_rewards_front_loading():
    good = bench_seq([
        {"blocks": [{"id": "block_in"}, {"id": "block_proc"}], "nodes": [],
         "text": [], "arrows": []},
        {"blocks": [], "nodes": [{"id": "img_rgb"}, {"id": "box_encoder"}],
         "text": [], "arrows": []},
    ])
    assert traversal_order_fidelity(good, pred_xml())["tof"] == 1.0


def test_tof_overview_first_penalises_early_detail():
    bad = bench_seq([
        {"blocks": [], "nodes": [{"id": "img_rgb"}], "text": [], "arrows": []},
        {"blocks": [{"id": "block_in"}, {"id": "block_proc"}], "nodes": [],
         "text": [], "arrows": []},
        {"blocks": [], "nodes": [{"id": "box_encoder"}], "text": [], "arrows": []},
    ])
    # img_rgb precedes both blocks -> 2 of 4 (top, deep) pairs are out of order.
    assert traversal_order_fidelity(bad, pred_xml())["tof"] == 0.5


def test_tof_detail_first_rewards_contiguous_blocks():
    contiguous = bench_seq([
        {"blocks": [{"id": "block_in"}], "nodes": [], "text": [], "arrows": []},
        {"blocks": [], "nodes": [{"id": "img_rgb"}], "text": [], "arrows": []},
        {"blocks": [{"id": "block_proc"}], "nodes": [], "text": [], "arrows": []},
        {"blocks": [], "nodes": [{"id": "box_encoder"}], "text": [], "arrows": []},
    ], traversal_order="detail_first")
    assert traversal_order_fidelity(contiguous, pred_xml())["tof"] == 1.0


def test_tof_detail_first_penalises_revisiting():
    zigzag = bench_seq([
        {"blocks": [{"id": "block_in"}], "nodes": [], "text": [], "arrows": []},
        {"blocks": [{"id": "block_proc"}], "nodes": [], "text": [], "arrows": []},
        {"blocks": [], "nodes": [{"id": "img_rgb"}], "text": [], "arrows": []},
        {"blocks": [], "nodes": [{"id": "box_encoder"}], "text": [], "arrows": []},
    ], traversal_order="detail_first")
    assert traversal_order_fidelity(zigzag, pred_xml())["tof"] < 1.0


# -- DOVR ------------------------------------------------------------------

def test_dovr_clean_when_endpoints_precede_edge():
    seq = bench_seq([
        {"blocks": [], "nodes": [{"id": "img_rgb"}, {"id": "box_encoder"}],
         "text": [], "arrows": []},
        {"blocks": [], "nodes": [], "text": [], "arrows": [{"id": "edge_a"}]},
    ])
    assert dependence_order(seq, pred_xml())["dovr"] == 0.0


def test_dovr_flags_edge_before_its_endpoints():
    seq = bench_seq([
        {"blocks": [], "nodes": [], "text": [], "arrows": [{"id": "edge_a"}]},
        {"blocks": [], "nodes": [{"id": "img_rgb"}, {"id": "box_encoder"}],
         "text": [], "arrows": []},
    ])
    result = dependence_order(seq, pred_xml())
    assert result["dovr"] == 1.0
    assert result["dovr_violations"][0]["edge"] == "edge_a"


# -- SSCR ------------------------------------------------------------------

def test_sscr_fails_bbox_style_with_text_or_arrows():
    seq = bench_seq([
        {"blocks": [{"id": "block_in"}], "nodes": [], "text": [{"id": "t1"}],
         "arrows": []},
    ], style="hopping_bounding_box")
    result = style_schema_compliance(seq, "hopping_bounding_box")
    assert result["sscr_pass"] is False
    assert any("text must be empty" in v for v in result["sscr_violations"])


def test_sscr_passes_bbox_style_with_only_blocks():
    seq = bench_seq([
        {"blocks": [{"id": "block_in"}], "nodes": [], "text": [], "arrows": []},
        {"blocks": [{"id": "block_proc"}], "nodes": [], "text": [], "arrows": []},
    ], style="hopping_bounding_box")
    assert style_schema_compliance(seq, "hopping_bounding_box")["sscr_pass"] is True


# -- coverage --------------------------------------------------------------

def test_coverage_matches_through_alignment():
    gt_seq = bench_seq([
        {"blocks": [{"id": "input_block"}], "nodes": [
            {"id": "rgb1"}, {"id": "rgb2"}, {"id": "rgb3"}], "text": [], "arrows": []},
        {"blocks": [{"id": "proc_block"}], "nodes": [{"id": "encoder"}],
         "text": [], "arrows": []},
    ])
    pred_seq = bench_seq([
        {"blocks": [{"id": "block_in"}], "nodes": [{"id": "img_rgb"}],
         "text": [], "arrows": []},
        {"blocks": [{"id": "block_proc"}], "nodes": [{"id": "box_encoder"}],
         "text": [], "arrows": []},
    ])
    result = coverage(gt_seq, pred_seq, ALIGNMENT, "progressive_reveal")
    # One predicted image covers three GT thumbnails: full coverage either way.
    assert result["coverage_recall"] == 1.0
    assert result["coverage_precision"] == 1.0


def test_coverage_recall_drops_on_omission():
    gt_seq = bench_seq([
        {"blocks": [{"id": "input_block"}, {"id": "proc_block"}], "nodes": [
            {"id": "rgb1"}, {"id": "encoder"}], "text": [], "arrows": []},
    ])
    pred_seq = bench_seq([
        {"blocks": [{"id": "block_in"}], "nodes": [{"id": "img_rgb"}],
         "text": [], "arrows": []},
    ])
    result = coverage(gt_seq, pred_seq, ALIGNMENT, "progressive_reveal")
    assert result["coverage_recall"] == 0.5      # 2 of 4 GT groups animated
    assert result["coverage_precision"] == 1.0   # nothing hallucinated


def test_coverage_excludes_text_and_arrows_for_bbox_styles():
    gt_seq = bench_seq([
        {"blocks": [{"id": "input_block"}], "nodes": [], "text": [], "arrows": []},
    ], style="hopping_bounding_box")
    pred_seq = bench_seq([
        {"blocks": [{"id": "block_in"}], "nodes": [], "text": [{"id": "ignored"}],
         "arrows": [{"id": "edge_a"}]},
    ], style="hopping_bounding_box")
    result = coverage(gt_seq, pred_seq, ALIGNMENT, "hopping_bounding_box")
    # text/arrows are out of contract for this style, so they must not count
    # against precision.
    assert result["coverage_precision"] == 1.0


# -- alignment validation --------------------------------------------------

def test_validate_drops_hallucinated_ids():
    raw = {"groups": [
        {"gt": ["input_block", "nonexistent"], "pred": ["block_in"]},
        {"gt": ["encoder"], "pred": ["invented_id"]},
    ]}
    alignment = validate(raw, gt_xml(), pred_xml())
    assert alignment.group_of_gt("input_block") is not None
    assert alignment.group_of_gt("nonexistent") is None
    # Second group lost its only prediction, so it is discarded entirely.
    assert alignment.group_of_gt("encoder") is None
    assert any("not in reference XML" in n for n in alignment.notes)


def test_validate_rejects_duplicate_placement():
    raw = {"groups": [
        {"gt": ["input_block"], "pred": ["block_in"]},
        {"gt": ["input_block"], "pred": ["block_proc"]},
    ]}
    alignment = validate(raw, gt_xml(), pred_xml())
    assert len(alignment.groups) == 1
    assert any("already in another group" in n for n in alignment.notes)


def test_validate_excludes_composite_parts():
    """Sub-part decomposition is language-dependent (design doc 2.1.3)."""
    with_parts = DiagramXml.parse("""<Diagram depth="1">
      <composite_whole id="w" depth="1"><composite_part id="p" depth="2" /></composite_whole>
    </Diagram>""")
    raw = {"groups": [{"gt": ["input_block"], "pred": ["p"]}]}
    alignment = validate(raw, gt_xml(), with_parts)
    assert alignment.group_of_pred("p") is None
    assert "p" not in alignment.pred_unmatched


# -- AIF -------------------------------------------------------------------

def test_aif_zero_for_purely_additive_animation():
    diagram = "\\begin{tikzpicture}\n\\node (a) {A};\n\\end{tikzpicture}"
    animation = ("\\begin{animateinline}{1}\n\\begin{tikzpicture}\n"
                 "\\node (a) {A};\n\\end{tikzpicture}\n\\end{animateinline}")
    assert integration_footprint(diagram, animation)["aif"] == 0.0


def test_aif_counts_rewritten_diagram_lines():
    diagram = "\\node (a) {A};\n\\node (b) {B};"
    animation = "\\node[opacity=\\opA] (a) {A};\n\\node (b) {B};\n\\multiframe{2}{}"
    result = integration_footprint(diagram, animation)
    assert result["aif_lines_touched"] == 1     # node (a) was rewritten
    assert result["aif"] > 0
