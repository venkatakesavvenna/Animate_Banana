"""AnimateBench metrics, on fixtures with hand-computed answers.

Every metric here is checked against a case small enough to verify by eye,
including the ones that motivated the design: many-to-one alignment (the
reference draws three thumbnails where the candidate draws one), and depth
conventions that disagree between documents.

No API calls -- the alignment is constructed directly, which is exactly what
the judge's output becomes after validation.
"""
from __future__ import annotations

from img_2_svg_pretraining.animatebench import descriptions
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

# Every suite the viewer and the report know about. Named once so a new suite
# cannot be added without the formula/note invariants below covering it.
SUITES = ("stage1", "xml", "sequence", "stage3", "animation")

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


# -- descriptions: formulas and their instantiation -------------------------
#
# The explain view shows each score beside the formula that produced it. A
# formula that names the wrong record keys is worse than none at all -- it
# would present a confident, wrong account of how a number was reached. These
# tests pin the formulas to the records the metrics actually write.

def test_every_ordered_metric_has_a_formula_and_meaning():
    for suite in SUITES:
        for key in descriptions.ordered(suite):
            meta = descriptions.describe(key)
            assert meta.get("formula"), f"{key} has no formula"
            assert meta.get("what"), f"{key} has no description"


def test_every_suite_has_a_note():
    for suite in SUITES:
        assert descriptions.suite_note(suite).get("note"), suite


def test_instantiate_reproduces_a_ratio_from_its_own_terms():
    # PAA of 0.5 over 4 matched elements must read back as "2 / 4".
    record = {"paa": 0.5, "matched": 4, "scorable_gt_elements": 4}
    assert descriptions.instantiate("paa", record) == "2 / 4 = 0.5"


def test_instantiate_uses_counted_numerators_directly():
    record = {"matched_coverage": 0.5, "matched": 3, "scorable_gt_elements": 6}
    assert descriptions.instantiate("matched_coverage", record) == "3 / 6 = 0.5"


def test_instantiate_is_none_for_judged_metrics():
    """A judge score has no arithmetic to show; inventing one would lie."""
    record = {"rendering_fidelity": 0.79,
              "rendering_fidelity_breakdown": {"semantic": 0.85}}
    assert descriptions.instantiate("rendering_fidelity", record) is None


def test_instantiate_survives_a_null_denominator():
    """Bounding-box styles animate no arrows, so DOVR is null with 0 tested.

    That is a correct absence, not missing data, and must not become a
    division by zero or a fabricated 0/0.
    """
    assert descriptions.instantiate("dovr", {"dovr": None, "edges_tested": 0}) is None
    assert descriptions.instantiate("aif", {"aif": None, "aif_lines_touched": 0,
                                            "aif_lines_added": 0}) is None


def test_instantiate_prefers_a_stored_detail_string():
    """TOF states its own pair counts; show those rather than recomputing."""
    record = {"tof": 0.5, "tof_detail": "118/232 pairs front-loaded correctly"}
    assert descriptions.instantiate("tof", record) == \
        "118/232 pairs front-loaded correctly"


def test_edge_f1_shows_both_inputs():
    record = {"edge_f1": 0.72, "edge_precision": 0.6, "edge_recall": 0.9}
    out = descriptions.instantiate("edge_f1", record)
    assert "0.6" in out and "0.9" in out and "0.72" in out


def test_instantiate_ignores_unknown_metrics():
    assert descriptions.instantiate("not_a_metric", {"x": 1}) is None


# -- the animation evaluation tree ------------------------------------------
#
# The tree is judged end to end, so these use a stub judge that replays canned
# replies. What is being tested is never the model's opinion -- it is the
# machinery around it: that exactly one frame goes out per call, that state is
# threaded forward, that the counts are arithmetic over evidence rather than a
# model's own tally, and that a malformed reply degrades to missing data
# instead of to a score.

class StubJudge:
    """Replays scripted replies and records what it was asked."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def _next(self, prompt, images, videos=None):
        self.calls.append({"prompt": prompt, "images": list(images or []),
                           "videos": list(videos or [])})
        if not self._replies:
            raise AssertionError("stub judge ran out of replies")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def ask_json(self, prompt, images=None, tag="judge", videos=None, **kw):
        return self._next(prompt, images, videos)

    def ask_text(self, prompt, images=None, tag="judge", videos=None,
                 accept=None, **kw):
        return self._next(prompt, images, videos)

    def provenance(self):
        return {"judge_model": "stub"}


def _checklist(blocks=(), nodes=(), edges=()):
    """A Checklist from plain labels, one instance each.

    Entries carry a frequency now (several distinct elements routinely share
    one label). These tests are about the fold's behaviour rather than about
    frequency, so they name labels and let this fill in the rest; the
    frequency-specific cases build Entries directly.
    """
    from img_2_svg_pretraining.animatebench.checklist import Checklist, Entry

    def entries(labels):
        return [Entry(label=lbl) if isinstance(lbl, str) else lbl
                for lbl in labels]

    return Checklist(blocks=entries(blocks), nodes=entries(nodes),
                     edges=entries(edges))


def _frame_set(tmp, n, policy="all"):
    """A FrameSet over `n` throwaway JPEGs, bypassing the downscale step."""
    from img_2_svg_pretraining.animatebench.frames import FrameSet

    paths = []
    for i in range(1, n + 1):
        path = tmp / f"frame-{i:02d}.jpg"
        path.write_bytes(b"x")
        paths.append(path)
    return FrameSet(paths=paths, indices=list(range(n)), source_count=n,
                    policy=policy, long_edge=1568,
                    labels=[p.name for p in paths])


def test_frame_selection_matches_what_each_adapter_claims():
    """The adapter tells the judge what it is looking at; the policy decides
    what is attached. If they disagree, the judge scores the wrong thing under
    the right instructions and nothing downstream can tell."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    for style in aq.STYLES:
        expected = "all" if style == "alpha_masking" else "last"
        assert aq.VFS_POLICY[style] == expected, style
        assert aq.ASCS_POLICY[style] == "all", style
    # The design document's own sampling rule for the one 84-frame style.
    assert aq.WALK_POLICY["sliding_bounding_box"] == "every_4"
    assert aq.WALK_POLICY["progressive_reveal"] == "all"


def test_every_style_renders_every_prompt_without_leftover_placeholders():
    import re

    from img_2_svg_pretraining.animatebench.judge import PROMPTS_ROOT
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq
    from img_2_svg_pretraining.pipeline.prompts import load_and_render

    leftover = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    cases = [("animation_omission.yaml", aq.STYLES, {"remaining": "-"}),
             ("animation_style.yaml", aq.STYLES,
              # {previous_note} arrived when ASCS started sending the previous
              # frame as evidence for its temporal rules.
              {"style_name": "S", "frame_context": "c", "previous_note": "n"}),
             ("animation_repetition.yaml", aq.REPETITION_STYLES,
              {"frequencies": "-"})]
    for prompt_file, styles, extra in cases:
        for style in styles:
            text = load_and_render(f"{prompt_file}#frames", {
                "style_adapter": aq._adapter(prompt_file, style),
                "output_schema": aq._adapter(prompt_file, style, "schema"),
                **extra}, root=PROMPTS_ROOT)
            assert not leftover.search(text), f"{prompt_file}/{style}"


def test_fidelity_prompt_body_is_identical_across_styles():
    """One rubric, five adapters. Five copies of a rubric is five chances for
    it to differ between styles, which would make the scores incomparable."""
    from img_2_svg_pretraining.animatebench.judge import PROMPTS_ROOT
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq
    from img_2_svg_pretraining.pipeline.prompts import load_and_render

    for modality in ("frames", "video"):
        bodies = set()
        for style in aq.STYLES:
            adapter = load_and_render(
                f"animation_fidelity.yaml#adapter_{modality}_{style}", {},
                root=PROMPTS_ROOT)
            rendered = load_and_render(f"animation_fidelity.yaml#{modality}",
                                       {"style_adapter": adapter},
                                       root=PROMPTS_ROOT)
            bodies.add(rendered.replace(adapter, "<ADAPTER>"))
        assert len(bodies) == 1, f"{modality} body differs between styles"


def test_omission_sends_one_image_per_frame_and_counts_what_never_popped(tmp_path):
    """The invariant the whole design rests on: frames are never batched."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    checklist = _checklist(blocks=["Generator"], nodes=["Input Stack", "Depth Map"],
                          edges=["from: 'Input Stack' to: 'Generator'"])
    judge = StubJudge([
        {"reasoning": "a", "elements_popped_in_this_frame":
            {"blocks": ["Generator"], "nodes": [], "edges": []}},
        {"reasoning": "b", "elements_popped_in_this_frame":
            {"blocks": [], "nodes": ["Input Stack"], "edges": []}},
    ])
    frames = _frame_set(tmp_path, 2)
    out = aq.omitted_elements(judge, tmp_path / "fig.png", frames,
                              "progressive_reveal", checklist)

    assert len(judge.calls) == 2
    for call in judge.calls:                      # figure + exactly one frame
        assert len(call["images"]) == 2
    assert out["element_omission_count"] == 1     # "Depth Map" never appeared
    assert out["omission_elements_remaining"] == [
        {"label": "Depth Map", "unseen_count": 1}]
    assert out["arrow_omission_count"] == 1
    assert out["omission_rate"] == 1 / 3


def test_omission_decrements_per_instance_not_per_label(tmp_path):
    """Three nodes sharing the label "Layer Norm" are three things the
    animation owes. Seeing one must leave two outstanding, not clear all
    three -- that was how real omissions used to disappear."""
    from img_2_svg_pretraining.animatebench.checklist import Entry
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    checklist = _checklist(nodes=[Entry("Layer Norm", 3, ["n1", "n2", "n3"])])
    judge = StubJudge([
        {"elements_popped_in_this_frame":
            {"nodes": [{"label": "Layer Norm", "instances_seen": 1}]}},
        {"elements_popped_in_this_frame":
            {"nodes": [{"label": "Layer Norm", "instances_seen": 1}]}},
    ])
    out = aq.omitted_elements(judge, tmp_path / "fig.png",
                              _frame_set(tmp_path, 2), "progressive_reveal",
                              checklist)
    assert out["element_omission_count"] == 1          # 3 owed, 2 seen
    assert out["omission_elements_remaining"] == [
        {"label": "Layer Norm", "unseen_count": 1}]
    assert out["omission_rate"] == 1 / 3               # denominator is instances


def test_omission_cannot_pop_more_instances_than_are_owed(tmp_path):
    """A judge claiming five sightings of a label with two instances must not
    drive the count negative and manufacture a perfect score."""
    from img_2_svg_pretraining.animatebench.checklist import Entry
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    checklist = _checklist(nodes=[Entry("Norm", 2, ["n1", "n2"])],
                           blocks=["Never Shown"])
    judge = StubJudge([
        {"elements_popped_in_this_frame":
            {"nodes": [{"label": "Norm", "instances_seen": 5}]}},
    ])
    out = aq.omitted_elements(judge, tmp_path / "fig.png",
                              _frame_set(tmp_path, 1), "progressive_reveal",
                              checklist)
    # Norm is fully popped (clamped at 2); the untouched block still counts.
    assert out["element_omission_count"] == 1
    assert out["omission_elements_remaining"] == [
        {"label": "Never Shown", "unseen_count": 1}]


def test_omission_still_reads_bare_string_pops(tmp_path):
    """The frequency-aware schema is new; a model that answers in the older
    bare-string shape means one sighting, not zero."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    judge = StubJudge([
        {"elements_popped_in_this_frame": {"blocks": ["B"]}},
    ])
    out = aq.omitted_elements(judge, tmp_path / "fig.png",
                              _frame_set(tmp_path, 1), "progressive_reveal",
                              _checklist(blocks=["B"]))
    assert out["element_omission_count"] == 0


def test_omission_ignores_names_that_were_never_outstanding(tmp_path):
    """A judge naming something already popped, or never on the list, must not
    shrink it -- otherwise a hallucination silently improves the score."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    checklist = _checklist(blocks=["Generator"], nodes=["Input Stack"])
    judge = StubJudge([
        {"elements_popped_in_this_frame": {"blocks": ["Generator"]}},
        {"elements_popped_in_this_frame":
            {"blocks": ["Generator", "Invented Box"], "nodes": []}},
    ])
    out = aq.omitted_elements(judge, tmp_path / "fig.png",
                              _frame_set(tmp_path, 2), "progressive_reveal",
                              checklist)
    assert out["element_omission_count"] == 1
    assert out["omission_elements_remaining"] == [
        {"label": "Input Stack", "unseen_count": 1}]


def test_omission_ignores_edges_for_bounding_box_styles(tmp_path):
    """Boxes never highlight arrows, so 0 there means 'not applicable'."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    checklist = _checklist(blocks=["B"], edges=["from: 'B' to: 'C'"])
    judge = StubJudge([{"elements_popped_in_this_frame": {"blocks": ["B"]}}])
    out = aq.omitted_elements(judge, tmp_path / "fig.png",
                              _frame_set(tmp_path, 1), "hopping_bounding_box",
                              checklist)
    assert out["arrow_omission_count"] == 0
    assert out["omission_scores_edges"] is False


def test_a_failed_frame_does_not_lose_the_fold(tmp_path):
    """Losing one frame of thirty degrades the evidence; aborting loses the
    cell. The failure is recorded, not swallowed."""
    from img_2_svg_pretraining.animatebench.judge import JudgeError
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    judge = StubJudge([
        JudgeError("backend fell over"),
        {"elements_popped_in_this_frame": {"blocks": ["B"]}},
    ])
    out = aq.omitted_elements(judge, tmp_path / "fig.png",
                              _frame_set(tmp_path, 2), "progressive_reveal",
                              _checklist(blocks=["B"]))
    assert len(out["omission_errors"]) == 1
    assert out["element_omission_count"] == 0


def test_style_compliance_discards_on_any_discarded_frame(tmp_path):
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    judge = StubJudge([{"frame_verdict": "ACCEPT"}, {"frame_verdict": "DISCARD"},
                       {"frame_verdict": "ACCEPT"}])
    out = aq.style_compliance(judge, tmp_path / "fig.png",
                              _frame_set(tmp_path, 3), "progressive_reveal")
    assert out["ascs_pass"] is False
    assert out["ascs_frames_discarded"] == 1
    assert out["ascs_frames_judged"] == 3


def test_style_compliance_sends_the_previous_frame_as_temporal_evidence(tmp_path):
    """Every style has a rule phrased as "compare to the PREVIOUS frame".
    Asking that while showing one frame invites an invented answer: the judge
    cannot see what changed, but nothing stops it asserting that something
    did. Frame 1 has no predecessor and is told so instead."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    frames = _frame_set(tmp_path, 3)
    judge = StubJudge([{"frame_verdict": "ACCEPT"}] * 3)
    aq.style_compliance(judge, tmp_path / "fig.png", frames, "progressive_reveal")

    first, second, third = judge.calls
    # figure + current only, and the temporal rules explicitly excused
    assert len(first["images"]) == 2
    assert "NO previous frame is provided" in first["prompt"]
    assert "{previous_note}" not in first["prompt"]
    assert "PREVIOUS frame in the sequence" not in first["prompt"]

    # figure + previous + current, in that order
    for call, position in ((second, 1), (third, 2)):
        assert len(call["images"]) == 3
        assert call["images"][1] == frames.paths[position - 1]
        assert call["images"][2] == frames.paths[position]
        assert "NO previous frame" not in call["prompt"]
        assert "{previous_note}" not in call["prompt"]


def test_unenforceable_style_rules_are_recorded_not_passed(tmp_path):
    """A rule with nowhere to put its answer is recorded, never silently
    passed -- silence from such a rule reads exactly like compliance.

    Only ONE such rule survives. "Mobile Boxes, Static Elements" was
    unenforceable for both box styles until the schema gained a field for it
    and ASCS began sending the previous frame; what remains is sliding's
    "Hopping Boxes", which is genuinely sequence-level and cannot be settled
    from any single frame."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    judge = StubJudge([{"frame_verdict": "ACCEPT"}])
    out = aq.style_compliance(judge, tmp_path / "fig.png",
                              _frame_set(tmp_path, 1), "sliding_bounding_box")
    assert out["ascs_unenforced_rules"] == [
        "Hopping Boxes (HIGHLY CRITICAL): sequence-level, "
        "unanswerable from one frame"]
    # The box style whose only unenforceable rule became enforceable.
    judge = StubJudge([{"frame_verdict": "ACCEPT"}])
    assert aq.style_compliance(judge, tmp_path / "fig.png",
                               _frame_set(tmp_path, 1),
                               "hopping_bounding_box")["ascs_unenforced_rules"] == []
    judge = StubJudge([{"frame_verdict": "ACCEPT"}])
    assert aq.style_compliance(judge, tmp_path / "fig.png",
                               _frame_set(tmp_path, 1),
                               "progressive_reveal")["ascs_unenforced_rules"] == []


def test_repetition_is_unscored_where_no_prompt_exists(tmp_path):
    """Absent is not zero: 'no prompt was written' and 'no repetition found'
    are different claims, and only one of them is a measurement."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    for style in ("progressive_reveal", "colour_pop"):
        out = aq.unnecessary_repetition(StubJudge([]), tmp_path / "fig.png",
                                        _frame_set(tmp_path, 1), style,
                                        _checklist(blocks=["B"]))
        assert out["repetition_status"] == "not_specified"
        assert "unnecessary_repetition_count_element" not in out


def test_repetition_counts_only_unjustified_revisits(tmp_path):
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    judge = StubJudge([
        {"frequency_updates_this_frame": {"blocks": [{"name": "B", "new_frequency": 1}]},
         "repetition_evaluation": {"unjustified_freq_increase_element": []}},
        {"frequency_updates_this_frame": {"blocks": [{"name": "B", "new_frequency": 2}]},
         "repetition_evaluation": {"unjustified_freq_increase_element": ["B"]}},
        {"frequency_updates_this_frame": {"nodes": [{"name": "N", "new_frequency": 1}]},
         "repetition_evaluation": {"unjustified_freq_increase_element": []}},
    ])
    out = aq.unnecessary_repetition(judge, tmp_path / "fig.png",
                                    _frame_set(tmp_path, 3), "alpha_masking",
                                    _checklist(blocks=["B"], nodes=["N"]))
    assert out["repetition_frequencies"] == {"B": 2, "N": 1}
    assert out["repetition_revisited"] == ["B"]        # freq >= 2
    assert out["unnecessary_repetition_count_element"] == 1


def test_band_collapse_rules_differ_between_sss_and_gps():
    """SSS takes the lower band; GPS anchors on complexity then drops one.
    They look like one rule and are not -- crossing them would corrupt both."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    sss = aq.parse_bands("### RATIONALE\nwhy\n### CRITERION_A_BAND: 3\n"
                         "### CRITERION_B_BAND: 1\n### FINAL_BAND: 1\n"
                         "### LABEL: Major Violation", aq.SSS_KEYS)
    assert sss["criterion_a_band"] == 3 and sss["criterion_b_band"] == 1
    assert sss["final_band"] == 1 and sss["is_valid"]
    assert sss["label"] == "Major Violation"
    assert sss["rationale"] == "why"

    gps = aq.parse_bands("### RATIONALE\nr\n### VOLUME_BAND: 2\n"
                         "### COMPLEXITY_BAND: 2\n### FINAL_BAND: 1\n"
                         "### LABEL: Major Overload", aq.GPS_KEYS)
    assert gps["volume_band"] == 2 and gps["complexity_band"] == 2
    assert gps["final_band"] == 1 and gps["is_valid"]


def test_an_unparseable_band_is_missing_data_not_a_zero():
    """A parse failure must not land in the mean as a severe violation."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    verdict = aq.parse_bands("the animation looked fine to me", aq.SSS_KEYS)
    assert verdict["is_valid"] is False
    assert verdict["final_band"] is None


def test_checklist_validation_drops_what_it_cannot_use():
    """A hallucinated entry is unfalsifiable -- nothing on screen can ever
    match it, so it would count as an omission in every frame forever."""
    from img_2_svg_pretraining.animatebench import checklist as cl

    out = cl.validate({"animation_checklist": {
        "blocks": ["Generator", "Generator", "", 7],
        "nodes": ["Input Stack"],
        "edges": "not a list"}})
    assert [e.label for e in out.blocks] == ["Generator"]
    assert [e.label for e in out.nodes] == ["Input Stack"]
    assert out.edges == []
    # duplicate, empty string, non-string 7, and the edges list that wasn't one
    assert len(out.notes) == 4
    # Bare strings predate frequencies and mean one instance each.
    assert out.total() == 2


def test_checklist_frequency_counts_instances_not_labels():
    """Several distinct elements routinely share one label. Collapsing them
    would let an animation that showed one of three score as having shown
    all three, hiding two real omissions."""
    from img_2_svg_pretraining.animatebench import checklist as cl

    out = cl.validate({"animation_checklist": {
        "nodes": [{"label": "Layer Norm", "frequency": 2,
                   "source_ids": ["n1", "n2"]},
                  {"label": "Input", "frequency": 1, "source_ids": ["n3"]}],
        "blocks": [{"label": "Layer Norm", "frequency": 1,
                    "source_ids": ["b1"]}]}})
    # Same label in two buckets is legitimate -- a node and a block are
    # distinct entities, and the prompt says so explicitly.
    assert [(e.label, e.frequency) for e in out.nodes] == [
        ("Layer Norm", 2), ("Input", 1)]
    assert [(e.label, e.frequency) for e in out.blocks] == [("Layer Norm", 1)]
    assert out.total(False) == 4          # 2 + 1 + 1 instances, not 3 labels


def test_checklist_frequency_cannot_exceed_the_ids_named():
    """A frequency larger than the ids backing it is unfalsifiable padding:
    every unnamed instance would count as an omission forever."""
    from img_2_svg_pretraining.animatebench import checklist as cl

    out = cl.validate({"animation_checklist": {
        "nodes": [{"label": "Ghost", "frequency": 9, "source_ids": ["n1"]},
                  {"label": "Zero", "frequency": 0, "source_ids": []}]}})
    by_label = {e.label: e.frequency for e in out.nodes}
    assert by_label["Ghost"] == 1         # clamped to the one id it could name
    assert by_label["Zero"] == 1          # a sub-1 frequency still means one
    assert any("frequency 9" in n for n in out.notes)


def test_step_frames_refuses_an_ambiguous_mapping(tmp_path):
    """26 of 29 cells map 1:1. Where they do not, a frame from the wrong step
    yields a band that is wrong while looking plausible."""
    from img_2_svg_pretraining.animatebench import frames as fr

    paths = [tmp_path / f"frame-{i}.png" for i in range(1, 18)]
    assert fr.step_frames(paths, 17) is not None
    assert fr.step_frames(paths, 67) is None      # pipe00004 / progressive_reveal
    assert fr.step_frames([], 3) is None


def test_sequence_view_buckets_from_the_xml_not_the_sequence():
    """27 of 30 stored sequences are native dialect, where every focus id would
    otherwise be reported as a node and every blocks/edges list come back
    empty -- a plumbing bug that would read as a finding."""
    from img_2_svg_pretraining.animatebench.checklist import sequence_view

    xml = DiagramXml.parse("""<Diagram depth="1">
      <block id="b1" depth="2"><child_node id="n1" depth="3" /></block>
      <edge id="e1" source="n1" target="b1" depth="2" />
    </Diagram>""")
    seq = AnimationSequence.from_dict({"style": "progressive_reveal",
        "nodes": [{"id": "t1", "focus": ["b1", "n1", "e1"], "action": "reveal"}],
        "traversal": ["t1"]})
    view = sequence_view(seq, xml)
    assert "blocks=['b1']" in view
    assert "nodes=['n1']" in view
    assert "edges=['e1']" in view


# -- the video judge (E1b) -------------------------------------------------

def _tiny_png(path):
    """A 2x2 PNG, written by hand so the tests need no PIL."""
    import base64
    path.write_bytes(base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR4nGP8"
        b"z8Dwn4GBgYEJRIAAAA8EAQGvOQEHAAAAAElFTkSuQmCC"))
    return path


def test_video_part_changes_the_cache_fingerprint(tmp_path):
    """The one that matters.

    A VideoPart the fingerprint does not hash makes two requests that differ
    only in their video collide on one cache entry, and the second cell is
    served the first's verdict -- silently, with a plausible score. This is not
    hypothetical: the TikZ and SVG bench configs share a dataset root, so the
    same (sample, style) cell sends an identical prompt and an identical source
    image in both, differing only in the mp4.
    """
    from img_2_svg_pretraining.pipeline.backends import Message
    from img_2_svg_pretraining.pipeline.backends.base import _fingerprint

    image = _tiny_png(tmp_path / "source.png")
    a = tmp_path / "a.mp4"; a.write_bytes(b"\x00\x00\x00\x18ftypmp42AAAA")
    b = tmp_path / "b.mp4"; b.write_bytes(b"\x00\x00\x00\x18ftypmp42BBBB")

    msg_a = Message.user("identical prompt", images=[image], videos=[a])
    msg_b = Message.user("identical prompt", images=[image], videos=[b])

    assert _fingerprint("m", [msg_a], {}) != _fingerprint("m", [msg_b], {})
    # And identical bytes must still share an entry, or the cache is useless.
    same = tmp_path / "a_copy.mp4"; same.write_bytes(a.read_bytes())
    msg_same = Message.user("identical prompt", images=[image], videos=[same])
    assert _fingerprint("m", [msg_a], {}) == _fingerprint("m", [msg_same], {})


def test_fingerprint_refuses_a_part_it_cannot_hash():
    """An unhashable part must raise, not fall through. A silent fall-through
    is precisely how the video collision above would have gone unnoticed."""
    import pytest

    from img_2_svg_pretraining.pipeline.backends import Message
    from img_2_svg_pretraining.pipeline.backends.base import _fingerprint

    class RoguePart:
        pass

    msg = Message(role="user", content=[RoguePart()])
    with pytest.raises(TypeError, match="cannot hash"):
        _fingerprint("m", [msg], {})


def test_video_part_order_matches_what_the_prompt_claims(tmp_path):
    """The video prompt says "the original source diagram as a still image,
    followed by a video". Parts must actually arrive in that order."""
    from img_2_svg_pretraining.pipeline.backends import (
        ImagePart, Message, TextPart, VideoPart)

    image = _tiny_png(tmp_path / "s.png")
    video = tmp_path / "v.mp4"; video.write_bytes(b"\x00")
    msg = Message.user("prompt", images=[image], videos=[video])
    assert [type(p) for p in msg.content] == [ImagePart, VideoPart, TextPart]
    assert msg.videos()[0].media_type() == "video/mp4"


def test_video_fidelity_sends_one_image_and_one_video(tmp_path):
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    judge = StubJudge([{"video_evaluation": {
        "assessment": {"element_alteration": "fine"},
        "temporal_defects_observed": "None observed",
        "visual_fidelity_score": 7.5}}])
    image = _tiny_png(tmp_path / "s.png")
    video = tmp_path / "v.mp4"; video.write_bytes(b"\x00")

    out = aq.video_fidelity(judge, image, video, "progressive_reveal")
    assert len(judge.calls) == 1
    assert len(judge.calls[0]["images"]) == 1
    assert len(judge.calls[0]["videos"]) == 1
    assert out["vfs_video"] == 0.75
    assert out["vfs_video_raw"] == 7.5
    assert out["vfs_video_temporal_defects"] == "None observed"


def test_video_fidelity_accepts_a_bare_payload(tmp_path):
    """Some replies omit the `video_evaluation` wrapper. Parse those too rather
    than recording a null score for a judgement the model actually made."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    judge = StubJudge([{"visual_fidelity_score": 10.0,
                        "temporal_defects_observed": "None"}])
    image = _tiny_png(tmp_path / "s.png")
    video = tmp_path / "v.mp4"; video.write_bytes(b"\x00")
    assert aq.video_fidelity(judge, image, video, "colour_pop")["vfs_video"] == 1.0


def test_video_fidelity_records_a_missing_score_instead_of_raising(tmp_path):
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    judge = StubJudge([{"video_evaluation": {"assessment": {}}}])
    image = _tiny_png(tmp_path / "s.png")
    video = tmp_path / "v.mp4"; video.write_bytes(b"\x00")
    out = aq.video_fidelity(judge, image, video, "alpha_masking")
    assert out["vfs_video"] is None
    assert out["vfs_video_errors"]


def test_vfs_video_is_opt_in_and_never_runs_by_default():
    """Pins the decision. vfs_video is the only node calling a quota'd external
    judge on every cell; folding it into STAGES would make every default full
    run spend, and would change what `set(stages) != set(STAGES)` means in
    run_eval's partial-run merge."""
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    assert "vfs_video" not in aq.STAGES
    assert "vfs_video" in aq.EXTRA_STAGES
    assert set(aq.ALL_STAGES) == set(aq.STAGES) | set(aq.EXTRA_STAGES)


def test_video_prompt_renders_for_every_style_without_placeholders():
    """The video body and its five adapters were transcribed from the design
    doc but had no caller until now, so nothing had ever rendered them."""
    import re

    from img_2_svg_pretraining.animatebench.judge import PROMPTS_ROOT
    from img_2_svg_pretraining.animatebench.metrics.animation_quality import (
        STYLES, _adapter)
    from img_2_svg_pretraining.pipeline.prompts import load_and_render

    for style in STYLES:
        text = load_and_render("animation_fidelity.yaml#video", {
            "style_adapter": _adapter("animation_fidelity.yaml", style,
                                      "adapter_video"),
        }, root=PROMPTS_ROOT)
        leftover = [m for m in re.findall(r"\{([a-zA-Z_]\w*)\}", text)]
        assert not leftover, f"{style}: unrendered placeholders {leftover}"
        assert "visual_fidelity_score" in text
        assert "temporal_defects_observed" in text


def test_ascs_reports_frames_sent_and_frames_verdicted_separately(tmp_path):
    """`FrameSet.manifest()` emits `frames_judged`, which becomes
    `ascs_frames_judged` and collides with the count of frames that came back
    with a parseable verdict. The manifest meaning wins -- it is what all 64
    stored records already hold, and what compare_judges.py reads it as -- so
    the unambiguous count gets its own key rather than redefining the old one.

    Three frames are sent; the middle reply omits `frame_verdict`, so only two
    are verdicted. Before the fix both numbers were reported as 3.
    """
    from img_2_svg_pretraining.animatebench.frames import FrameSet
    from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq

    paths = [_tiny_png(tmp_path / f"frame-{i}.png") for i in (1, 2, 3)]
    frames = FrameSet(paths=paths, indices=[0, 1, 2], source_count=3,
                      policy="all", long_edge=1568,
                      labels=[p.name for p in paths])
    ok = {"frame_verdict": "ACCEPT", "generic_quality_checks": {},
          "style_specific_checks": {}}
    unparseable = {"generic_quality_checks": {}, "style_specific_checks": {}}
    judge = StubJudge([ok, unparseable, ok])

    out = aq.style_compliance(judge, _tiny_png(tmp_path / "s.png"), frames,
                              "progressive_reveal")

    assert out["ascs_frames_judged"] == 3     # frames SENT (from the manifest)
    assert out["ascs_frames_verdicted"] == 2  # frames with a real verdict
    assert out["ascs_pass"] is True
    assert len(out["ascs_frame_detail"]) == 3


# -- keyframe extraction ---------------------------------------------------

def test_adaptive_activity_threshold_scales_with_the_video():
    """The published extractors use a fixed activity_threshold of 0.01. A
    hopping bounding box on a 4236x4236 figure peaks at 0.0027, so under the
    fixed rule the animation reads as perfectly static from start to finish and
    a 32-frame video extracts to one keyframe. The threshold must scale to the
    video's own range."""
    from img_2_svg_pretraining.animatebench.keyframes import (
        adaptive_activity_threshold as thr)

    small = [0.0000, 0.0012, 0.0023, 0.0027, 0.0014]   # measured, arch00554
    large = [0.0, 0.10, 0.25, 0.30, 0.05]
    assert 0 < thr(small) < 0.01, "must sit below the fixed 0.01 on small motion"
    assert thr(large) > thr(small), "must rise with the video's own range"
    # A genuinely static video has no threshold that means anything.
    assert thr([0.0, 0.0, 0.0]) == float("inf")
    assert thr([]) == float("inf")


def test_dwell_fraction_uses_the_adaptive_threshold():
    """Measured against the fixed 0.01, a hopping-box video whose every frame
    differs reported dwell 1.00 -- 'nothing ever moves', the opposite of the
    truth -- and would have been sent to a detector that then found nothing."""
    from img_2_svg_pretraining.animatebench.keyframes import _dwell_fraction

    moving = [0.0012, 0.0023, 0.0023, 0.0022, 0.0027, 0.0014, 0.0015]
    assert _dwell_fraction(moving) < 1.0
    assert _dwell_fraction([0.0, 0.0, 0.0, 0.0]) == 1.0


def test_keyframe_names_sort_under_the_projects_own_frame_index():
    """The extracted deck must be a drop-in frames_dir. list_frames sorts by
    export.render._frame_index, which concatenates every digit in the stem, so
    the originals' mixed `0_frame_0000` / `state_0001_frame_0012` scheme yields
    0 and 10012 -- an order that holds by luck. Zero-padded frame-NN is what
    the rest of the pipeline already assumes."""
    from pathlib import Path

    from img_2_svg_pretraining.pipeline.export.render import _frame_index

    names = [f"frame-{n:02d}.png" for n in range(1, 13)]
    assert [_frame_index(Path(n)) for n in names] == list(range(1, 13))
    # The scheme it replaces, for contrast.
    assert _frame_index(Path("0_frame_0000.png")) == 0
    assert _frame_index(Path("state_0001_frame_0012.png")) == 10012


def test_every_style_has_a_keyframe_extractor():
    """A new style must fail loudly here rather than falling through to a
    default that happens to be wrong for it."""
    from img_2_svg_pretraining.animatebench.keyframes import EXTRACTORS
    from img_2_svg_pretraining.animatebench.metrics.animation_quality import STYLES

    assert set(EXTRACTORS) == set(STYLES)


def test_ssim_fallback_matches_skimage():
    """scikit-image is absent from the pipeline container and installing it can
    move numpy off the 1.26.4 pin cv2 and torch are built against, so SSIM is
    implemented on cv2 primitives. The two must agree, or every threshold tuned
    against one silently means something else under the other.

    This SKIPS in both environments, so it cannot be left to CI to notice. It
    was run once, deliberately, against a real scikit-image installed off to one
    side so the pinned environment stayed untouched -- and it passed to 1e-6:

        docker exec <container> bash -lc 'PIP_CONSTRAINT= \\
          /environments/img_2_svg_pretraining/bin/pip install --target=/tmp/skimage_check \\
          --no-deps scikit-image==0.24.0'
        docker exec <container> bash -lc 'cd /code && \\
          PYTHONPATH=/tmp/skimage_check:src <python> -m pytest \\
          ...::test_ssim_fallback_matches_skimage'

    Re-run it that way if `_ssim_map` is ever touched.
    """
    import pytest

    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    skimage_metrics = pytest.importorskip("skimage.metrics")

    from img_2_svg_pretraining.animatebench.keyframes import _ssim_map

    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    b = a.copy()
    b[20:40, 20:40] = 0

    _, reference = skimage_metrics.structural_similarity(
        a, b, data_range=255, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False, full=True)
    assert np.allclose(_ssim_map(a, b), reference, atol=1e-6)
