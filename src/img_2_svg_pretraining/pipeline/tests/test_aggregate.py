"""Aggregation over eval records.

Every test here pins a way of being *quietly* wrong. An aggregate that
double-counts or mis-averages still prints a plausible number, so none of these
failures would be visible in the output -- which is the whole reason they are
tests rather than a careful read.
"""
from __future__ import annotations

import json

import pytest

from img_2_svg_pretraining.animatebench import aggregate, results


def write(root, config, style, sample, suite, record):
    path = results.suite_path(root, config, style, sample, suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def evals(tmp_path):
    return tmp_path / "evals"


# -- collect() must not mistake a sibling directory for a model --------------

def test_collect_ignores_non_config_siblings(evals):
    """`evals/` holds alignment/, checklist/, raw/, _frames/, _compile_work/ and
    archived _stale_prompts_<date>/ beside the real configs, all at the same
    depth. Admitting one is not a crash -- it is a phantom model in the table.
    """
    write(evals, "real_model", "progressive_reveal", "s1", "stage1", {"csr": 1.0})
    # A checklist entry: same depth, but its filename is a sample id, not a suite.
    (evals / "checklist" / "real_model" / "progressive_reveal").mkdir(parents=True)
    (evals / "checklist" / "real_model" / "progressive_reveal" / "s1.json").write_text("{}")
    (evals / "alignment" / "real_model").mkdir(parents=True)
    (evals / "alignment" / "real_model" / "s1.json").write_text("{}")
    (evals / "_stale_prompts_2026-08-19" / "m" / "s").mkdir(parents=True)
    (evals / "_stale_prompts_2026-08-19" / "m" / "s" / "stage1.json").write_text('{"csr": 0.0}')

    assert sorted(results.collect(evals)) == ["real_model"]


# -- the four kinds of absence ----------------------------------------------

def test_indicator_metric_is_reported_as_a_rate_not_a_mean(evals):
    for i, csr in enumerate([1.0, 1.0, 0.0, 1.0]):
        write(evals, "m", "progressive_reveal", f"s{i}", "stage1", {"csr": csr})
    stat = aggregate.summarise(evals)["configs"]["m"]["all"]["metrics"]["stage1.csr"]
    assert stat["kind"] == "rate"
    assert stat["rate"] == pytest.approx(0.75)
    assert stat["n"] == 4
    # A median over 0/1 values is 0 or 1 and reads like a score; it is withheld.
    assert "median" not in stat


def test_not_specified_repetition_is_excluded_rather_than_counted_as_zero(evals):
    """`repetition_rate` is undefined for progressive_reveal and colour_pop.

    The record still carries `repetition_rate: 0.0` alongside
    `repetition_status: "not_specified"`. Averaging those in turns "we never
    measured this" into a perfect score -- and on a bench that is mostly
    progressive_reveal, it would dominate the column.
    """
    write(evals, "m", "progressive_reveal", "s1", "animation",
          {"repetition_rate": 0.0, "repetition_status": "not_specified"})
    write(evals, "m", "progressive_reveal", "s2", "animation",
          {"repetition_rate": 0.0, "repetition_status": "not_specified"})
    write(evals, "m", "alpha_masking", "s3", "animation",
          {"repetition_rate": 0.4, "repetition_status": "scored"})

    stat = aggregate.summarise(evals)["configs"]["m"]["all"]["metrics"]["animation.repetition_rate"]
    assert stat["n"] == 1                       # not 3
    assert stat["mean"] == pytest.approx(0.4)   # not 0.133
    assert stat["not_applicable"] == 2


def test_gated_fidelity_is_reported_both_ways(evals):
    """rendering_fidelity is forced to 0.0 when the diagram does not compile.

    Flat, that conflates "did not compile" with "compiled and looked wrong";
    conditional-only, it hides the failures. Both rows are emitted.
    """
    write(evals, "m", "progressive_reveal", "s1", "stage1",
          {"csr": 1.0, "rendering_fidelity": 0.9})
    write(evals, "m", "progressive_reveal", "s2", "stage1",
          {"csr": 1.0, "rendering_fidelity": 0.7})
    write(evals, "m", "progressive_reveal", "s3", "stage1",
          {"csr": 0.0, "rendering_fidelity": 0.0})

    metrics = aggregate.summarise(evals)["configs"]["m"]["all"]["metrics"]
    flat = metrics["stage1.rendering_fidelity"]
    gated = metrics["stage1.rendering_fidelity (compiled)"]
    assert flat["n"] == 3 and flat["mean"] == pytest.approx(1.6 / 3)
    assert gated["n"] == 2 and gated["mean"] == pytest.approx(0.8)
    assert gated["not_applicable"] == 1


def test_errored_suite_contributes_nothing_but_is_counted(evals):
    """A record that errored holds no metric keys at all, so rule 1 already
    drops it -- but the error itself must still be visible."""
    write(evals, "m", "progressive_reveal", "s1", "stage1", {"csr": 1.0})
    write(evals, "m", "progressive_reveal", "s2", "stage1",
          {"error": "boom", "suite": "stage1", "provenance": {}})

    entry = aggregate.summarise(evals)["configs"]["m"]["all"]
    assert entry["metrics"]["stage1.csr"]["n"] == 1
    assert entry["errors"] == 1


def test_per_frame_errors_do_not_discard_a_real_score(evals):
    """The animation suite isolates per-frame failures and still writes a
    number alongside `*_errors`. Those cells count -- only a TOP-LEVEL `error`
    means the suite produced nothing."""
    write(evals, "m", "progressive_reveal", "s1", "animation",
          {"vfs": 0.8, "vfs_errors": ["frame-3 timed out"]})
    stat = aggregate.summarise(evals)["configs"]["m"]["all"]["metrics"]["animation.vfs"]
    assert stat["n"] == 1 and stat["mean"] == pytest.approx(0.8)


# -- denominators ------------------------------------------------------------

def test_cells_denominator_exposes_an_incomplete_run(evals, tmp_path):
    """The failure this prevents: a model that crashed on most samples is
    averaged over its survivors, so failing MORE makes its mean look BETTER.
    Only `cells` distinguishes it from a model that finished everything.
    """
    dataset = tmp_path / "bench"
    for i in range(5):
        seq = dataset / f"s{i}" / "reference" / "seq"
        seq.mkdir(parents=True)
        (seq / f"progressive_reveal_s{i}_svg.json").write_text("{}")

    write(evals, "finished", "progressive_reveal", "s0", "stage1",
          {"csr": 1.0, "rendering_fidelity": 0.5, "target": "svg"})
    for i in range(1, 5):
        write(evals, "finished", "progressive_reveal", f"s{i}", "stage1",
              {"csr": 1.0, "rendering_fidelity": 0.5, "target": "svg"})
    # Crashed everywhere but its one best sample.
    write(evals, "crashed", "progressive_reveal", "s0", "stage1",
          {"csr": 1.0, "rendering_fidelity": 0.99, "target": "svg"})

    summary = aggregate.summarise(evals, dataset_root=dataset, target="svg")
    finished = summary["configs"]["finished"]["all"]
    crashed = summary["configs"]["crashed"]["all"]

    assert crashed["metrics"]["stage1.rendering_fidelity"]["mean"] > \
           finished["metrics"]["stage1.rendering_fidelity"]["mean"]
    assert (crashed["scored"], crashed["cells"]) == (1, 5)
    assert (finished["scored"], finished["cells"]) == (5, 5)


def test_cells_uses_each_configs_own_target(evals, tmp_path):
    """SVG and TikZ coverage differ, so one global --target would give the
    wrong denominator to whichever config is not it."""
    dataset = tmp_path / "bench"
    for i in range(3):
        seq = dataset / f"s{i}" / "reference" / "seq"
        seq.mkdir(parents=True)
        (seq / f"progressive_reveal_s{i}_svg.json").write_text("{}")
    (dataset / "s0" / "reference" / "seq" / "progressive_reveal_s0_tikz.json").write_text("{}")

    write(evals, "svg_run", "progressive_reveal", "s0", "stage1", {"csr": 1.0, "target": "svg"})
    write(evals, "tikz_run", "progressive_reveal", "s0", "stage1", {"csr": 1.0, "target": "tikz"})

    summary = aggregate.summarise(evals, dataset_root=dataset, target="svg")
    assert summary["configs"]["svg_run"]["all"]["cells"] == 3
    assert summary["configs"]["tikz_run"]["all"]["cells"] == 1


def test_styles_are_pooled_micro_not_macro(evals):
    """A 4/1 split: micro weights each cell equally, macro would weight the
    one-sample style as heavily as the four-sample one."""
    for i in range(4):
        write(evals, "m", "progressive_reveal", f"s{i}", "xml", {"paa": 0.5})
    write(evals, "m", "colour_pop", "s9", "xml", {"paa": 1.0})

    stat = aggregate.summarise(evals)["configs"]["m"]["all"]["metrics"]["xml.paa"]
    assert stat["mean"] == pytest.approx(0.6)   # micro: (4*0.5 + 1.0)/5
    assert stat["mean"] != pytest.approx(0.75)  # macro would be (0.5 + 1.0)/2
    assert stat["n"] == 5


# -- presentation ------------------------------------------------------------

def test_pass_metrics_render_with_an_up_arrow(evals):
    """`describe()["better"]` is "pass" for the indicator metrics -- neither
    "higher" nor "lower". Treating it as the else branch prints a down-arrow on
    a pass rate, which states the opposite of the truth."""
    write(evals, "m", "progressive_reveal", "s1", "sequence", {"sscr_pass": True})
    md = aggregate.to_markdown(aggregate.summarise(evals), scope="all")
    row = next(line for line in md.splitlines() if "Style-Schema" in line)
    assert "↑" in row and "↓" not in row


def test_metric_missing_for_one_model_is_a_dash_not_a_zero(evals):
    """The comparison case: model A was scored on a metric, model B was not.

    B's cell must read "—". A 0.000 there is indistinguishable from B scoring
    zero, which would make an unmeasured model look like the worst one.
    """
    write(evals, "has_it", "progressive_reveal", "s1", "stage1",
          {"csr": 1.0, "rendering_fidelity": 0.8})
    write(evals, "lacks_it", "progressive_reveal", "s1", "stage1", {"csr": 1.0})

    md = aggregate.to_markdown(aggregate.summarise(evals), scope="all")
    row = next(line for line in md.splitlines()
               if line.startswith("| Rendering Fidelity ↑"))
    has, lacks = [c.strip() for c in row.split("|")[2:4]]
    assert has.startswith("0.800")
    assert lacks == "—"


def test_a_row_empty_for_every_model_is_omitted_entirely(evals):
    """Distinct from the above: a metric nobody was scored on is not a row of
    dashes, it is not a row. Otherwise the table is mostly empty rows."""
    write(evals, "m", "progressive_reveal", "s1", "stage1", {"csr": 1.0})
    md = aggregate.to_markdown(aggregate.summarise(evals), scope="all")
    assert "Rendering Fidelity" not in md
    assert "Compilation Success Rate" in md


# -- ground truth must be read for the run's OWN target ----------------------

def test_ground_truth_accessors_honour_the_target(tmp_path):
    """`GroundTruth`'s accessors all default to target="tikz".

    On an SVG run that default silently reads TikZ ground truth, which exists
    only for samples that happen to ship both targets. Nothing errors -- the
    metrics just record `alignment_missing` and the column quietly shrinks. On
    the v4 bench that took the GT-derived columns from n=20 to n=2, and it was
    invisible on v3 because every v3 sample carries both targets.

    So this pins that an SVG-only sample is found via target="svg" and NOT via
    the default. If the default ever changes, this still passes; what it
    forbids is a caller silently getting the wrong target's file.
    """
    from img_2_svg_pretraining.animatebench.gt import load_ground_truth

    sample = tmp_path / "S1"
    (sample / "reference" / "xml").mkdir(parents=True)
    (sample / "reference" / "seq").mkdir(parents=True)
    (sample / "reference" / "xml" / "S1_svg.xml").write_text("<Diagram/>")
    (sample / "reference" / "seq" / "progressive_reveal_S1_svg.json").write_text("{}")

    gt = load_ground_truth(tmp_path, "S1")
    assert gt.xml_path("svg").exists()
    assert not gt.xml_path("tikz").exists()      # the trap: silently absent
    assert gt.has_style("progressive_reveal", "svg")
    assert not gt.has_style("progressive_reveal", "tikz")


def test_run_eval_passes_its_config_target_to_ground_truth():
    """Guards the actual call sites rather than the accessor contract.

    Every `gt.` call in run_eval must name the target explicitly; a bare
    `gt.xml()` there reintroduces the bug with no visible symptom.
    """
    import inspect
    import re

    from img_2_svg_pretraining.animatebench import run_eval

    src = inspect.getsource(run_eval)
    bare = re.findall(r"gt\.(?:xml|xml_path|sequence|sequence_path|has_style"
                      r"|diagram_code_path)\(\s*(?:style\s*)?\)", src)
    assert not bare, f"GT accessor called without a target in run_eval: {bare}"
