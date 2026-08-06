"""Stage-1c critic loop: the gate, and the accept/reject rules.

The loop's judgement calls are what these cover, because a bug in any of them
silently ships worse code than the pipeline already had:

  * the gate (don't pay for repairs the code doesn't need),
  * only keep a repair that scores *higher*,
  * reject a repair that drops `xml id`s (breaks every later stage),
  * reject a repair that doesn't compile.

Everything is driven through fakes -- no API, no LaTeX. `refine` takes its
model calls from module-level functions, so substituting them exercises the
real control flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from img_2_svg_pretraining.pipeline.transmuter import critic

CODE = """\\documentclass{standalone}
\\begin{document}
\\begin{tikzpicture}
\\node[xml id=a, xml class=block] (a) {A};
\\node[xml id=b, xml class=child_node] (b) {B};
\\end{tikzpicture}
\\end{document}"""


@dataclass
class FakeRender:
    ok: bool
    png_path: Path | None
    log: str = ""


class FakeCtx:
    """Just enough AgentContext for refine()."""

    def __init__(self):
        self.calls: list[str] = []

    def params(self):
        return {}

    class _Paths:
        def compile_cache(self):
            return Path("/tmp/fake_cache")

    paths = _Paths()


@pytest.fixture
def patched(monkeypatch):
    """Install fakes; tests set `state` to script the responses."""
    state = {
        "renders_ok": True,          # do candidate renders compile?
        "scores": [],                # consumed in order by score_render
        "findings": [{"problem": "x", "cause": "y", "code_fragment": "z",
                      "fix": "w", "severity": "critical"}],
        "repair": None,              # code returned by repair()
        "diagnose_calls": 0,
        "repair_calls": 0,
    }

    def fake_render(code, cache_dir):
        # The baseline always compiles; candidates obey the flag.
        ok = True if code == CODE else state["renders_ok"]
        return FakeRender(ok=ok, png_path=Path("/tmp/r.png") if ok else None)

    def fake_score(ctx, sample, render_path, params):
        score = state["scores"].pop(0) if state["scores"] else None
        return score, {}, "note"

    def fake_diagnose(ctx, sample, code, render_path, params):
        state["diagnose_calls"] += 1
        return state["findings"]

    def fake_repair(ctx, sample, code, render_path, findings, params):
        state["repair_calls"] += 1
        return state["repair"]

    monkeypatch.setattr(critic, "_render", fake_render)
    monkeypatch.setattr(critic, "score_render", fake_score)
    monkeypatch.setattr(critic, "diagnose", fake_diagnose)
    monkeypatch.setattr(critic, "repair", fake_repair)
    return state


def run(state, threshold=0.7, max_rounds=3):
    return critic.refine(FakeCtx(), object(), CODE, threshold, max_rounds)


# -- the gate --------------------------------------------------------------

def test_gate_skips_repair_when_already_faithful(patched):
    patched["scores"] = [0.85]
    code, info = run(patched)
    assert code == CODE
    assert "skipped" in info
    assert patched["diagnose_calls"] == 0     # cost: one scoring call, no more
    assert patched["repair_calls"] == 0


def test_gate_opens_exactly_at_threshold(patched):
    """0.7 with threshold 0.7 is good enough; anything below is not."""
    patched["scores"] = [0.7]
    _, info = run(patched)
    assert "skipped" in info and patched["diagnose_calls"] == 0


def test_gate_engages_below_threshold(patched):
    patched["scores"] = [0.30, 0.80]
    patched["repair"] = CODE.replace("{A}", "{A fixed}")
    code, info = run(patched)
    assert patched["diagnose_calls"] == 1
    assert info["baseline_score"] == 0.30
    assert info["final_score"] == 0.80
    assert code != CODE


# -- accept / reject -------------------------------------------------------

def test_repair_that_scores_worse_is_discarded(patched):
    """The critic can degrade code; the best version must survive, not the last."""
    patched["scores"] = [0.40, 0.20]
    patched["repair"] = CODE.replace("{A}", "{WORSE}")
    code, info = run(patched)
    assert code == CODE                    # original kept
    assert info["final_score"] == 0.40
    assert info["improved"] is False


def test_repair_that_drops_xml_ids_is_rejected(patched):
    """Later stages address elements by id; losing one breaks them."""
    patched["scores"] = [0.40, 0.99]       # would have scored well
    patched["repair"] = CODE.replace("\\node[xml id=b, xml class=child_node] (b) {B};", "")
    code, info = run(patched)
    assert code == CODE
    assert info["final_score"] == 0.40
    assert any("dropped xml id" in (r["notes"] or "") for r in info["rounds"])


def test_repair_that_does_not_compile_is_rejected(patched):
    """Never trade a rendering defect for a broken document."""
    patched["scores"] = [0.40]
    patched["renders_ok"] = False
    patched["repair"] = CODE + "\n\\broken"
    code, info = run(patched)
    assert code == CODE
    assert any("does not compile" in (r["notes"] or "") for r in info["rounds"])


def test_stops_early_once_threshold_reached(patched):
    patched["scores"] = [0.40, 0.90]
    patched["repair"] = CODE.replace("{A}", "{A2}")
    _, info = run(patched, max_rounds=3)
    assert info["rounds_run"] == 1          # did not spend the remaining budget
    assert info["final_score"] == 0.90


def test_respects_round_budget_when_never_converging(patched):
    """Improving but never reaching threshold: spend the budget, keep the best."""
    patched["scores"] = [0.10, 0.20, 0.30, 0.40]
    patched["repair"] = CODE.replace("{A}", "{A2}")
    _, info = run(patched, max_rounds=3)
    assert info["rounds_run"] == 3
    assert info["final_score"] == 0.40
    assert info["improved"] is True


def test_no_findings_stops_the_loop(patched):
    patched["scores"] = [0.50]
    patched["findings"] = []
    code, info = run(patched)
    assert code == CODE
    assert patched["repair_calls"] == 0


def test_unparseable_repair_stops_cleanly(patched):
    patched["scores"] = [0.50]
    patched["repair"] = None               # extractor found no code block
    code, info = run(patched)
    assert code == CODE
    assert any("no code" in (r["notes"] or "") for r in info["rounds"])


# -- degenerate inputs -----------------------------------------------------

def test_non_compiling_baseline_is_repaired_first(patched, monkeypatch):
    """A document that produces no PDF is fixed from its log, then scored.

    Three of five real bench samples failed this way -- an undefined colour,
    an undefined style, a reference to an undeclared node -- so the loop has
    to clear the compile error before it can compare anything.
    """
    fixed = CODE.replace("{A}", "{A repaired}")
    state = {"n": 0}

    def render(code, cache):
        # The original never compiles; the repaired version does.
        ok = code == fixed
        return FakeRender(ok=ok, png_path=Path("/tmp/r.png") if ok else None,
                          log="! Package xcolor Error: Undefined color `amber'.")

    monkeypatch.setattr(critic, "_render", render)
    monkeypatch.setattr(critic, "fix_compile",
                        lambda ctx, s, code, log, params: fixed)
    patched["scores"] = [0.80]          # repaired version scores above the gate

    code, info = run(patched)
    assert code == fixed
    assert info["compile_repaired"] is True
    assert any(r.get("compiles") for r in info["compile_rounds"])


def test_compile_repair_that_drops_ids_is_rejected(patched, monkeypatch):
    """Silencing an error by deleting the offending element loses content."""
    stripped = CODE.replace("\\node[xml id=b, xml class=child_node] (b) {B};", "")
    monkeypatch.setattr(critic, "_render",
                        lambda code, cache: FakeRender(
                            ok=False, png_path=None, log="! error"))
    monkeypatch.setattr(critic, "fix_compile",
                        lambda ctx, s, code, log, params: stripped)

    code, info = run(patched)
    assert code == CODE                        # the stripped version is refused
    assert "does not compile" in info["skipped"]
    assert any("dropped xml id" in r.get("notes", "")
               for r in info["compile_rounds"])


def test_gives_up_after_budget_when_repair_never_compiles(patched, monkeypatch):
    monkeypatch.setattr(critic, "_render",
                        lambda code, cache: FakeRender(
                            ok=False, png_path=None, log="! error"))
    monkeypatch.setattr(critic, "fix_compile",
                        lambda ctx, s, code, log, params: code + "\n% try")
    _, info = run(patched, max_rounds=2)
    assert "does not compile" in info["skipped"]
    assert len(info["compile_rounds"]) == 2    # spent the budget, then stopped


def test_latex_errors_extracts_the_bang_lines():
    log = ("This is pdfTeX\n(preamble noise)\n"
           "! Package xcolor Error: Undefined color `amber'.\n"
           "l.42 \\node[fill=amber]\n"
           "more context\n"
           "(further noise)\n")
    out = critic._latex_errors(log)
    assert "Undefined color" in out
    assert "preamble noise" not in out


def test_unscoreable_baseline_is_left_alone(patched):
    patched["scores"] = [None]
    code, info = run(patched)
    assert code == CODE
    assert "could not score" in info["skipped"]


def test_findings_are_ordered_worst_first():
    """A truncated fix prompt should still carry the damaging items."""
    raw = [{"problem": "p1", "severity": "minor"},
           {"problem": "p2", "severity": "critical"},
           {"problem": "p3", "severity": "major"}]

    class Backend:
        def generate(self, *a, **k):
            import json
            from types import SimpleNamespace
            return SimpleNamespace(ok=True, text=json.dumps({"findings": raw}))

    class Ctx:
        backend = Backend()

    class Sample:
        image_path = Path("/tmp/fig.png")

    got = critic.diagnose(Ctx(), Sample(), "code", Path("/tmp/r.png"), {})
    assert [f["severity"] for f in got] == ["critical", "major", "minor"]
