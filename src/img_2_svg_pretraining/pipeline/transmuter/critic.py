"""Stage 1c -- Diagram Critic: render, compare, diagnose, repair.

Closes the loop Stage 1 never had. The code converter (1a) and raster
integrator (1b) are both write-only: nothing ever looked at what the code
actually draws. A document can compile with zero warnings and still render
almost nothing, which is not a hypothetical -- on CVPR_2025_pipe00041 the
converter declared every `fit` block *after* the nodes it contains and gave
it an opaque fill, so each container painted over its own children. All 11
raster crops were embedded in the PDF and the figure was still empty.

One round is: score -> (gate) -> diagnose -> fix -> re-score -> (accept?).

Four decisions worth stating, because each is load-bearing:

- **Scoring and diagnosis are separate calls.** A model asked to grade and
  repair in one breath tends to justify its grade instead of finding the
  defect, and the score has to stay honest: it is both the entry gate and the
  accept/reject test.
- **A round is only kept if the score improves.** The critic can make things
  worse, so a repair that scores lower than what it replaced is discarded and
  the loop stops. The best version seen always wins, never the last one.
- **Repairs must compile.** A fix that breaks the document is rejected before
  it is ever scored, so the critic can never trade a rendering defect for a
  compile failure.
- **The gate is checked before any repair work.** Code already above
  threshold costs exactly one scoring call, which is what keeps the stage
  affordable on a good sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..backends import Message
from ..cache import write_text
from ..extract import extract_code, extract_json, looks_truncated
from ..prompts import load_and_render
from ..runner import AgentContext, SampleOutcome, StageReport
from ..samples import PaperSample

AGENT = "diagram_critic"

DEFAULT_THRESHOLD = 0.7
DEFAULT_MAX_ROUNDS = 3


@dataclass
class Attempt:
    """One version of the code and what it scored."""

    round_index: int          # 0 = the converter's own output
    code: str
    score: float | None
    compiles: bool
    render_path: Path | None = None
    breakdown: dict = field(default_factory=dict)
    notes: str = ""
    findings: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"round": self.round_index, "score": self.score,
                "compiles": self.compiles, "breakdown": self.breakdown,
                "notes": self.notes[:500], "findings": self.findings}


def _render(code: str, cache_dir: Path):
    # Imported lazily: compile_tikz shells out to latexmk, and keeping it out
    # of module scope means `--help` and config validation never touch it.
    from img_2_svg_pretraining.viewer.compile import compile_tikz
    return compile_tikz(code, cache_dir)


def score_render(ctx: AgentContext, sample: PaperSample, render_path: Path,
                 params: dict) -> tuple[float | None, dict, str]:
    """Rendering fidelity of one render against the source figure.

    Same rubric as the AnimateBench metric of the same name, so the loop's
    internal judgement and the external evaluation agree about what "faithful"
    means rather than optimising against different targets.
    """
    prompt = load_and_render("diagram_transmuter/critic.yaml#score", {})
    result = ctx.backend.generate(
        [Message.user(prompt, images=[sample.image_path, render_path])], **params)
    if not result.ok:
        return None, {}, f"scoring call failed: {result.error}"

    data = extract_json(result.text)
    if not isinstance(data, dict):
        reason = ("scoring response truncated" if looks_truncated(result.text or "")
                  else "scoring response was not JSON")
        return None, {}, reason

    raw = data.get("score")
    score = float(raw) if isinstance(raw, (int, float)) else None
    if score is not None:
        score = max(0.0, min(1.0, score))
    return score, data.get("breakdown") or {}, str(data.get("notes") or "")


def diagnose(ctx: AgentContext, sample: PaperSample, code: str,
             render_path: Path, params: dict) -> list[dict]:
    """What is wrong with this render, as line-specific findings."""
    prompt = load_and_render("diagram_transmuter/critic.yaml#diagnose",
                             {"diagram_code": code})
    result = ctx.backend.generate(
        [Message.user(prompt, images=[sample.image_path, render_path])], **params)
    if not result.ok:
        return []

    data = extract_json(result.text)
    if not isinstance(data, dict):
        return []

    findings = []
    for entry in data.get("findings") or []:
        if isinstance(entry, dict) and entry.get("problem"):
            findings.append({
                "problem": str(entry.get("problem"))[:400],
                "cause": str(entry.get("cause") or "")[:400],
                "code_fragment": str(entry.get("code_fragment") or "")[:300],
                "fix": str(entry.get("fix") or "")[:400],
                "severity": str(entry.get("severity") or "major"),
            })

    # Worst first, so a truncated fix prompt still carries the damaging items.
    rank = {"critical": 0, "major": 1, "minor": 2}
    findings.sort(key=lambda f: rank.get(f["severity"], 1))
    return findings


def repair(ctx: AgentContext, sample: PaperSample, code: str, render_path: Path,
           findings: list[dict], params: dict) -> str | None:
    """Apply the findings, returning corrected code (or None)."""
    listing = "\n".join(
        f"{i}. [{f['severity']}] {f['problem']}\n"
        f"   cause: {f['cause']}\n"
        f"   code:  {f['code_fragment']}\n"
        f"   fix:   {f['fix']}"
        for i, f in enumerate(findings, 1))

    prompt = load_and_render("diagram_transmuter/critic.yaml#fix",
                             {"diagram_code": code, "findings": listing})
    result = ctx.backend.generate(
        [Message.user(prompt, images=[sample.image_path, render_path])], **params)
    if not result.ok:
        return None
    return extract_code(result.text, ctx.cfg.target)


def _ids(code: str) -> set[str]:
    """`xml id` values declared in the code."""
    import re
    return set(re.findall(r"xml\s+id\s*=\s*\{?\"?([\w:.-]+)\"?\}?", code))


def _latex_errors(log: str, limit: int = 6) -> str:
    """The `!` lines from a latexmk log, which carry the actual diagnosis.

    The full log is thousands of lines of package banners; the error lines and
    the few lines after each are what a repair needs, and keeping the prompt
    short leaves budget for the code itself.
    """
    lines = log.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        if line.startswith("!"):
            out.extend(lines[i:i + 4])
            out.append("")
            if len([l for l in out if l.startswith("!")]) >= limit:
                break
    return "\n".join(out) if out else log[-1500:]


def fix_compile(ctx: AgentContext, sample: PaperSample, code: str, log: str,
                params: dict) -> str | None:
    """Repair a document that does not compile, from its LaTeX log."""
    prompt = load_and_render("diagram_transmuter/critic.yaml#compile_fix",
                             {"diagram_code": code, "compile_log": _latex_errors(log)})
    result = ctx.backend.generate([Message.user(prompt)], **params)
    if not result.ok:
        return None
    return extract_code(result.text, ctx.cfg.target)


def make_compilable(ctx: AgentContext, sample: PaperSample, code: str,
                    cache_dir: Path, params: dict,
                    max_rounds: int) -> tuple[str, object, list[dict]]:
    """Repair until the document compiles, or the budget runs out.

    Returns `(code, render_result, history)`. Runs before any rendering
    comparison, because a document that produces no PDF cannot be scored at
    all -- and these failures are the model referencing something it never
    defined (an undeclared colour, style, or node), which the LaTeX log names
    precisely enough to fix without seeing the figure.

    Repairs that drop element ids are rejected here too: silencing an error by
    deleting the element that triggered it loses figure content, which is
    worse than the error.
    """
    history: list[dict] = []
    rendered = _render(code, cache_dir)
    if rendered.ok:
        return code, rendered, history

    original_ids = _ids(code)
    for round_index in range(1, max_rounds + 1):
        candidate = fix_compile(ctx, sample, code, rendered.log, params)
        if candidate is None:
            history.append({"round": round_index, "phase": "compile",
                            "notes": "repair produced no code"})
            break

        lost = original_ids - _ids(candidate)
        if lost:
            history.append({"round": round_index, "phase": "compile",
                            "notes": f"rejected: dropped xml id(s) {sorted(lost)[:5]}"})
            break

        attempt = _render(candidate, cache_dir)
        history.append({
            "round": round_index, "phase": "compile", "compiles": bool(attempt.ok),
            "notes": ("now compiles" if attempt.ok
                      else _latex_errors(attempt.log, limit=1)[:200]),
        })
        code = candidate
        rendered = attempt
        if attempt.ok:
            break

    return code, rendered, history


def refine(ctx: AgentContext, sample: PaperSample, code: str,
           threshold: float, max_rounds: int) -> tuple[str, dict]:
    """Run the critic loop over one sample. Returns (best code, report)."""
    params = ctx.params()
    cache_dir = ctx.paths.compile_cache()
    history: list[Attempt] = []

    # Phase 1: make it compile. A document that produces no PDF cannot be
    # compared against anything, and these failures are self-contained --
    # the model used a colour, style or node it never declared, and the
    # LaTeX log names which.
    code, baseline_render, compile_history = make_compilable(
        ctx, sample, code, cache_dir, params, max_rounds)

    if not baseline_render.ok or baseline_render.png_path is None:
        return code, {"skipped": "diagram does not compile after repair attempts",
                      "compile_rounds": compile_history, "rounds": []}

    score, breakdown, notes = score_render(ctx, sample, baseline_render.png_path, params)
    best = Attempt(0, code, score, True, baseline_render.png_path, breakdown, notes)
    history.append(best)

    # Every early return carries the compile history too: a sample that only
    # renders because it was repaired first is a materially different result
    # from one that compiled on its own, and the report must not lose that.
    common = {"compile_rounds": compile_history,
              "compile_repaired": bool(compile_history)}

    if score is None:
        return code, {**common, "skipped": f"could not score baseline ({notes})",
                      "rounds": [best.summary()]}
    if score >= threshold:
        return code, {**common,
                      "skipped": f"baseline fidelity {score:.2f} >= {threshold}",
                      "baseline_score": score, "final_score": score,
                      "rounds": [best.summary()]}

    original_ids = _ids(code)

    for round_index in range(1, max_rounds + 1):
        findings = diagnose(ctx, sample, best.code, best.render_path, params)
        if not findings:
            history.append(Attempt(round_index, best.code, best.score, True,
                                   notes="no findings; stopping"))
            break

        candidate = repair(ctx, sample, best.code, best.render_path, findings, params)
        if candidate is None:
            history.append(Attempt(round_index, best.code, None, True,
                                   notes="repair produced no code; stopping",
                                   findings=findings))
            break

        # A repair that drops element ids breaks every later stage that
        # references them, so it is rejected however well it renders.
        lost = original_ids - _ids(candidate)
        if lost:
            history.append(Attempt(round_index, candidate, None, True,
                                   notes=f"rejected: dropped xml id(s) {sorted(lost)[:5]}",
                                   findings=findings))
            break

        rendered = _render(candidate, cache_dir)
        if not rendered.ok or rendered.png_path is None:
            history.append(Attempt(round_index, candidate, None, False,
                                   notes="rejected: repair does not compile",
                                   findings=findings))
            break

        new_score, new_breakdown, new_notes = score_render(
            ctx, sample, rendered.png_path, params)
        attempt = Attempt(round_index, candidate, new_score, True, rendered.png_path,
                          new_breakdown, new_notes, findings)
        history.append(attempt)

        if new_score is None or new_score <= (best.score or 0.0):
            attempt.notes = (f"rejected: {new_score} did not improve on "
                             f"{best.score:.2f}") if new_score is not None else \
                            "rejected: could not score repair"
            break

        best = attempt
        if new_score >= threshold:
            break

    return best.code, {
        "baseline_score": history[0].score,
        "final_score": best.score,
        "rounds_run": len(history) - 1,
        "improved": bool(best.round_index > 0),
        # Non-empty when the code had to be repaired before it could render at
        # all; the fidelity scores below are then measured on the repaired
        # version, not on what stage 1a originally emitted.
        "compile_rounds": compile_history,
        "compile_repaired": bool(compile_history),
        "rounds": [a.summary() for a in history],
    }


def run(cfg, samples: list[PaperSample], force: bool = False) -> StageReport:
    ctx = AgentContext(cfg, AGENT)
    report = StageReport(agent=AGENT)
    threshold = float(ctx.agent.option("fidelity_threshold", DEFAULT_THRESHOLD))
    max_rounds = int(ctx.agent.option("max_rounds", DEFAULT_MAX_ROUNDS))

    try:
        for sample in samples:
            out = ctx.paths.code_reviewed(sample.id)
            if out.exists() and not force:
                report.outcomes.append(SampleOutcome(sample.id, "skipped", out, "cached"))
                continue
            try:
                # reviewed=False: read stage 1a/1b output, never a previous
                # run of this critic, so --force re-reviews from the same
                # starting point rather than compounding on itself.
                source = ctx.paths.resolve_code(sample.id, reviewed=False)
                code = Path(source).read_text(encoding="utf-8")
            except FileNotFoundError as e:
                report.outcomes.append(SampleOutcome(sample.id, "failed", None, str(e)))
                continue

            try:
                final, info = refine(ctx, sample, code, threshold, max_rounds)
            except Exception as e:
                report.outcomes.append(SampleOutcome(
                    sample.id, "failed", None, f"{type(e).__name__}: {e}"))
                continue

            write_text(out, final)
            import json
            write_text(out.parent / f"{sample.id}.critic.json",
                       json.dumps({**info, "provenance": ctx.provenance()},
                                  indent=2, default=str))

            if "skipped" in info:
                detail = info["skipped"]
                status = "ok"
            else:
                base, final_score = info.get("baseline_score"), info.get("final_score")
                detail = (f"fidelity {base:.2f} -> {final_score:.2f} "
                          f"in {info.get('rounds_run', 0)} round(s)"
                          if base is not None and final_score is not None
                          else "no usable score")
                # Below threshold after the full budget is a real outcome, not
                # a failure: the code is still the best version produced.
                status = ("ok" if (final_score or 0) >= threshold else "unresolved")
            report.outcomes.append(SampleOutcome(sample.id, status, out, detail))
    finally:
        ctx.unload()

    return report
