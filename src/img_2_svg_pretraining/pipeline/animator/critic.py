"""Stage 3c -- Animation Critic.

Reviews generated animation code over a bounded number of rounds. Its
strongest signal is not a model judgment at all: does the code compile? A
latexmk log names the failing line and cause, which is far better correction
material than a model's guess, so the log is fed straight back to the
designer prompt.

When it does compile, the first rendered frame is shown for visual review --
canvas pinning, transparency groups, frame gating.

Compilation goes through `viewer/compile.py::compile_tikz`, the same renderer
the annotation tooling uses, so a "compiles" verdict means the same thing
everywhere in this project. Note that helper passes `-singlefile` to
pdftoppm, so it rasterizes page 1 only -- correct for this check, but the
exporter needs its own call to extract every frame.
"""
from __future__ import annotations

from pathlib import Path

from ..backends import Message
from ..cache import write_text
from ..extract import extract_code
from ..prompts import load_and_render
from ..runner import AgentContext, StageReport, run_iterative_agent
from ..samples import PaperSample
from ..schema import AnimationSequence

AGENT = "animator_critic"

MAX_LOG_CHARS = 6000


def _trim_log(log: str) -> str:
    """Keep the informative part of a latexmk log.

    The first real error matters most, but the tail carries the summary, so
    both ends are kept when the log is long.
    """
    log = log.strip()
    if len(log) <= MAX_LOG_CHARS:
        return log
    head = log[: MAX_LOG_CHARS // 2]
    tail = log[-MAX_LOG_CHARS // 2:]
    return f"{head}\n\n... [log truncated] ...\n\n{tail}"


def compile_animation(code: str, cache_dir: Path):
    """Compile and rasterize the first frame. Returns a CompileResult."""
    from img_2_svg_pretraining.viewer.compile import compile_tikz
    cache_dir.mkdir(parents=True, exist_ok=True)
    return compile_tikz(code, cache_dir)


class _Critic:
    def __init__(self, cfg):
        self.ctx = AgentContext(cfg, AGENT)
        self.cfg = cfg
        self.compile_check = self.ctx.agent.option("compile_check", True)

    def _load_current(self, sample: PaperSample, round_index: int) -> str:
        path = (self.ctx.paths.animation(sample.id) if round_index == 0
                else self.ctx.paths.animation(sample.id, round_index - 1))
        if not path.exists():
            raise FileNotFoundError(
                f"no animation code for '{sample.id}' at {path}. Run the design stage first."
            )
        return path.read_text(encoding="utf-8")

    def _sequence_json(self, sample: PaperSample) -> str:
        for path in (self.ctx.paths.sequence_final(sample.id),
                     self.ctx.paths.sequence(sample.id)):
            if path.exists():
                return AnimationSequence.load(path).to_json()
        return "(sequence unavailable)"

    def step(self, sample: PaperSample, round_index: int) -> tuple[bool, str, str]:
        code = self._load_current(sample, round_index)

        compiled_ok, log, png_path = True, "", None
        if self.compile_check and self.cfg.target == "tikz":
            result = compile_animation(code, self.ctx.paths.compile_cache())
            compiled_ok, log, png_path = result.ok, result.log, result.png_path

        # Compiles and has already been reviewed once: done.
        if compiled_ok and round_index > 0:
            return True, code, f"compiles after {round_index} repair round(s)"

        images = [sample.image_path]
        if compiled_ok and png_path is not None:
            images = [png_path, sample.image_path]

        text = load_and_render(self.ctx.agent.prompt, {
            "compile_status": "SUCCESS" if compiled_ok else "FAILED",
            "compile_log": _trim_log(log) if log else "(no compiler output)",
            "sequence_json": self._sequence_json(sample),
            "animation_code": code,
        })

        result = self.ctx.backend.generate([Message.user(text, images=images)],
                                           **self.ctx.params())
        if not result.ok:
            raise RuntimeError(f"critic call failed: {result.error}")

        revised = extract_code(result.text, self.cfg.target)
        if revised is None:
            raise ValueError(f"critic response contained no {self.cfg.target} document")

        write_text(self.ctx.paths.animation(sample.id, round_index), revised)

        now_ok = True
        if self.compile_check and self.cfg.target == "tikz":
            now_ok = compile_animation(revised, self.ctx.paths.compile_cache()).ok

        before = "compiled" if compiled_ok else "failed"
        after = "compiles" if now_ok else "still fails"
        return now_ok, revised, f"round {round_index}: {before} -> {after}"


def run(cfg, samples: list[PaperSample], force: bool = False) -> StageReport:
    critic = _Critic(cfg)
    try:
        return run_iterative_agent(
            ctx=critic.ctx,
            samples=samples,
            output_path=lambda s: critic.ctx.paths.animation_final(s.id),
            step=critic.step,
            max_rounds=int(critic.ctx.agent.option("max_rounds", 3)),
            force=force,
        )
    finally:
        critic.ctx.unload()
