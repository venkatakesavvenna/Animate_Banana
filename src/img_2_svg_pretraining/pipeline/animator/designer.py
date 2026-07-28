"""Stage 3a -- Animation Designer.

Rewrites the static diagram code as animation code realizing the validated
sequence in the requested style. For TikZ this means an `animateinline` /
`\\multiframe` wrapper with per-element opacity gated on the frame counter --
the idiom the hand-authored examples in `examples/animation/` use.
"""
from __future__ import annotations

from pathlib import Path

from ..backends import Message
from ..extract import extract_code
from ..prompts import load_and_render
from ..runner import AgentContext, StageReport, run_agent
from ..samples import PaperSample
from ..schema import AnimationSequence
from ..styles import get_style

AGENT = "designer"


def build_request(ctx: AgentContext, sample: PaperSample) -> list[Message]:
    sequence_path = ctx.paths.sequence_final(sample.id)
    if not sequence_path.exists():
        # Fall back to the un-reviewed sequence so the animator can run before
        # the critic stage has been executed.
        sequence_path = ctx.paths.sequence(sample.id)
    if not sequence_path.exists():
        raise FileNotFoundError(
            f"no sequence for '{sample.id}'. Run the sequence (and critique) stages first."
        )

    sequence = AnimationSequence.load(sequence_path)
    code = Path(ctx.paths.resolve_code(sample.id)).read_text(encoding="utf-8")

    text = load_and_render(ctx.agent.prompt, {
        "style_block": get_style(ctx.cfg.style).prompt_block(),
        "sequence_json": sequence.to_json(),
        "diagram_code": code,
    })
    return [Message.user(text, images=[sample.image_path])]


def parse_response(ctx: AgentContext, sample: PaperSample, raw: str) -> str:
    code = extract_code(raw, ctx.cfg.target)
    if code is None:
        raise ValueError(
            f"no {ctx.cfg.target} document in the response; "
            f"raw output kept at {ctx.paths.raw_output(AGENT, sample.id)}"
        )
    return code


def run(cfg, samples: list[PaperSample], force: bool = False) -> StageReport:
    ctx = AgentContext(cfg, AGENT)
    try:
        return run_agent(
            ctx=ctx,
            samples=samples,
            output_path=lambda s: ctx.paths.animation(s.id),
            build_request=lambda s: build_request(ctx, s),
            handle_response=lambda s, raw: parse_response(ctx, s, raw),
            force=force,
        )
    finally:
        ctx.unload()
