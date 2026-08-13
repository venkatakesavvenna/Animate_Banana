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
from ..prompts import has_placeholders, load_prompt, render
from ..runner import AgentContext, StageReport, run_agent
from ..samples import PaperSample
from ..schema import AnimationSequence
from ..styles import get_style

AGENT = "designer"


def build_request(ctx: AgentContext, sample: PaperSample) -> list[Message]:
    # Narrated first: 2e writes the spoken script onto an otherwise identical
    # structure, and that script carries pacing and emphasis the designer can
    # act on. Reading `sequence_final` directly meant narration was generated,
    # stored, and then never seen by the stage that turns a sequence into
    # motion -- on pipe00137, 15/15 narrated nodes reached the designer as 0.
    # Then the reviewed sequence, then the raw one so the animator can still
    # run before the critic stage has executed.
    for candidate in (ctx.paths.sequence_narrated(sample.id),
                      ctx.paths.sequence_final(sample.id),
                      ctx.paths.sequence(sample.id)):
        if candidate.exists():
            sequence_path = candidate
            break
    else:
        raise FileNotFoundError(
            f"no sequence for '{sample.id}'. Run the sequence (and critique) stages first."
        )

    sequence = AnimationSequence.load(sequence_path)
    code = Path(ctx.paths.resolve_code(sample.id)).read_text(encoding="utf-8")

    context = {
        "style_block": get_style(ctx.cfg.style).prompt_block(),
        "sequence_json": sequence.to_json(),
        "diagram_code": code,
    }

    template = load_prompt(ctx.agent.prompt)
    if has_placeholders(template):
        text = render(template, context)
    else:
        # `svg_designer.yaml` is written as standalone instructions with no
        # {placeholders} -- it names its inputs in prose ("You will be provided
        # with a JSON Animation Sequence and the Target Static SVG Code") and
        # expects them appended. Substituting into it silently supplied
        # *nothing*, so the model received the prompt and the figure but not
        # the document it was told to animate -- and duly redrew the diagram
        # from scratch, discarding every element id and every spliced raster.
        # Same convention, and the same fix, as `planner/sequencer.py`.
        text = "\n\n".join([
            template,
            "### TARGET STATIC SVG CODE\n" + code,
            "### JSON ANIMATION SEQUENCE\n" + context["sequence_json"],
            f"### ANIMATION STYLE\n{ctx.cfg.style}\n\n{context['style_block']}",
        ])

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
