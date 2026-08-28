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


# The SVG designer prompt is keyed BY STYLE (`svg_designer.yaml#<key>`) where
# the TikZ one is not, and two of its keys do not match the pipeline's style
# names. Mapping them here rather than renaming the YAML keeps the prompt file
# byte-identical to the one the design docs specify.
SVG_DESIGNER_KEY = {
    "hopping_bounding_box": "hopping_box",
    "sliding_bounding_box": "sliding_box",
}


def _resolve_prompt(spec: str, style: str) -> str:
    """Expand a `{style}` token in a prompt spec to this run's style key.

    Lets ONE config serve every style -- `svg_designer.yaml#{style}` --
    instead of a near-duplicate config per style, which is how the style and
    the designer prompt drift apart. A spec without the token is returned
    unchanged, so every existing config keeps working.
    """
    if "{style}" not in spec:
        return spec
    return spec.replace("{style}", SVG_DESIGNER_KEY.get(style, style))


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

    sequence_json = sequence.to_json()
    context = {
        "style_block": get_style(ctx.cfg.style).prompt_block(),
        "sequence_json": sequence_json,
        "diagram_code": code,
        # BOTH SPELLINGS, deliberately. The prompt was rewritten to take
        # `{svg_code}` / `{json_sequence}` while this dict supplied
        # `diagram_code` / `sequence_json`. Nothing errored: `has_placeholders`
        # saw braces and took the substitution branch, `render` replaced only
        # the keys it recognised, and the model was handed the LITERAL text
        # "{svg_code}" where the diagram should have been -- so it redrew the
        # figure from scratch, losing every element id and every spliced
        # raster. A renamed placeholder must never be able to fail this
        # quietly, so both names resolve to the same value.
        "svg_code": code,
        "json_sequence": sequence_json,
    }

    template = load_prompt(_resolve_prompt(ctx.agent.prompt, ctx.cfg.style))
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
