"""Stage 2a -- Strategizer.

Decides the traversal strategy (OVERVIEW_FIRST or DETAIL_FIRST) and sketches
a high-level narrative outline, from the figure plus as much paper context as
the configured tier allows.

Independent of the parser (2b): this agent reads the figure and paper text,
not the structure XML, so the two can run in either order and re-running one
does not invalidate the other's cache.

The four context tiers map onto the four prompt variants in
`prompts/animation_planner/strategizer.yaml`. When the config selects a tier,
the matching variant is used automatically unless the prompt reference
already names one explicitly.
"""
from __future__ import annotations

import json

from ..backends import Message
from ..extract import extract_strategy
from ..prompts import load_prompt, parse_ref
from ..runner import AgentContext, StageReport, run_agent
from ..samples import PaperSample

AGENT = "strategizer"

TIER_PROMPT_KEY = {
    "image_only": "image-only-prompt",
    "caption": "image-plus-caption-prompt",
    "abstract": "image-plus-title-caption-abstract-prompt",
    "full": "full-context-prompt",
}

# Order matters: this is how the context is laid out for the model.
_FIELD_LABELS = [
    ("title", "PAPER TITLE"),
    ("caption", "FIGURE CAPTION"),
    ("abstract", "PAPER ABSTRACT"),
    ("methods", "METHOD SECTION"),
]


def _prompt_ref(ctx: AgentContext, tier: str) -> str:
    """Resolve the prompt variant for this tier.

    An explicit `#key` in the config wins, so a user can pin one variant
    while still labelling the run with a different tier.
    """
    file_part, key = parse_ref(ctx.agent.prompt)
    return f"{file_part}#{key or TIER_PROMPT_KEY[tier]}"


def _context_block(sample: PaperSample, tier: str) -> str:
    ctx_fields = sample.context_for(tier)
    parts = [f"{label}:\n{ctx_fields[field]}"
             for field, label in _FIELD_LABELS if ctx_fields.get(field)]
    return "\n\n".join(parts)


def build_request(ctx: AgentContext, sample: PaperSample) -> list[Message]:
    tier = ctx.agent.option("context_tier", "full")
    if not sample.supports_tier(tier):
        raise ValueError(
            f"tier '{tier}' needs {sample.missing_for_tier(tier)}, which this sample lacks"
        )

    prompt = load_prompt(_prompt_ref(ctx, tier))
    block = _context_block(sample, tier)
    text = f"{prompt}\n\n{block}" if block else prompt
    return [Message.user(text, images=[sample.image_path])]


def parse_response(ctx: AgentContext, sample: PaperSample, raw: str) -> str:
    strategy = extract_strategy(raw)
    if strategy is None:
        raise ValueError(
            "response has no TRAVERSAL_STYLE line; "
            f"raw output kept at {ctx.paths.raw_output(AGENT, sample.id)}"
        )
    if not strategy["sequence"]:
        raise ValueError("response has no numbered SEQUENCE steps")

    strategy["provenance"] = ctx.provenance(
        context_tier=ctx.agent.option("context_tier", "full"))
    return json.dumps(strategy, indent=2)


def run(cfg, samples: list[PaperSample], force: bool = False) -> StageReport:
    ctx = AgentContext(cfg, AGENT)
    try:
        return run_agent(
            ctx=ctx,
            samples=samples,
            output_path=lambda s: ctx.paths.strategy(s.id),
            build_request=lambda s: build_request(ctx, s),
            handle_response=lambda s, raw: parse_response(ctx, s, raw),
            force=force,
        )
    finally:
        ctx.unload()
