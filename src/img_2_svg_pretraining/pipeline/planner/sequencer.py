"""Stage 2c -- Sequencer.

Consumes the strategizer's traversal strategy (2a) and the parser's structure
XML (2b) -- the point where the two independent Stage-2 lineages merge -- and
produces a concrete animation sequence: a node hierarchy plus an explicit
traversal order, conditioned on the requested animation style.

The style is an input here, not at animation time, because it changes what a
good plan looks like: progressive reveal wants a single build-up pass, while
highlight-and-dim can revisit freely.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..backends import Message
from ..extract import extract_json
from ..prompts import load_and_render
from ..runner import AgentContext, StageReport, run_agent
from ..samples import PaperSample
from ..schema import AnimationSequence
from ..styles import get_style
from .parser import load_element_ids

AGENT = "sequencer"
DEFAULT_MAX_DEPTH = 3


def _load_strategy(ctx: AgentContext, sample: PaperSample) -> dict:
    path = ctx.paths.strategy(sample.id)
    if not path.exists():
        raise FileNotFoundError(
            f"no strategy for '{sample.id}' at {path}. Run the strategize stage first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_xml(ctx: AgentContext, sample: PaperSample) -> str:
    path = ctx.paths.xml(sample.id)
    if not path.exists():
        raise FileNotFoundError(
            f"no structure XML for '{sample.id}' at {path}. Run the parse stage first."
        )
    return path.read_text(encoding="utf-8")


def build_request(ctx: AgentContext, sample: PaperSample) -> list[Message]:
    strategy = _load_strategy(ctx, sample)
    structure_xml = _load_xml(ctx, sample)
    style = get_style(ctx.cfg.style)
    max_depth = ctx.agent.option("max_depth", DEFAULT_MAX_DEPTH)

    outline = "\n".join(f"{i}. {step}"
                        for i, step in enumerate(strategy.get("sequence", []), 1))

    text = load_and_render(ctx.agent.prompt, {
        "max_depth": str(max_depth),
        "style_block": style.prompt_block(),
        "traversal_style": strategy.get("traversal_style", "OVERVIEW_FIRST"),
        "strategy_reasoning": strategy.get("reasoning", ""),
        "strategy_sequence": outline,
        "structure_xml": structure_xml,
    })
    return [Message.user(text, images=[sample.image_path])]


def parse_response(ctx: AgentContext, sample: PaperSample, raw: str) -> str:
    data = extract_json(raw)
    if data is None:
        raise ValueError(
            "no JSON object in the response; "
            f"raw output kept at {ctx.paths.raw_output(AGENT, sample.id)}"
        )

    data.setdefault("style", ctx.cfg.style)
    sequence = AnimationSequence.from_dict(data)
    sequence.provenance = ctx.provenance(
        max_depth=ctx.agent.option("max_depth", DEFAULT_MAX_DEPTH))

    # Record violations rather than rejecting: repairing them is exactly the
    # critic's job, and a sequence that fails validation is still the input
    # that stage needs. A response we cannot parse at all is different -- that
    # raises above.
    element_ids = load_element_ids(ctx.paths.xml(sample.id))
    problems = sequence.validate(
        element_ids=element_ids,
        max_depth=ctx.agent.option("max_depth", DEFAULT_MAX_DEPTH),
    )
    if problems:
        sequence.provenance["validation_at_generation"] = problems
        print(f"    [{sample.id}] {len(problems)} violation(s) for the critic to repair")

    return sequence.to_json()


def run(cfg, samples: list[PaperSample], force: bool = False) -> StageReport:
    ctx = AgentContext(cfg, AGENT)
    try:
        return run_agent(
            ctx=ctx,
            samples=samples,
            output_path=lambda s: ctx.paths.sequence(s.id),
            build_request=lambda s: build_request(ctx, s),
            handle_response=lambda s, raw: parse_response(ctx, s, raw),
            force=force,
        )
    finally:
        ctx.unload()
