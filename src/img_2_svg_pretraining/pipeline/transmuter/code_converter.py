"""Stage 1a -- Diagram Code Converter.

Turns a diagram image into animation-aware code (TikZ or SVG). "Animation
aware" means every drawn element carries structural metadata -- `xml id`,
`xml class`, `xml parent`, `xml source`/`xml target` -- declared as no-op
TikZ keys so they survive compilation while remaining machine-parseable.

Those ids are what the whole downstream pipeline addresses: the parser turns
them into the structure XML, the sequencer's `focus` lists reference them,
and the designer animates the elements they name.
"""
from __future__ import annotations

from pathlib import Path

from ..backends import Message
from ..extract import extract_code
from ..prompts import load_prompt
from ..runner import AgentContext, StageReport, run_agent
from ..samples import PaperSample

AGENT = "code_converter"


def build_request(ctx: AgentContext, sample: PaperSample) -> list[Message]:
    prompt = load_prompt(ctx.agent.prompt)
    return [Message.user(prompt, images=[sample.image_path])]


def parse_response(ctx: AgentContext, sample: PaperSample, raw: str) -> str:
    code = extract_code(raw, ctx.cfg.target)
    if code is None:
        marker = "\\documentclass" if ctx.cfg.target == "tikz" else "<svg>"
        raise ValueError(
            f"no {ctx.cfg.target} document in the response (no {marker} found); "
            f"raw output kept at {ctx.paths.raw_output(AGENT, sample.id)}"
        )
    return code


def run(cfg, samples: list[PaperSample], force: bool = False) -> StageReport:
    ctx = AgentContext(cfg, AGENT)
    try:
        return run_agent(
            ctx=ctx,
            samples=samples,
            output_path=lambda s: ctx.paths.code(s.id),
            build_request=lambda s: build_request(ctx, s),
            handle_response=lambda s, raw: parse_response(ctx, s, raw),
            force=force,
        )
    finally:
        ctx.unload()
