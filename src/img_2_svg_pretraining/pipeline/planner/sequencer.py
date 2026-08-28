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
from ..extract import extract_json, looks_truncated
from ..prompts import has_placeholders, load_prompt, render
from ..runner import AgentContext, StageReport, run_agent
from ..samples import PaperSample
from ..schema import AnimationSequence
from ..styles import get_style
from .parser import referenceable_ids

AGENT = "sequencer"
DEFAULT_MAX_DEPTH = 3


def _load_strategy(ctx: AgentContext, sample: PaperSample) -> dict:
    """The strategizer's plan, or an empty one when 2a did not run.

    Optional rather than required: the bench sequencer prompts decide the
    traversal order themselves ("You must determine the optimal Traversal
    Style"), so a strategy is only useful to the placeholder-style prompt that
    interpolates it. Missing strategy is therefore a configuration choice, not
    an error.
    """
    path = ctx.paths.strategy(sample.id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_xml(ctx: AgentContext, sample: PaperSample) -> str:
    path = ctx.paths.xml(sample.id)
    if not path.exists():
        raise FileNotFoundError(
            f"no structure XML for '{sample.id}' at {path}. Run the parse stage first."
        )
    return path.read_text(encoding="utf-8")


def _paper_context(ctx: AgentContext, sample: PaperSample) -> str:
    """The paper's own words, when this sequencer is configured to want them.

    OFF by default, and deliberately so. The bench prompts name exactly three
    inputs (code, structural graph, style) and decide traversal from them;
    adding the abstract and methods gives the model prose that can disagree
    with the figure, which is a different experiment rather than a strictly
    better one. It is a config option so that experiment can be run:

        planner:
          sequencer:
            context_tier: full      # image_only | caption | abstract | full

    `context_tier` also lands in the sequence lineage when set, so a run with
    context and one without never share a cache entry.

    Absent context is a configuration choice, not an error: a sample missing
    an abstract simply contributes less, rather than failing the stage.
    """
    tier = ctx.agent.option("context_tier", None)
    if not tier or tier == "image_only":
        return ""
    if not sample.supports_tier(tier):
        return ""
    fields = sample.context_for(tier)
    blocks = [f"### {k.upper()}\n{v}" for k, v in fields.items() if v]
    if not blocks:
        return ""
    return ("### PAPER CONTEXT (background only -- the figure and the graph "
            "above remain authoritative)\n\n" + "\n\n".join(blocks))


def build_request(ctx: AgentContext, sample: PaperSample) -> list[Message]:
    strategy = _load_strategy(ctx, sample)
    structure_xml = _load_xml(ctx, sample)
    style = get_style(ctx.cfg.style)
    max_depth = ctx.agent.option("max_depth", DEFAULT_MAX_DEPTH)

    outline = "\n".join(f"{i}. {step}"
                        for i, step in enumerate(strategy.get("sequence", []), 1))

    context = {
        "max_depth": str(max_depth),
        "style_block": style.prompt_block(),
        "traversal_style": strategy.get("traversal_style", "OVERVIEW_FIRST"),
        "strategy_reasoning": strategy.get("reasoning", ""),
        "strategy_sequence": outline,
        "structure_xml": structure_xml,
    }

    template = load_prompt(ctx.agent.prompt)
    if has_placeholders(template):
        # This form interpolates the strategizer's plan, so it genuinely needs
        # one. Say so rather than rendering an empty outline into the prompt.
        if not strategy:
            raise FileNotFoundError(
                f"no strategy for '{sample.id}' at {ctx.paths.strategy(sample.id)}. "
                f"'{ctx.agent.prompt}' interpolates the strategizer's plan, so "
                f"either run the strategize stage or point the sequencer at a "
                f"bench prompt (tikz_sequencer / svg_sequencer), which plans "
                f"the traversal itself."
            )
        text = render(template, context)
    else:
        # The AnimateBench prompts (tikz_sequencer / svg_sequencer) are written
        # as standalone instructions with no {placeholders} -- they name their
        # inputs in prose and expect them appended. Substitution would silently
        # supply nothing, so the same context is laid out as a labelled suffix
        # instead. They also read the diagram code directly, since text_node
        # elements are deliberately absent from the structure XML.
        #
        # Exactly the three inputs the prompt's own "YOUR INPUTS" section names
        # -- code, structural graph, style. Deliberately no traversal hint: the
        # prompt states "You must determine the optimal Traversal Style", so
        # supplying one would work against the instruction it is being asked
        # to follow.
        code = Path(ctx.paths.resolve_code(sample.id)).read_text(encoding="utf-8")
        parts = [
            template,
            "### ORIGINAL DIAGRAM CODE\n" + code,
            "### HIERARCHICAL GRAPH (STRUCTURE XML)\n" + structure_xml,
            f"### ANIMATION STYLE\n{ctx.cfg.style}\n\n{style.prompt_block()}",
        ]
        paper = _paper_context(ctx, sample)
        if paper:
            parts.append(paper)
        text = "\n\n".join(parts)

    return [Message.user(text, images=[sample.image_path])]


def parse_response(ctx: AgentContext, sample: PaperSample, raw: str) -> str:
    data = extract_json(raw)
    if data is None:
        reason = ("response was truncated mid-JSON -- raise this agent's "
                  "params.max_tokens" if looks_truncated(raw)
                  else "no JSON object in the response")
        raise ValueError(f"{reason}; raw output kept at {ctx.paths.raw_output(AGENT, sample.id)}")

    data.setdefault("style", ctx.cfg.style)
    sequence = AnimationSequence.from_dict(data)
    sequence.provenance = ctx.provenance(
        max_depth=ctx.agent.option("max_depth", DEFAULT_MAX_DEPTH))

    # Record violations rather than rejecting: repairing them is exactly the
    # critic's job, and a sequence that fails validation is still the input
    # that stage needs. A response we cannot parse at all is different -- that
    # raises above.
    # XML *and* code: `text_node` elements are deliberately absent from the
    # XML, so checking against it alone flags every correctly-referenced text
    # label as missing.
    element_ids = referenceable_ids(
        ctx.paths.xml(sample.id),
        Path(ctx.paths.resolve_code(sample.id)).read_text(encoding="utf-8"),
    )
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
