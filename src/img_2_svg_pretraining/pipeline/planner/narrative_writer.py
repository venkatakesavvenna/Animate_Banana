"""Stage 2e -- Narrative Writer.

Takes the finished animation sequence and writes the spoken narration for it,
one script segment per playback step. This is the last planner stage: the
structure is already settled by the sequencer (2c) and its critic (2d), and
this agent only fills in what a presenter would *say* at each moment.

Separating narration from sequencing is what makes the split worth having.
The sequencer reasons about the diagram's topology and has to satisfy hard
structural constraints; asking it to simultaneously write good spoken prose
grounded in the paper's method section made both jobs worse. Here the
structure is an input, not an output.

Which is also the failure mode to defend against. A model handed a JSON
document and asked to edit one field will occasionally "improve" the rest --
drop a step it finds redundant, renumber ids, reorder the traversal. The
prompt forbids that, but the prompt is not the enforcement: `_graft` copies
*only* narrative text back onto the sequence that came in, keyed by node id.
Structural edits in the response are therefore not rejected, they are simply
inert. Anything the model returns for an id we did not send is reported, so a
prompt that has started drifting is visible rather than silently ignored.

Context tier works as it does for the strategizer: `full` grounds the script
in the paper's method section, `image_only` deliberately withholds everything
but the figure, which is the honest setting for benchmarking a system that
will not have the paper at inference time.
"""
from __future__ import annotations

from pathlib import Path

from ..backends import Message
from ..extract import extract_json, looks_truncated
from ..ablation import drop_declared_inputs
from ..prompts import load_and_render
from ..runner import AgentContext, StageReport, run_agent
from ..samples import PaperSample
from ..schema import AnimationSequence

AGENT = "narrative_writer"
DEFAULT_TIER = "full"

# Prompt variant per context tier. The two middle tiers reuse the full-context
# template: it renders whatever fields the sample has and the absent ones
# resolve to "(not available)", so a caption-only sample simply gets a shorter
# prompt rather than needing a template of its own.
_PROMPT_KEYS = {
    "image_only": "image-only-prompt",
    "caption": "full-context-prompt",
    "abstract": "full-context-prompt",
    "full": "full-context-prompt",
}

_MISSING = "(not available)"


def _tier_for(ctx: AgentContext, sample: PaperSample) -> str:
    """The requested tier, degraded to the best one this sample supports.

    A sample missing its method section must not fail the stage -- narration
    from the caption alone is worse but still usable, and refusing would mean
    losing the whole run over an incomplete paper directory.
    """
    requested = ctx.agent.option("context_tier", DEFAULT_TIER)
    if sample.supports_tier(requested):
        return requested
    available = sample.available_tiers()
    return available[-1] if available else "image_only"


def _load_sequence(ctx: AgentContext, sample: PaperSample) -> AnimationSequence:
    """The sequence to narrate: the critic's repair when there is one."""
    path = ctx.paths.sequence_final(sample.id)
    if not path.exists():
        path = ctx.paths.sequence(sample.id)
    if not path.exists():
        raise FileNotFoundError(
            f"no sequence for '{sample.id}'. Run the sequence (and critique) stages first."
        )
    return AnimationSequence.load(path)


def build_request(ctx: AgentContext, sample: PaperSample) -> list[Message]:
    sequence = _load_sequence(ctx, sample)
    tier = _tier_for(ctx, sample)

    prompt_ref = ctx.agent.prompt
    # The config names the file; the tier picks the variant, so one config
    # entry covers all four tiers without four prompt lines to keep in sync.
    if "#" not in prompt_ref:
        prompt_ref = f"{prompt_ref}#{_PROMPT_KEYS[tier]}"

    context = {"sequence_json": sequence.to_json()}
    if tier != "image_only":
        available = sample.context_for(tier)
        context.update({
            "title": available.get("title") or sample.title or _MISSING,
            "caption": available.get("caption") or _MISSING,
            "abstract": available.get("abstract") or _MISSING,
            "methods": available.get("methods") or _MISSING,
        })

    # ABLATION FLAG. `context_tier: image_only` already handles the
    # without-context ablation by selecting `image-only-prompt`; this handles
    # the orthogonal one, withholding the diagram image itself.
    send_image = bool(ctx.agent.option("send_image", True))
    text = load_and_render(prompt_ref, context)
    if not send_image:
        text = drop_declared_inputs(text, {"image"})
    return [Message.user(text,
                        images=[sample.image_path] if send_image else None)]


def _graft(original: AnimationSequence, data: dict) -> tuple[AnimationSequence, list[str]]:
    """Copy narrative text from the response onto the original sequence.

    Returns the narrated sequence and a list of notes about anything the
    response got wrong. The original object is what gets written out, so no
    structural change in the response can survive -- only the text moves.
    """
    returned: dict[str, str | None] = {}
    for raw in data.get("nodes") or []:
        if isinstance(raw, dict) and "id" in raw:
            returned[str(raw["id"])] = raw.get("narrative") or raw.get("narration")

    notes: list[str] = []
    known = {n.id for n in original.nodes}

    unknown = sorted(set(returned) - known)
    if unknown:
        notes.append(f"response invented {len(unknown)} unknown node id(s): {unknown[:5]}")

    narrated = 0
    for node in original.nodes:
        text = returned.get(node.id)
        if isinstance(text, str) and text.strip():
            node.narrative = text.strip()
            narrated += 1
        else:
            node.narrative = None

    missing = len(known) - len(known & set(returned))
    if missing:
        notes.append(f"{missing} node(s) absent from the response")
    if narrated == 0:
        notes.append("no narration produced for any step")

    return original, notes


def parse_response(ctx: AgentContext, sample: PaperSample, raw: str) -> str:
    data = extract_json(raw)
    if not isinstance(data, dict):
        reason = ("response was truncated mid-JSON -- raise this agent's "
                  "params.max_tokens" if looks_truncated(raw)
                  else "no JSON object in the response")
        raise ValueError(
            f"{reason}; raw output kept at {ctx.paths.raw_output(AGENT, sample.id)}")

    sequence = _load_sequence(ctx, sample)
    narrated, notes = _graft(sequence, data)

    tier = _tier_for(ctx, sample)
    narrated.provenance = {
        **sequence.provenance,
        "narrative": ctx.provenance(
            context_tier=tier,
            narrated_steps=sum(1 for n in narrated.nodes if n.narrative),
            total_steps=len(narrated.nodes),
            notes=notes,
        ),
    }
    if notes:
        print(f"    [{sample.id}] {'; '.join(notes)}")

    if not narrated.has_narrative():
        raise ValueError(
            "response contained no usable narration; raw output kept at "
            f"{ctx.paths.raw_output(AGENT, sample.id)}")

    # The TTS stage consumes JSONL, so write it alongside rather than making
    # every downstream consumer re-derive it from the sequence.
    from ..cache import write_text
    write_text(ctx.paths.narration_jsonl(sample.id), narrated.narration_jsonl())

    return narrated.to_json()


def run(cfg, samples: list[PaperSample], force: bool = False) -> StageReport:
    ctx = AgentContext(cfg, AGENT)
    try:
        return run_agent(
            ctx=ctx,
            samples=samples,
            output_path=lambda s: ctx.paths.sequence_narrated(s.id),
            build_request=lambda s: build_request(ctx, s),
            handle_response=lambda s, raw: parse_response(ctx, s, raw),
            force=force,
        )
    finally:
        ctx.unload()
