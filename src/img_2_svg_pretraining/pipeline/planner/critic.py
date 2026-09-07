"""Stage 2d -- Sequence Critic.

Reviews and repairs the sequencer's output over a bounded number of rounds.
Two complementary signals drive it:

1. **Programmatic checks** run first and are cheap and certain -- dangling
   focus ids, cycles, depth violations, uncovered nodes, style breaches. They
   are handed to the model as findings rather than asking it to rediscover
   them, and they also decide when the loop can stop early.

2. **Set-of-Mark grounding** for what automation cannot judge: whether a
   step's focus is actually the region its narration describes, whether the
   order reads coherently, whether anything important is missing.

The marking reuses the same trick as `data_engine/edges.py`: numbered markers
are drawn on the figure and the model sees only integers, so it never has to
copy long ids and parsing maps numbers back to ids afterwards. Geometry comes
from the TikZ source (which is pixel-mapped by the converter prompt), because
the structure XML deliberately discards coordinates.

Every round's sequence is persisted so the refinement history is inspectable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..backends import Message
from ..cache import write_text
from ..extract import extract_json, looks_truncated
from ..ablation import drop_declared_inputs
from ..prompts import load_and_render
from ..runner import AgentContext, StageReport, run_iterative_agent
from ..samples import PaperSample
from ..schema import AnimationSequence
from ..styles import get_style
from .parser import referenceable_ids
from .sequencer import DEFAULT_MAX_DEPTH

AGENT = "planner_critic"

# \node[... xml id=foo ...] (foo) at (123, 456) {...}
_NODE_AT_RE = re.compile(
    r"xml\s+id\s*=\s*(?P<id>[\w:.-]+)[^;]*?\bat\s*\(\s*(?P<x>-?[\d.]+)\s*,\s*(?P<y>-?[\d.]+)\s*\)",
    re.DOTALL,
)
# Fallback: any element declaring an xml id, order-preserved.
_ANY_ID_RE = re.compile(r"xml\s+id\s*=\s*(?P<id>[\w:.-]+)")


def _element_positions(code: str) -> dict[str, tuple[float, float]]:
    """Pixel positions of elements that declare an explicit `at (x, y)`.

    The converter prompt mandates `x=1pt, y=-1pt` with a top-left origin, so
    TikZ coordinates are already image pixels. Elements positioned relatively
    (`below=of foo`) or by `fit` have no literal coordinate and are omitted --
    they simply do not get a marker.
    """
    positions: dict[str, tuple[float, float]] = {}
    for match in _NODE_AT_RE.finditer(code):
        try:
            positions[match.group("id")] = (float(match.group("x")), abs(float(match.group("y"))))
        except ValueError:
            continue
    return positions


def _mark_figure(sample: PaperSample, code: str, sequence: AnimationSequence,
                 out_path: Path) -> tuple[Path | None, str]:
    """Draw numbered markers on the focus elements, return (image, legend).

    Marks are numbered by traversal order so the numbering itself carries the
    proposed sequence -- the model can see at a glance whether the order zigzags
    across the figure.
    """
    from PIL import Image, ImageDraw, ImageFont

    positions = _element_positions(code)
    if not positions:
        return None, "(no element coordinates available in the diagram code)"

    image = Image.open(sample.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()

    legend_lines: list[str] = []
    mark_id = 1
    by_id = {n.id: n for n in sequence.nodes}
    for node_id in sequence.traversal:
        node = by_id.get(node_id)
        if node is None:
            continue
        for element in node.focus:
            pos = positions.get(element)
            if pos is None:
                legend_lines.append(
                    f"  (unmarked) step '{node_id}' -> element '{element}' (no coordinate)")
                continue
            cx, cy = pos
            radius = 13
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                         fill="yellow", outline="black", width=2)
            draw.text((cx, cy), str(mark_id), fill="black", font=font, anchor="mm")
            legend_lines.append(f"  {mark_id} -> element '{element}' (step '{node_id}')")
            mark_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path, "\n".join(legend_lines) if legend_lines else "(no elements marked)"


class _Critic:
    """Per-run state shared across the rounds of every sample."""

    def __init__(self, cfg):
        self.ctx = AgentContext(cfg, AGENT)
        self.cfg = cfg
        self.max_depth = self.ctx.agent.option("max_depth", DEFAULT_MAX_DEPTH)
        self.use_som = self.ctx.agent.option("use_set_of_mark", True)

    def _load_current(self, sample: PaperSample, round_index: int) -> AnimationSequence:
        """Round 0 reads the sequencer's output; later rounds read the
        previous round's repair."""
        path = (self.ctx.paths.sequence(sample.id) if round_index == 0
                else self.ctx.paths.sequence(sample.id, round_index - 1))
        if not path.exists():
            raise FileNotFoundError(
                f"no sequence for '{sample.id}' at {path}. Run the sequence stage first."
            )
        return AnimationSequence.load(path)

    def step(self, sample: PaperSample, round_index: int) -> tuple[bool, str, str]:
        sequence = self._load_current(sample, round_index)
        element_ids = referenceable_ids(
            self.ctx.paths.xml(sample.id),
            Path(self.ctx.paths.resolve_code(sample.id)).read_text(encoding="utf-8"),
        )
        findings = sequence.validate(element_ids=element_ids, max_depth=self.max_depth)

        # A clean sequence still gets one review pass, because the mechanical
        # checks cannot see grounding, ordering or completeness problems. Only
        # a clean sequence that has already been reviewed is done.
        if not findings and round_index > 0:
            return True, sequence.to_json(), f"converged after {round_index} round(s)"

        code = Path(self.ctx.paths.resolve_code(sample.id)).read_text(encoding="utf-8")

        # ABLATION FLAG. Withholding the image also disables set-of-mark:
        # the markers are drawn ON the image, and the prompt's whole first
        # input is "the diagram image. Numbered markers have been overlaid on
        # it". Sending a legend for markers the model cannot see would be
        # worse than sending nothing.
        send_image = bool(self.ctx.agent.option("send_image", True))
        images = [sample.image_path] if send_image else None
        legend = "(set-of-mark disabled)"
        if self.use_som and send_image:
            marked_path = (self.ctx.paths.root / "marked" /
                           self.ctx.paths.sequence_lineage /
                           f"{sample.id}.round{round_index}.png")
            marked, legend = _mark_figure(sample, code, sequence, marked_path)
            if marked is not None:
                images = [marked]

        text = load_and_render(self.ctx.agent.prompt, {
            "max_depth": str(self.max_depth),
            "style_block": get_style(self.cfg.style).prompt_block(),
            "marker_legend": legend,
            "findings": ("\n".join(f"- {f}" for f in findings)
                         if findings else "(none detected mechanically)"),
            "structure_xml": self.ctx.paths.xml(sample.id).read_text(encoding="utf-8"),
            "sequence_json": sequence.to_json(),
        })

        if not send_image:
            text = drop_declared_inputs(text, {"image"})
        result = self.ctx.backend.generate([Message.user(text, images=images)],
                                           **self.ctx.params())
        if not result.ok:
            raise RuntimeError(f"critic call failed: {result.error}")

        data = extract_json(result.text)
        if data is None:
            raise ValueError(
                "critic response was truncated mid-JSON -- raise this agent's "
                "params.max_tokens" if looks_truncated(result.text)
                else "critic response contained no JSON object")

        review = data.pop("review", "") if isinstance(data, dict) else ""
        data.setdefault("style", self.cfg.style)
        revised = AnimationSequence.from_dict(data)
        revised.provenance = {
            **sequence.provenance,
            "critic": self.ctx.provenance(round=round_index, review=review,
                                          findings_addressed=findings),
        }

        write_text(self.ctx.paths.sequence(sample.id, round_index), revised.to_json())

        remaining = revised.validate(element_ids=element_ids, max_depth=self.max_depth)
        detail = f"round {round_index}: {len(findings)} -> {len(remaining)} violation(s)"
        if remaining:
            revised.provenance["critic"]["remaining"] = remaining
        return (not remaining), revised.to_json(), detail


def run(cfg, samples: list[PaperSample], force: bool = False) -> StageReport:
    critic = _Critic(cfg)
    try:
        return run_iterative_agent(
            ctx=critic.ctx,
            samples=samples,
            output_path=lambda s: critic.ctx.paths.sequence_final(s.id),
            step=critic.step,
            max_rounds=int(critic.ctx.agent.option("max_rounds", 2)),
            force=force,
        )
    finally:
        critic.ctx.unload()
