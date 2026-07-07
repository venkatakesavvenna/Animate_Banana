"""Edge discovery via Set-of-Mark prompting.

Draws a numbered marker at each detected node's bbox center onto a copy of
the source image, then asks a chat VLM (from the existing
`benchmark.models` registry -- same model family/loading pattern as
`benchmark/infer.py::ModelRunner` and `benchmark/metrics/judge.py::JudgeRunner`,
reused here rather than duplicated) which numbered nodes are connected, and
in what direction. The model only ever sees integer marks, never internal
node ids, so parsing maps mark numbers back to ids after generation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor

from img_2_svg_pretraining.benchmark.models import ModelSpec
from img_2_svg_pretraining.data_engine.schema import Arrow, Diagram

EDGE_PROMPT = """You are given a technical diagram with numbered markers placed on \
each labeled node/box. The legend below maps each marker number to that node's label.

LEGEND:
{legend}

Identify every directed connection (arrow) between these numbered nodes in the \
diagram, following the arrows/lines as drawn. Respond ONLY with a JSON list of \
objects: [{{"from": <marker number>, "to": <marker number>}}, ...]. If there are no \
connections, respond with an empty JSON list: []
"""


@dataclass
class MarkedNode:
    mark_id: int
    node_id: str
    cx: float
    cy: float


def _node_center(bbox) -> tuple[float, float]:
    return ((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)


def draw_marks(image: Image.Image, diagram: Diagram) -> tuple[Image.Image, list[MarkedNode]]:
    """Overlay a numbered circle at the center of every node/block bbox.
    Blocks are marked too (edges can point at a container, not just a leaf
    node) but nodes nested inside an already-marked block are still marked
    independently -- the model decides which level an arrow targets."""
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()

    marks: list[MarkedNode] = []
    mark_id = 0
    all_items = list(diagram.blocks) + diagram.all_nodes()
    for item in all_items:
        cx, cy = _node_center(item.bbox)
        marks.append(MarkedNode(mark_id=mark_id, node_id=item.id, cx=cx, cy=cy))
        radius = 12
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                      fill="yellow", outline="black", width=2)
        draw.text((cx, cy), str(mark_id), fill="black", font=font, anchor="mm")
        mark_id += 1
    return marked, marks


def _build_legend(marks: list[MarkedNode], diagram: Diagram) -> str:
    lines = []
    for mark in marks:
        item = diagram.find_id(mark.node_id)
        label = item.text if item is not None else mark.node_id
        lines.append(f"{mark.mark_id}: {label}")
    return "\n".join(lines)


def _parse_edges(raw_output: str, marks: list[MarkedNode]) -> list[tuple[str, str]]:
    by_mark = {m.mark_id: m.node_id for m in marks}
    match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if not match:
        return []
    try:
        pairs = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    edges = []
    for pair in pairs:
        try:
            src_mark, dst_mark = int(pair["from"]), int(pair["to"])
        except (KeyError, TypeError, ValueError):
            continue
        if src_mark in by_mark and dst_mark in by_mark and src_mark != dst_mark:
            edges.append((by_mark[src_mark], by_mark[dst_mark]))
    return edges


class EdgeRunner:
    """Loads one chat VLM (any `benchmark.models.MODELS` entry) and keeps it
    resident on GPU across calls."""

    def __init__(self, spec: ModelSpec, device: str = "cuda"):
        self.spec = spec
        self.device = device
        dtype = getattr(torch, spec.dtype)
        self.processor = AutoProcessor.from_pretrained(
            spec.hf_repo, trust_remote_code=spec.trust_remote_code,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            spec.hf_repo,
            trust_remote_code=spec.trust_remote_code,
            attn_implementation=spec.attn_implementation,
            dtype=dtype,
        ).to(device).eval()

    def unload(self):
        del self.model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def discover_edges(self, image: Image.Image, diagram: Diagram,
                        max_new_tokens: int = 1024) -> Diagram:
        """Returns a new Diagram with `arrows` populated at the top level
        (edges reference ids across blocks, so they aren't nested)."""
        marked_image, marks = draw_marks(image, diagram)
        if not marks:
            return diagram

        legend = _build_legend(marks, diagram)
        prompt = EDGE_PROMPT.format(legend=legend)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": marked_image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw_output = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

        edge_pairs = _parse_edges(raw_output, marks)
        diagram.arrows = [Arrow(src_id=s, dst_id=d, src="som") for s, d in edge_pairs]
        return diagram