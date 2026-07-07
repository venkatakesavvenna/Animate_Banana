"""(image, Diagram) -> generated code, backend-selectable (TikZ or SVG).

Reuses the chat-VLM loading pattern from `benchmark/infer.py::ModelRunner`
(same registry, same `apply_chat_template` + `.generate()` call shape) but
adds the assembled XML as structural grounding in the prompt, and extracts
either TikZ (`_extract_tikz`, adapted from `infer.py`) or SVG
(`_extract_svg`) depending on `backend`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from img_2_svg_pretraining.benchmark.models import ModelSpec
from img_2_svg_pretraining.data_engine.schema import Diagram, to_xml

TIKZ_INSTRUCTIONS = r"""Convert this diagram into absolutely faithful TikZ code. ENSURE A GENEROUS CANVAS SIZE.

FOLLOW THE INSTRUCTIONS GIVEN BELOW STRICTLY:

#1. You must use the fit library to draw parent blocks and composite elements, but you are strictly FORBIDDEN from using 'local bounding box' or the backgrounds library.
- For objects within blocks: Include them directly within the fit module (e.g., \node[fit=(obj1) (obj2)] (group_id) {};).
- For composite visual elements: You must first define an invisible container node with exact dimensions to act as the group's anchor. Then, draw the composite paths inside that region.

#2. If there are 3D elements or pictures that you are UNABLE TO DRAW DIRECTLY, insert placeholders in their positions using a placeholder_node style.

#3. Ensure that all arrows are drawn as \draw (obj1) -- (obj2) and their directions are faithfully preserved from the original image and from the XML structure below.

#4. Define all styles at the beginning of the code, inside a standalone tikzpicture document.

#5. Strongly ensure that the code compiles without errors. Use standard ASCII spaces only, no tabs or non-breaking spaces.
"""

SVG_INSTRUCTIONS = """Convert this diagram into a faithful, self-contained SVG document.

FOLLOW THE INSTRUCTIONS GIVEN BELOW STRICTLY:

#1. Emit a single well-formed <svg>...</svg> document with an explicit viewBox covering the full diagram extent, generous margins.

#2. Represent each block/node from the XML structure below as a <rect>/<g> with matching approximate position and size, and its label as a nested <text> element.

#3. Represent each arrow as a <line> or <path> with a marker-end arrowhead (define a <marker> in <defs>), connecting the correct node boundaries in the correct direction.

#4. For raster/photo regions, emit a <rect> placeholder with a distinct fill and a <text> label naming the region -- do not attempt to embed real image data.

#5. Ensure the SVG is well-formed XML (all tags closed, attributes quoted) since it will be parsed and rasterized directly.
"""

INSTRUCTIONS_BY_BACKEND = {"tikz": TIKZ_INSTRUCTIONS, "svg": SVG_INSTRUCTIONS}


def build_prompt(diagram: Diagram, backend: str) -> str:
    if backend not in INSTRUCTIONS_BY_BACKEND:
        raise ValueError(f"Unknown backend '{backend}'. Known: {sorted(INSTRUCTIONS_BY_BACKEND)}")
    xml_text = to_xml(diagram)
    return (
        f"{INSTRUCTIONS_BY_BACKEND[backend]}\n\n"
        "Here is the detected structure of the diagram (node/block ids, labels, "
        "approximate bounding boxes, and edges) to use as grounding -- treat the "
        "image as the source of truth for visual style, and this XML as the "
        f"source of truth for structure/connectivity:\n\n{xml_text}"
    )


@dataclass
class CodegenResult:
    raw_output: str
    code: str | None  # extracted TikZ/SVG source, or None if extraction failed


def _extract_tikz(raw_output: str) -> str | None:
    text = raw_output.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if "```" in text:
        parts = text.split("```")
        candidates = [parts[i] for i in range(1, len(parts), 2)]
        if candidates:
            text = max(candidates, key=len).strip()
            for prefix in ("latex", "tex", "LaTeX", "TeX"):
                if text.startswith(prefix):
                    text = text[len(prefix):].lstrip()
    if "\\documentclass" not in text:
        return None
    text = text[text.index("\\documentclass"):]
    if "\\end{document}" in text:
        text = text[: text.index("\\end{document}") + len("\\end{document}")]
    return text


def _extract_svg(raw_output: str) -> str | None:
    text = raw_output.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if "```" in text:
        parts = text.split("```")
        candidates = [parts[i] for i in range(1, len(parts), 2)]
        if candidates:
            text = max(candidates, key=len).strip()
            for prefix in ("svg", "xml"):
                if text.startswith(prefix):
                    text = text[len(prefix):].lstrip()
    if "<svg" not in text:
        return None
    text = text[text.index("<svg"):]
    if "</svg>" in text:
        text = text[: text.index("</svg>") + len("</svg>")]
    return text


EXTRACTORS = {"tikz": _extract_tikz, "svg": _extract_svg}


class CodegenRunner:
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
    def generate(self, image_path: Path, diagram: Diagram, backend: str,
                 max_new_tokens: int = 16384) -> CodegenResult:
        image = Image.open(image_path).convert("RGB")
        prompt = build_prompt(diagram, backend)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
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

        code = EXTRACTORS[backend](raw_output)
        return CodegenResult(raw_output=raw_output, code=code)
