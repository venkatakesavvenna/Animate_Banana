"""Runs image -> TikZ inference using plain `transformers` generation (no
vLLM/serving layer -- see models.py for why), then compiles the result the
same way viewer/compile.py does so scores are computed against the same
renderer annotators see.

The model is loaded once per benchmark run and reused across all samples
(see ModelRunner) -- loading per-sample would repeatedly pay multi-second
weight-transfer-to-GPU cost for no benefit, since nothing about the model
changes between samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModelForMultimodalLM

from img_2_svg_pretraining.benchmark.models import ModelSpec
from img_2_svg_pretraining.viewer.compile import compile_tikz

PROMPT = r"""Convert this diagram into absolutely faithful TikZ code. ENSURE A GENEROUS CANVAS SIZE.

FOLLOW THE INSTRUCTIONS GIVEN BELOW STRICTLY:

#1. You must use the fit library to draw parent blocks and composite elements, but you are strictly FORBIDDEN from using 'local bounding box' or the backgrounds library.
- For objects within blocks: Include them directly within the fit module (e.g., \node[fit=(obj1) (obj2)] (group_id) {};).
- For composite visual elements: You must first define an invisible container node with exact dimensions to act as the group's anchor. Then, draw the composite paths inside that region.
    Example:
    \coordinate (chart_tl) at (0, 2.5);
    \coordinate (chart_br) at (6.2, -0.4);
    % ... draw the complex charts/paths here ...
    \node[fit=(chart_tl) (chart_br)] (group_id) {};
- Ensure all arrows pointing to these composite elements anchor ONLY to the group id of these fit nodes, not the internal paths.

#2. If there are 3D elements or pictures that you are UNABLE TO DRAW DIRECTLY, insert placeholders in their positions in the following way:
- Example: \node[placeholder_node, minimum width=x cm, minimum height=y cm] (ex_id) at (coordinates) {ex_name};
- If you are ABLE TO DRAW some 3D element (say 3D Convolution blocks), then DO NOT use the placeholder for them.

#3. Ensure that all arrows are drawn as \draw (obj1) -- (obj2) and their directions are faithfully preserved from the original image.

#4. Define all the styles at the beginning of the code.
- For example:
    \documentclass[tikz, border=10pt]{standalone}
    \usetikzlibrary{fit, positioning, arrows.meta, shapes, calc}
    \begin{document}
    \begin{tikzpicture}[
        >=Stealth,
        placeholder_node/.style={...}
    ]

#5. Strongly ensure that the code compiles without errors.

#6. Strictly use standard ASCII spaces for all indentation. Do NOT output non-breaking spaces or tabs anywhere in the code.
"""


@dataclass
class InferenceResult:
    sample_id: str
    raw_output: str
    tex: str | None          # extracted TikZ source, or None if extraction failed
    compiled_ok: bool
    png_path: Path | None
    compile_log: str


def _extract_tikz(raw_output: str) -> str | None:
    """Extract the TikZ document from raw model output.

    Reasoning models (e.g. Qwen3.5) emit a "<think>...</think>" block of
    chain-of-thought before the actual answer -- strip that first, since the
    thinking text often contains prose mentions of \\documentclass/tikz
    syntax while planning, which would otherwise be mistaken for the answer.
    """
    text = raw_output.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    if "```" in text:
        parts = text.split("```")
        # take the largest fenced block -- handles ```latex / ```tex / bare ```
        candidates = [parts[i] for i in range(1, len(parts), 2)]
        if candidates:
            text = max(candidates, key=len).strip()
            for prefix in ("latex", "tex", "LaTeX", "TeX"):
                if text.startswith(prefix):
                    text = text[len(prefix):].lstrip()

    if "\\documentclass" not in text:
        return None
    # Trim any leading prose before the actual document, and stop at
    # \end{document} in case generation continued past it (or was truncated
    # after it with trailing commentary).
    text = text[text.index("\\documentclass"):]
    if "\\end{document}" in text:
        text = text[: text.index("\\end{document}") + len("\\end{document}")]
    return text


class ModelRunner:
    """Loads one model+processor and keeps them resident on GPU for the
    duration of a benchmark run."""

    def __init__(self, spec: ModelSpec, device: str = "cuda"):
        self.spec = spec
        self.device = device
        dtype = getattr(torch, spec.dtype)
        self.processor = AutoProcessor.from_pretrained(
            spec.hf_repo, trust_remote_code=spec.trust_remote_code,
        )
        # If not gemma4
        if spec.name != "gemma-4-12b" and spec.name != "gemma-4-e4b":
            self.model = AutoModelForImageTextToText.from_pretrained(
                spec.hf_repo,
                trust_remote_code=spec.trust_remote_code,
                attn_implementation=spec.attn_implementation,
                dtype=dtype,
            ).to(device).eval()
        else:
            self.model = AutoModelForMultimodalLM.from_pretrained(
                spec.hf_repo,
                trust_remote_code=spec.trust_remote_code,
                attn_implementation=spec.attn_implementation,
                dtype=dtype,
            ).to(device).eval()

    def unload(self):
        del self.model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(self, image_path: Path, max_new_tokens: int = 16384) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]
        output_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
        new_tokens = output_ids[0][input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(self, image_paths: list[Path], max_new_tokens: int = 16384) -> list[str]:
        """Batched variant of generate(). Left-pads so the generation prompt
        ends at the same position for every sequence in the batch -- required
        for decoder-only `.generate()`, since right-padding would insert pad
        tokens between the prompt and the first generated token.
        """
        images = [Image.open(p).convert("RGB") for p in image_paths]
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
            for image in images
        ]

        original_padding_side = getattr(self.processor.tokenizer, "padding_side", None)
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"
        try:
            inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        finally:
            if original_padding_side is not None:
                self.processor.tokenizer.padding_side = original_padding_side

        input_len = inputs["input_ids"].shape[-1]
        output_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
        new_tokens = output_ids[:, input_len:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)


def _to_inference_result(sample_id: str, raw_output: str, cache_dir: Path) -> InferenceResult:
    tex = _extract_tikz(raw_output)
    if tex is None:
        reason = ("no \\documentclass found in model output -- likely truncated by "
                   "max_new_tokens before reaching the answer" if "</think>" in raw_output
                   and "\\documentclass" not in raw_output.split("</think>", 1)[1]
                   else "no \\documentclass found in model output")
        return InferenceResult(
            sample_id=sample_id, raw_output=raw_output, tex=None,
            compiled_ok=False, png_path=None,
            compile_log=reason,
        )

    result = compile_tikz(tex, cache_dir)
    return InferenceResult(
        sample_id=sample_id, raw_output=raw_output, tex=tex,
        compiled_ok=result.ok, png_path=result.png_path, compile_log=result.log,
    )


def run_inference(
    runner: ModelRunner,
    sample_id: str,
    image_path: Path,
    cache_dir: Path,
    max_new_tokens: int = 16384,
) -> InferenceResult:
    raw_output = runner.generate(image_path, max_new_tokens=max_new_tokens)
    return _to_inference_result(sample_id, raw_output, cache_dir)


def run_inference_batch(
    runner: ModelRunner,
    sample_ids: list[str],
    image_paths: list[Path],
    cache_dir: Path,
    max_new_tokens: int = 16384,
) -> list[InferenceResult]:
    """Batched variant of run_inference(): one forward-generation pass over
    all images in the batch, then per-sample TikZ extraction + compilation
    (compilation is a latexmk subprocess, not GPU work, so it stays
    per-sample rather than batched)."""
    raw_outputs = runner.generate_batch(image_paths, max_new_tokens=max_new_tokens)
    return [
        _to_inference_result(sample_id, raw_output, cache_dir)
        for sample_id, raw_output in zip(sample_ids, raw_outputs)
    ]