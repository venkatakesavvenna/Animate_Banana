"""Pull structured artifacts out of raw model output.

Models wrap answers in markdown fences, prefix them with reasoning blocks,
and append commentary. These extractors are provider-agnostic -- the same
text post-processing applies whether the output came from Gemini, a vLLM
server or a local checkpoint.

`extract_tikz` is ported from `benchmark/infer.py::_extract_tikz`, which was
already provider-neutral; the SVG/XML/JSON siblings follow its approach.
"""
from __future__ import annotations

import json
import re

_FENCE_LANGS = ("latex", "tex", "LaTeX", "TeX", "xml", "svg", "json", "html")


def strip_reasoning(text: str) -> str:
    """Drop a leading <think>...</think> block.

    Reasoning models plan in prose that often mentions \\documentclass or
    quotes XML fragments; without stripping, that planning text gets mistaken
    for the answer.
    """
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def largest_fenced_block(text: str) -> str:
    """Return the largest ```-fenced block, or the text unchanged if unfenced."""
    if "```" not in text:
        return text
    parts = text.split("```")
    candidates = [parts[i] for i in range(1, len(parts), 2)]
    if not candidates:
        return text
    block = max(candidates, key=len).strip()
    for lang in _FENCE_LANGS:
        if block.lower().startswith(lang.lower()):
            block = block[len(lang):].lstrip()
            break
    return block


def extract_tikz(raw_output: str) -> str | None:
    """Extract a compilable LaTeX document, or None if absent."""
    text = largest_fenced_block(strip_reasoning(raw_output))
    if "\\documentclass" not in text:
        return None
    text = text[text.index("\\documentclass"):]
    end = "\\end{document}"
    if end in text:
        text = text[: text.index(end) + len(end)]
    return text.strip()


def extract_svg(raw_output: str) -> str | None:
    """Extract an <svg>...</svg> document, or None if absent."""
    text = largest_fenced_block(strip_reasoning(raw_output))
    start = text.find("<svg")
    if start == -1:
        return None
    end = text.rfind("</svg>")
    if end == -1:
        return None
    return text[start:end + len("</svg>")].strip()


def extract_code(raw_output: str, target: str) -> str | None:
    """Dispatch to the extractor for the configured target format."""
    if target == "tikz":
        return extract_tikz(raw_output)
    if target == "svg":
        return extract_svg(raw_output)
    raise ValueError(f"unknown target '{target}' (expected 'tikz' or 'svg')")


def extract_xml(raw_output: str, root_tag: str | None = None) -> str | None:
    """Extract an XML document.

    With `root_tag`, slices from the first `<root_tag` to its matching close,
    which is more robust than trusting the fence when the model emits a
    preamble sentence inside the block.
    """
    text = largest_fenced_block(strip_reasoning(raw_output))
    if root_tag:
        start = text.find(f"<{root_tag}")
        close = f"</{root_tag}>"
        end = text.rfind(close)
        if start != -1 and end != -1:
            return text[start:end + len(close)].strip()
        return None

    match = re.search(r"<([A-Za-z_][\w.-]*)", text)
    if not match:
        return None
    tag = match.group(1)
    close = f"</{tag}>"
    end = text.rfind(close)
    if end == -1:
        return None
    return text[match.start():end + len(close)].strip()


def extract_json(raw_output: str) -> object | None:
    """Extract and parse a JSON object or array.

    Tries the fenced block first, then falls back to the outermost {...} or
    [...] span. Returns None rather than raising so a malformed response is a
    recorded failure rather than a crashed batch.
    """
    text = largest_fenced_block(strip_reasoning(raw_output))

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


# -- strategizer output --------------------------------------------------

_TRAVERSAL_RE = re.compile(r"TRAVERSAL_STYLE\s*:\s*\[?\s*(OVERVIEW_FIRST|DETAIL_FIRST)",
                           re.IGNORECASE)
_REASONING_RE = re.compile(r"REASONING\s*:\s*(.+?)(?=\n\s*TRAVERSAL_STYLE\s*:|\Z)",
                           re.IGNORECASE | re.DOTALL)
_STEP_RE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.+?)\s*$")


def extract_strategy(raw_output: str) -> dict | None:
    """Parse the strategizer's REASONING / TRAVERSAL_STYLE / SEQUENCE format.

    Returns {"reasoning", "traversal_style", "sequence": [...]}, or None if
    no traversal style is present -- that field is the one the downstream
    sequencer cannot proceed without.
    """
    text = strip_reasoning(raw_output)

    style_match = _TRAVERSAL_RE.search(text)
    if not style_match:
        return None
    traversal_style = style_match.group(1).upper()

    reasoning = ""
    reasoning_match = _REASONING_RE.search(text)
    if reasoning_match:
        reasoning = " ".join(reasoning_match.group(1).split())

    sequence: list[str] = []
    after = text[style_match.end():]
    marker = re.search(r"SEQUENCE\s*:", after, re.IGNORECASE)
    if marker:
        for line in after[marker.end():].splitlines():
            step = _STEP_RE.match(line)
            if step:
                event = step.group(2).strip()
                # The prompt's own template uses bracketed placeholders; strip
                # the brackets when a model echoes that formatting.
                if event.startswith("[") and event.endswith("]"):
                    event = event[1:-1].strip()
                if event:
                    sequence.append(event)

    return {
        "reasoning": reasoning,
        "traversal_style": traversal_style,
        "sequence": sequence,
    }
