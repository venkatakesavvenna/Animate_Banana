"""Stage 2b -- Structure Parser.

Reads the metadata-infused diagram code and emits a strictly nested XML tree
of the diagram's structure: which elements exist, how they nest, and how they
connect. Visual properties (coordinates, colours, line widths) are discarded;
`text_node` elements are excluded entirely, since the tree describes
structure and routing rather than labels.

Independent of the strategizer (2a) -- see that module's note.

The element ids in this XML are the authoritative set that every downstream
`focus` reference is validated against, which is why `element_ids()` lives
here rather than being re-derived per consumer.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..backends import Message
from ..extract import extract_xml
from ..prompts import load_prompt
from ..runner import AgentContext, StageReport, run_agent
from ..samples import PaperSample

AGENT = "parser"
ROOT_TAG = "Diagram"

# Attributes any of the parser's tags may carry an id under.
_ID_ATTRS = ("id", "i", "name")


def build_request(ctx: AgentContext, sample: PaperSample) -> list[Message]:
    code_path = ctx.paths.resolve_code(sample.id)
    code = Path(code_path).read_text(encoding="utf-8")
    prompt = load_prompt(ctx.agent.prompt)
    # tikz_parser.yaml ends with "Here is the TikZ code to process:" -- the
    # code is meant to be appended to the prompt.
    return [Message.user(f"{prompt}\n\n{code}")]


def parse_response(ctx: AgentContext, sample: PaperSample, raw: str) -> str:
    xml_text = extract_xml(raw, root_tag=ROOT_TAG) or extract_xml(raw)
    if xml_text is None:
        raise ValueError(
            "no XML document in the response; "
            f"raw output kept at {ctx.paths.raw_output(AGENT, sample.id)}"
        )
    try:
        ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"response is not well-formed XML: {e}") from e
    return xml_text


def element_ids(xml_text: str) -> set[str]:
    """Every element id in the structure XML.

    Used to validate that a sequence's `focus` entries reference elements
    that actually exist.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return set()

    ids: set[str] = set()
    for el in root.iter():
        for attr in _ID_ATTRS:
            value = el.get(attr)
            if value:
                ids.add(value)
                break
    return ids


def load_element_ids(path: Path) -> set[str]:
    return element_ids(Path(path).read_text(encoding="utf-8"))


def run(cfg, samples: list[PaperSample], force: bool = False) -> StageReport:
    ctx = AgentContext(cfg, AGENT)
    try:
        return run_agent(
            ctx=ctx,
            samples=samples,
            output_path=lambda s: ctx.paths.xml(s.id),
            build_request=lambda s: build_request(ctx, s),
            handle_response=lambda s, raw: parse_response(ctx, s, raw),
            force=force,
        )
    finally:
        ctx.unload()
