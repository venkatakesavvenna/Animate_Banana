"""Prompt surgery for input-removal ablations.

An ablation that stops sending an input must ALSO stop the prompt claiming that
input was sent. Leaving the claim in place is not cosmetic: the sequencer prompt
says the structural graph is "your EXCLUSIVE source" for depth and parent
metadata, and the planner critic's prompt tells the model to use overlaid marker
numbers on an image. A model told it has an input it never received will either
hallucinate the missing content or refuse -- and either way the ablation would
measure prompt inconsistency rather than the value of the input.

WHAT THIS DOES, and deliberately no more:

  * removes whole items from a "You will be provided with:" / "### INPUTS"
    numbered list when the item declares a dropped input, INCLUDING the item's
    indented continuation lines, then renumbers what remains;
  * removes a dropped input from an inline enumeration such as
    "provided with a Target Image, its Static SVG Code, JSON Animation Sequence".

It does NOT try to rewrite objective or rule prose that merely mentions the
input in passing. Rewriting instructions automatically is how an ablation
quietly becomes a different experiment; anything beyond a declaration line is
left for a human to decide.
"""
from __future__ import annotations

import re

# What counts as "this line declares input X". Matched against the FIRST line of
# a numbered item, case-insensitively.
_DECLARES = {
    "image": re.compile(
        r"\b(an?\s+image\b|the\s+diagram\s+image\b|target\s+image\b|"
        r"diagram\s+image\b|source\s+image\b)", re.I),
    "xml": re.compile(
        r"\b(hierarchical\s+graph|structure\s+xml|structural\s+xml|"
        r"structural\s+(dependency\s+)?graph|xml/json)\b", re.I),
    "context": re.compile(
        r"\b(paper\s+title|abstract|diagram\s+caption|method\s+section)\b", re.I),
}

# Inline enumerations: "a Target Image, its Static SVG Code, ..." -- drop just
# the named noun phrase and its separator, leaving the sentence grammatical.
_INLINE = {
    "image": re.compile(r"\ba\s+Target\s+Image,\s*", re.I),
    "xml": re.compile(r"\bits\s+Structure\s+XML,\s*", re.I),
}

_ITEM = re.compile(r"^(\s*)(\d+)\.\s")


def _label(line: str) -> str:
    """The declaration part of a numbered item, not its whole description.

    Matching the full line is wrong and was measured wrong: the sequencer's
    item 1 is "**Original TikZ Code:** ... intentionally omitted from the
    structural XML/JSON graph", whose DESCRIPTION mentions XML while the item
    itself declares the code. Dropping "xml" removed the code input too.

    So match only up to the first colon when one appears early (the bolded
    "**Label:**" convention), else the opening clause.
    """
    cut = line.find(":")
    if 0 < cut <= 90:
        return line[:cut]
    return line[:90]



def drop_declared_inputs(text: str, drop: set[str] | frozenset[str]) -> str:
    """Strip declarations of `drop`-ed inputs from a prompt body.

    `drop` holds keys of `_DECLARES` (``{"image"}``, ``{"xml"}`` ...). Returns
    the text unchanged when nothing matches, so it is safe to call on every
    prompt regardless of whether that prompt declares its inputs at all.
    """
    if not drop:
        return text

    lines = text.split("\n")
    out: list[str] = []
    i = 0
    removed = False
    while i < len(lines):
        m = _ITEM.match(lines[i])
        if not m:
            out.append(lines[i]); i += 1; continue

        # Collect this numbered item: its own line plus any deeper-indented or
        # blank continuation lines, up to the next item or a dedented line.
        item = [lines[i]]
        base_indent = len(m.group(1))
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if _ITEM.match(nxt):
                break
            if nxt.strip() == "":
                # A blank line ends the item unless the list continues after it.
                k = j + 1
                if k < len(lines) and _ITEM.match(lines[k]):
                    break
                if k < len(lines) and (len(lines[k]) - len(lines[k].lstrip())) > base_indent:
                    item.append(nxt); j += 1; continue
                break
            if (len(nxt) - len(nxt.lstrip())) > base_indent:
                item.append(nxt); j += 1; continue
            break

        head = _label(item[0])
        if any(_DECLARES[d].search(head) for d in drop if d in _DECLARES):
            removed = True                      # drop the whole item
        else:
            out.extend(item)
        i = j

    text = "\n".join(out)

    if removed:
        text = _renumber(text)

    for d in drop:
        pat = _INLINE.get(d)
        if pat:
            text = pat.sub("", text)
    return text


def _renumber(text: str) -> str:
    """Renumber each numbered run so a removed item leaves no gap.

    A gap ("1. ... 3. ...") reads as a missing input the model should ask for,
    which is exactly the confusion this module exists to avoid.
    """
    lines = text.split("\n")
    out: list[str] = []
    counter = 0
    for ln in lines:
        m = _ITEM.match(ln)
        if m:
            counter += 1
            out.append(f"{m.group(1)}{counter}. " + ln[m.end():])
        else:
            if ln.strip() and not ln.startswith((" ", "\t")):
                counter = 0                     # a dedented non-item ends the run
            out.append(ln)
    return "\n".join(out)
