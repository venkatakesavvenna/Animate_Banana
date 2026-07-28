"""Animation styles, defined once as executable constraints.

Three consumers need to agree on what a style means: the sequencer (which
plans within its constraints), the designer (which realizes it in code), and
the benchmark's style-compliance metric (which scores it). If each carried
its own interpretation they would drift, and the benchmark would be scoring
a different contract than the pipeline was built against.

So each style declares:
- `description` / `directive` -- prose injected into agent prompts
- `actions`   -- the animation actions the sequencer may emit
- `check`     -- a predicate over a sequence returning violation strings

Style semantics follow section 4.4 of the AnimateBench design document.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

# Forward reference: schema.AnimationSequence. Kept loose to avoid a circular
# import, since schema.py validates against these styles.
Sequence = "AnimationSequence"


@dataclass
class Style:
    name: str
    description: str
    directive: str
    actions: tuple[str, ...]
    single_pass: bool
    check: Callable[[object], list[str]] = field(default=lambda seq: [])

    def prompt_block(self) -> str:
        """Style constraints as prompt text for the sequencer/designer."""
        return (
            f"ANIMATION STYLE: {self.name}\n"
            f"{self.description}\n\n"
            f"CONSTRAINTS YOU MUST SATISFY:\n{self.directive}\n\n"
            f"PERMITTED ACTIONS: {', '.join(self.actions)}"
        )


def _steps(seq) -> list:
    """Traversal steps as (node_id, node) pairs, skipping dangling ids.

    Dangling traversal entries are a validity problem reported by the
    validity metrics; style checks shouldn't double-report them.
    """
    by_id = {n.id: n for n in seq.nodes}
    return [(nid, by_id[nid]) for nid in seq.traversal if nid in by_id]


def _check_single_pass(seq) -> list[str]:
    counts = Counter(seq.traversal)
    repeated = sorted(n for n, c in counts.items() if c > 1)
    if repeated:
        return [f"node revisited in a single-pass style: {repeated}"]
    return []


def _check_actions(seq, allowed: tuple[str, ...]) -> list[str]:
    bad = sorted({n.action for n in seq.nodes if n.action and n.action not in allowed})
    if bad:
        return [f"action(s) {bad} not permitted for this style (allowed: {list(allowed)})"]
    return []


def _check_has_focus(seq) -> list[str]:
    """Every visited step must name the elements it acts on.

    A step with no focus can't be rendered as an attention box, a highlight
    or a viewport, so this is a real defect for the three styles that need it
    rather than a stylistic preference.
    """
    empty = [nid for nid, node in _steps(seq) if not node.focus]
    if empty:
        return [f"traversal step(s) with no focus elements: {sorted(set(empty))}"]
    return []


def _progressive_reveal(seq) -> list[str]:
    violations = _check_actions(seq, STYLES["progressive_reveal"].actions)
    violations += _check_single_pass(seq)
    # A revealed unit stays visible, so revealing the same element twice means
    # the second reveal is a no-op the viewer cannot perceive.
    seen: set[str] = set()
    for nid, node in _steps(seq):
        if node.action != "reveal":
            continue
        already = sorted(set(node.focus) & seen)
        if already:
            violations.append(f"step '{nid}' re-reveals already-visible element(s): {already}")
        seen.update(node.focus)
    return violations


def _sliding_bbox(seq) -> list[str]:
    violations = _check_actions(seq, STYLES["sliding_bbox"].actions)
    violations += _check_has_focus(seq)
    # The box covers one coherent region per step; a step spanning many
    # scattered elements has no meaningful single box.
    for nid, node in _steps(seq):
        if len(node.focus) > 6:
            violations.append(
                f"step '{nid}' focuses {len(node.focus)} elements; an attention box "
                f"should cover a compact region (<= 6)"
            )
    return violations


def _highlight_dim(seq) -> list[str]:
    violations = _check_actions(seq, STYLES["highlight_dim"].actions)
    violations += _check_has_focus(seq)
    return violations


def _zoom_pan(seq) -> list[str]:
    violations = _check_actions(seq, STYLES["zoom_pan"].actions)
    violations += _check_has_focus(seq)
    return violations


STYLES: dict[str, Style] = {
    "progressive_reveal": Style(
        name="progressive_reveal",
        description=(
            "Each traversal unit begins hidden or visually subdued. When visited it "
            "becomes visible and stays visible for the rest of the animation."
        ),
        directive=(
            "- Every element starts hidden and is revealed exactly once.\n"
            "- Previously revealed elements remain visible; never hide them again.\n"
            "- The traversal order is the order of revelation, so it must read as a "
            "narrative build-up from nothing to the complete figure.\n"
            "- Do not revisit a node you have already revealed."
        ),
        actions=("reveal", "emphasize"),
        single_pass=True,
        check=_progressive_reveal,
    ),
    "sliding_bbox": Style(
        name="sliding_bbox",
        description=(
            "The complete figure stays visible throughout. An attention box moves to "
            "cover the elements referenced by each traversal step."
        ),
        directive=(
            "- The whole figure is visible from the first frame; nothing is ever hidden.\n"
            "- Each step selects one compact focus region the box moves to cover.\n"
            "- Keep each step's focus elements spatially adjacent so a single box can "
            "contain them.\n"
            "- Consecutive steps should be near each other where possible, so the box "
            "travels smoothly rather than jumping across the figure."
        ),
        actions=("focus", "highlight"),
        single_pass=False,
        check=_sliding_bbox,
    ),
    "highlight_dim": Style(
        name="highlight_dim",
        description=(
            "The complete figure stays visible. At each step the selected elements are "
            "emphasized while everything else is de-emphasized."
        ),
        directive=(
            "- The whole figure is visible from the first frame.\n"
            "- Each step names the elements to emphasize; all others are dimmed.\n"
            "- Every step must name at least one focus element.\n"
            "- Steps change the active focus set only; they never alter the hierarchy."
        ),
        actions=("highlight", "emphasize", "focus"),
        single_pass=False,
        check=_highlight_dim,
    ),
    "zoom_pan": Style(
        name="zoom_pan",
        description=(
            "Each traversal step specifies a viewport containing the selected elements; "
            "the camera zooms and pans between viewports."
        ),
        directive=(
            "- Each step's focus elements define a viewport that must stay within the "
            "figure boundaries.\n"
            "- Focus elements must remain legible, so avoid viewports so tight that text "
            "is clipped or so wide that the focus is lost.\n"
            "- Order steps so the camera moves coherently rather than jumping randomly."
        ),
        actions=("zoom", "pan", "focus"),
        single_pass=False,
        check=_zoom_pan,
    ),
}

ALL_ACTIONS: tuple[str, ...] = tuple(
    sorted({a for s in STYLES.values() for a in s.actions})
)


def get_style(name: str) -> Style:
    if name not in STYLES:
        raise KeyError(f"unknown animation style '{name}'. Known: {sorted(STYLES)}")
    return STYLES[name]


def check_style(seq, style_name: str | None = None) -> list[str]:
    """Return style-constraint violations for a sequence ([] if compliant)."""
    style = get_style(style_name or seq.style)
    return style.check(seq)
