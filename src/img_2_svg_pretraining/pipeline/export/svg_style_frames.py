"""Per-style SVG frame extractors, transcribed from the reference scripts.

WHY PER-STYLE AND NOT ONE SAMPLER. The five animation styles encode "a state"
differently, and a single uniform grid gets all of them slightly wrong:

  progressive_reveal / colour_pop / alpha_masking
      Elements cross-fade. A uniform grid lands on the exact ticks where one
      element fades out and the next fades in -- which is where BOTH are near
      opacity 0. Measured on a narrated animation: 21 of 22 exported frames
      contained no subtitle, from CSS that was perfectly correct.

  hopping_bounding_box
      `step-end` timing: the box snaps between discrete rest positions and has
      no in-transit state at all. The states ARE the keyframes, so sampling
      anywhere inside each one is equivalent -- but sampling ON a boundary is
      a coin flip between two positions.

  sliding_bounding_box
      `ease-in-out`: the box holds on an element, then genuinely interpolates
      to the next. The in-transit positions are real states that a hold-only
      sampler throws away -- which is exactly why sliding and hopping have been
      indistinguishable in the exported decks despite correct, different CSS.

THE SHARED IDEA, from the reference extractors: sample the MIDDLE of each
state, never its edge. `state_midpoints` in svg_frames.py implements that for
the fade styles; `slide_samples` below adds the transition frames sliding needs.

These are sampling policies over one renderer, not five renderers. Frames are
still produced by `svg_frames.render_svg_frames`, which drives the Web
Animations API rather than stripping CSS and reconstructing visibility by hand
-- an important difference from the reference scripts, because a hand-rebuilt
frame can look correct while the animation the reader actually watches is
broken. Here the frames are what the browser really renders.
"""
from __future__ import annotations

# Styles whose box genuinely moves between rest positions. These need frames
# sampled DURING the transition, not only on the holds.
MOTION_STYLES = {"sliding_bounding_box", "sliding_bbox"}

# Styles whose box teleports; every distinct state is a hold.
DISCRETE_STYLES = {"hopping_bounding_box", "hopping_box"}

# How many samples to take inside each slide transition. The reference
# extractor uses 2 (at 1/3 and 2/3 of the gap), which is enough to show the box
# in flight without inflating the deck.
TRANSITION_SAMPLES = 2


def slide_samples(bounds: list[float], transition_samples: int = TRANSITION_SAMPLES
                  ) -> list[float]:
    """Hold midpoints PLUS in-transit samples, for a sliding box.

    `bounds` is the sorted list of state boundaries in ms. Between consecutive
    boundaries we take the midpoint (the box at rest, or mid-glide), and then
    additionally split each gap so the box is caught in flight.

    Without the in-transit samples a slide exports as a series of rest
    positions -- identical in structure to a hop, and the reason the two styles
    have looked the same. Measured: sliding yields 26 unique states at 2fps and
    194 at 25fps, so the motion is real and was simply never sampled.
    """
    out: list[float] = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        out.append((lo + hi) / 2.0)
        span = hi - lo
        if span <= 0:
            continue
        for k in range(1, transition_samples + 1):
            out.append(lo + span * k / (transition_samples + 1.0))
    return sorted(set(out))


def sample_times_for(style: str, timeline: dict, total_ms: float) -> list[float]:
    """The timestamps to screenshot for `style`, or [] to use the caller's grid.

    Every style routes through `state_midpoints`; sliding additionally asks for
    transition frames. Returning [] lets the caller fall back to its uniform
    grid, which is what happens for a document whose timeline exposes no
    keyframe stops to derive boundaries from.
    """
    from .svg_frames import BOUNDARY_EPSILON_MS, state_midpoints

    mids = state_midpoints(timeline, total_ms)
    if not mids:
        return []
    if style not in MOTION_STYLES:
        # Fades and hops alike: one sample per state, taken at its midpoint.
        return mids

    stops = timeline.get("stageTimes") or []
    times = sorted({0.0, float(total_ms)} |
                   {float(x) for x in stops if 0.0 <= float(x) <= total_ms})
    bounds: list[float] = []
    for ms in times:
        if not bounds or ms - bounds[-1] > BOUNDARY_EPSILON_MS:
            bounds.append(ms)
    return slide_samples(bounds)
