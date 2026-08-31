"""Rasterize an animated SVG into frames, by driving a real browser.

The animation designer emits CSS `@keyframes` ("Use pure CSS. NO Javascript"),
which cairosvg cannot render: it has no animation engine and no way to seek a
timeline. So frames come from headless chromium, which is asked to pause every
animation, set `currentTime` to an exact millisecond, and screenshot.

Promoted from `export/svg/ppt_and_video.py`, which was a standalone script
keyed to a hardcoded `alpha_making.html`. What is kept is the part that is
hard to get right:

- **Timing is read from the document, not guessed.** `document.getAnimations()`
  reports each animation's real duration and keyframe boundaries, so the frame
  rate is derived from the animation's own granularity rather than a fixed
  constant that might skip a whole stage.
- **Frames are deduped by computed style, not by pixels.** Two timestamps are
  the same visual state iff the resolved values of the properties the keyframes
  actually animate are identical -- no rendering, no image diff, no threshold
  to tune.
"""
from __future__ import annotations

import math
from pathlib import Path

# How many video frames to sample within the shortest gap between stage
# transitions, so no stage can pass by unsampled.
SAMPLES_PER_STAGE = 3
FLOOR_FPS = 4
CEIL_FPS = 30

# Two stage boundaries closer than this are treated as one. A cheap pre-filter;
# the authoritative dedupe is the computed-style fingerprint below.
DEDUPE_TOLERANCE_MS = 10
FINGERPRINT_ROUND_DECIMALS = 2

_HTML_WRAPPER = """<!doctype html>
<html><head><meta charset="utf-8">{base}<style>
  html, body {{ margin: 0; padding: 0; background: #fff; }}
  svg {{ display: block; }}
</style></head><body>
{svg}
</body></html>"""


class SvgRenderError(RuntimeError):
    """Raised when the browser path cannot run at all."""


def _require_playwright():
    """Import playwright, or explain how to get it.

    Install note, learned the hard way: the container sets
    `PIP_CONSTRAINT=/etc/pip/constraint.txt`, which pins
    `typing-extensions==4.12.2`. Installing playwright pulls that pin in and
    breaks `google-genai`, which needs >=4.14 for TypedDict `extra_items` --
    so every Gemini call then dies with a `_TypedDictMeta.__new__()` TypeError
    that looks nothing like a dependency problem. Repair with:

        PIP_CONSTRAINT= pip install -U 'typing_extensions>=4.14'
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise SvgRenderError(
            "SVG animation export needs playwright. Install it with:\n"
            "    pip install playwright && playwright install chromium\n"
            "    PIP_CONSTRAINT= pip install -U 'typing_extensions>=4.14'\n"
            f"(import failed: {e})"
        ) from e
    return sync_playwright


def suggest_fps(min_gap_ms: float, samples_per_stage: int = SAMPLES_PER_STAGE,
                floor_fps: int = FLOOR_FPS, ceil_fps: int = CEIL_FPS) -> int:
    """Pick FPS from the animation's own granularity instead of a fixed guess."""
    if not min_gap_ms or min_gap_ms <= 0:
        return floor_fps
    fps = math.ceil((samples_per_stage * 1000) / min_gap_ms)
    return max(floor_fps, min(ceil_fps, fps))


_TIMELINE_JS = """() => {
    const animations = document.getAnimations();
    if (animations.length === 0) return { totalDurationMs: 0, minGapMs: 0 };
    let maxEnd = 0;
    const eventTimes = [];
    animations.forEach(a => {
        const t = a.effect.getComputedTiming();
        const delay = t.delay || 0;
        const duration = t.duration || 0;
        // For an infinite loop, `duration` is one cycle, so the settled point
        // is delay + one cycle rather than Infinity.
        const end = delay + duration;
        if (isFinite(end)) maxEnd = Math.max(maxEnd, end);
        eventTimes.push(delay, end);
        (a.effect.getKeyframes() || []).forEach(kf => {
            if (typeof kf.computedOffset === 'number') {
                eventTimes.push(delay + kf.computedOffset * duration);
            }
        });
    });
    const uniq = [...new Set(eventTimes.filter(t => isFinite(t)))].sort((a, b) => a - b);
    let minGap = Infinity;
    for (let i = 1; i < uniq.length; i++) {
        const gap = uniq[i] - uniq[i - 1];
        if (gap > %(tol)d) minGap = Math.min(minGap, gap);
    }
    return {
        totalDurationMs: maxEnd,
        minGapMs: isFinite(minGap) ? minGap : 0,
        stageTimes: uniq,
    };
}""" % {"tol": DEDUPE_TOLERANCE_MS}

_FINGERPRINT_JS = """(t) => {
    const animations = document.getAnimations();
    animations.forEach(a => { a.pause(); a.currentTime = t; });
    const parts = [];
    animations.forEach((a, idx) => {
        const target = a.effect.target;
        const propNames = new Set();
        (a.effect.getKeyframes() || []).forEach(kf => {
            Object.keys(kf).forEach(k => {
                if (!['offset', 'easing', 'composite', 'computedOffset'].includes(k)) {
                    propNames.add(k);
                }
            });
        });
        const style = getComputedStyle(target);
        [...propNames].sort().forEach(prop => {
            const cssProp = prop.replace(/[A-Z]/g, m => '-' + m.toLowerCase());
            let value = style.getPropertyValue(cssProp);
            // SVG geometry properties (x/y/width/height) are not always
            // surfaced through getComputedStyle; fall back to the attribute.
            if (!value) value = target.getAttribute(prop) || '';
            const asNum = parseFloat(value);
            if (!isNaN(asNum) && String(asNum) === value.replace(/px$/, '')) {
                value = asNum.toFixed(%(round)d);
            }
            parts.push(idx + ':' + prop + '=' + value);
        });
    });
    return parts.join('|');
}""" % {"round": FINGERPRINT_ROUND_DECIMALS}

_SEEK_JS = """(t) => {
    document.getAnimations().forEach(a => { a.pause(); a.currentTime = t; });
}"""

_SIZE_JS = """() => {
    const svg = document.querySelector('svg');
    if (!svg) return null;
    const vb = svg.viewBox && svg.viewBox.baseVal;
    return {
        width: (vb && vb.width) || svg.clientWidth || 1000,
        height: (vb && vb.height) || svg.clientHeight || 260,
    };
}"""



# How close two keyframe boundaries may be before they are treated as one.
# CSS "instant cut" pairs (`0%, 3.33% { ... }`) and rounding noise put stops
# milliseconds apart; sampling between those yields a frame at a transition
# rather than inside a state. Matches DEDUPE_TOLERANCE_MS, which the timeline
# already uses for the same reason when computing minGapMs.
BOUNDARY_EPSILON_MS = 10.0


def state_midpoints(timeline: dict, total_ms: float) -> list[float]:
    """Timestamps at the MIDDLE of each animation state, in ms.

    WHY NOT A UNIFORM GRID. Sampling every 1/fps seconds lands on the exact
    instants where CSS transitions begin and end -- which is precisely where a
    fading element is at opacity 0. Measured on a real narrated animation: 21
    of 22 exported frames contained no subtitle at all, because each banner
    faded out on the same tick the next faded in, and the grid hit every one of
    those ticks. The animation was correct; the sampling was not.

    Instead, take every keyframe percentage across every animation as a state
    BOUNDARY, and sample halfway between consecutive boundaries. A midpoint is
    by construction inside a state and never on a transition, so whatever is
    meant to be visible then is fully visible.

    This also fixes the sibling problem for motion styles: a sliding box's
    in-transit positions are their own states, so they get sampled rather than
    skipped, and hopping (which genuinely has no in-transit state) is
    unaffected. It is the same trick used by the reference extractors.
    """
    # `stageTimes` is already every animation's delay, end, and per-keyframe
    # offset, in ms, de-duplicated and sorted -- exactly the state boundaries.
    stops = timeline.get("stageTimes") or []
    times = sorted({0.0, float(total_ms)} |
                   {float(x) for x in stops if 0.0 <= float(x) <= total_ms})
    bounds: list[float] = []
    for ms in times:
        if not bounds or ms - bounds[-1] > BOUNDARY_EPSILON_MS:
            bounds.append(ms)
    if len(bounds) < 2:
        return []
    return [(bounds[i] + bounds[i + 1]) / 2.0 for i in range(len(bounds) - 1)]


def render_svg_frames(svg_source: str, out_dir: Path, fps: int | None = None,
                      scale: float = 2.0, max_frames: int = 600,
                      base_dir: Path | None = None,
                      style: str | None = None,
                      steps_dir: Path | None = None,
                      n_steps: int | None = None) -> tuple[list[Path], int]:
    """Render an animated SVG to PNG frames.

    Returns `(frame_paths, fps)`. `fps` is echoed back because it may be
    derived from the animation rather than supplied.

    `scale` is a device pixel ratio rather than a DPI: the SVG's viewBox sets
    the logical size, and 2x keeps text legible in the video without changing
    the layout.

    `base_dir` is where RELATIVE references in `svg_source` should resolve
    from. It matters whenever the source did not come from `out_dir`: this
    function writes its wrapper HTML into `out_dir` and points the browser
    there, so a document authored beside its assets and rendered into a
    scratch directory has every relative path silently rebased.

    That is not hypothetical. The hand-authored reference animations live in
    `reference/animation/` and load crops as `../rasters/<name>.png`. Rendered
    into `reference/_frames/<style>_<target>/`, those became
    `reference/_frames/rasters/<name>.png` -- a directory that does not exist.
    Chromium does not fail on a missing image; it lays out the empty box and
    paints nothing, so every raster vanished from the reference videos and the
    render still reported success.

    Pass the source document's own directory and the `<base>` tag restores the
    author's intent. Omit it (both pipeline callers do, since generated code
    embeds its rasters as data: URIs) and behaviour is unchanged.
    """
    sync_playwright = _require_playwright()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("frame*.png"):
        stale.unlink()

    base = (f'<base href="file://{Path(base_dir).resolve()}/">'
            if base_dir is not None else "")
    html = _HTML_WRAPPER.format(svg=svg_source, base=base)
    html_path = out_dir / "_animation.html"
    html_path.write_text(html, encoding="utf-8")

    frames: list[Path] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(device_scale_factor=scale)
            page.goto(f"file://{html_path.resolve()}")

            size = page.evaluate(_SIZE_JS)
            if not size:
                raise SvgRenderError("no <svg> element in the document")
            width, height = int(size["width"]), int(size["height"])
            page.set_viewport_size({"width": width, "height": height})

            timeline = page.evaluate(_TIMELINE_JS)
            total = float(timeline.get("totalDurationMs") or 0)
            if total <= 0:
                # A static document: one frame is the whole animation.
                shot = out_dir / "frame-1.png"
                page.screenshot(path=str(shot),
                                clip={"x": 0, "y": 0, "width": width, "height": height})
                return [shot], fps or FLOOR_FPS

            if fps is None:
                fps = suggest_fps(float(timeline.get("minGapMs") or 0))

            step = 1000.0 / fps
            wanted = int(total / step) + 1
            count = min(wanted, max_frames)
            if count < wanted:
                # SILENT TRUNCATION, made loud. `count` caps the number of
                # sampled INSTANTS, not the number of frames kept, so hitting
                # the cap does not shorten the deck -- it stops the timeline
                # early and the deck simply ends mid-animation. Measured: a
                # 41s animation sampled at 30fps stops at 20s and yields 20 of
                # its 41 stages, with the tail missing and nothing to say so.
                # It reads as a short animation, not a truncated one.
                print(f"    ! svg_frames: {wanted} samples needed at {fps}fps "
                      f"for {total:.0f}ms but max_frames={max_frames}; "
                      f"only the first {count * step:.0f}ms will be sampled")

            # STATE-MIDPOINT SAMPLING, per style. A uniform grid lands on the
            # instants where one element fades out and the next fades in, which
            # is where both are near opacity 0 -- measured at 21 of 22 frames
            # with no subtitle on a correctly-authored animation. Sampling the
            # middle of each state cannot hit a transition. Sliding
            # additionally asks for in-transit frames, without which a slide
            # exports as rest positions only and is indistinguishable from a
            # hop. See export/svg_style_frames.py.
            from .svg_style_frames import sample_times_for

            picked = sample_times_for(style or "", timeline, total)
            if picked and len(picked) <= max_frames:
                sample_times = picked
            else:
                sample_times = [min(i * step, total) for i in range(count)]

            seen: set[str] = set()
            index = 0
            for t in sample_times:
                fingerprint = page.evaluate(_FINGERPRINT_JS, t)
                # Skip a timestamp that resolves to a state already captured --
                # long plateaus between stages otherwise emit dozens of
                # identical frames.
                if fingerprint and fingerprint in seen:
                    continue
                seen.add(fingerprint)
                index += 1
                shot = out_dir / f"frame-{index}.png"
                page.evaluate(_SEEK_JS, t)
                page.screenshot(path=str(shot),
                                clip={"x": 0, "y": 0, "width": width, "height": height})
                frames.append(shot)

            # A SECOND, STEP-ALIGNED DECK.
            #
            # The deck above samples every distinct animation STATE, which is
            # what the video judges and a human viewer want -- and for a slide
            # that is deliberately many more frames than there are timesteps.
            # But SSS, GPS and NAS each need "the frame for step i", and
            # `animatebench.frames.step_frames` accepts only n or n+1 frames
            # precisely so it never has to guess which of several frames a step
            # owns. A guessed frame yields a band that is wrong while looking
            # entirely plausible, and nothing downstream could detect it.
            #
            # So rather than loosen that guard, emit exactly n_steps frames
            # here, each sampled at the MIDPOINT of its own timestep's slice of
            # the timeline. Exact by construction: step i's frame comes from
            # step i's state, and the per-step judges keep their 1:1 mapping.
            if steps_dir and n_steps and n_steps > 0:
                steps_dir = Path(steps_dir)
                steps_dir.mkdir(parents=True, exist_ok=True)
                slice_ms = total / n_steps
                for i in range(n_steps):
                    t_mid = min((i + 0.5) * slice_ms, total)
                    page.evaluate(_SEEK_JS, t_mid)
                    page.screenshot(
                        path=str(steps_dir / f"frame-{i + 1}.png"),
                        clip={"x": 0, "y": 0, "width": width, "height": height})
        finally:
            browser.close()

    return frames, fps or FLOOR_FPS
