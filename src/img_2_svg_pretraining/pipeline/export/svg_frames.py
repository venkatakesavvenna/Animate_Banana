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
<html><head><meta charset="utf-8"><style>
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


def render_svg_frames(svg_source: str, out_dir: Path, fps: int | None = None,
                      scale: float = 2.0, max_frames: int = 600) -> tuple[list[Path], int]:
    """Render an animated SVG to PNG frames.

    Returns `(frame_paths, fps)`. `fps` is echoed back because it may be
    derived from the animation rather than supplied.

    `scale` is a device pixel ratio rather than a DPI: the SVG's viewBox sets
    the logical size, and 2x keeps text legible in the video without changing
    the layout.
    """
    sync_playwright = _require_playwright()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("frame*.png"):
        stale.unlink()

    html = _HTML_WRAPPER.format(svg=svg_source)
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
            count = min(int(total / step) + 1, max_frames)

            seen: set[str] = set()
            index = 0
            for i in range(count):
                t = min(i * step, total)
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
        finally:
            browser.close()

    return frames, fps or FLOOR_FPS
