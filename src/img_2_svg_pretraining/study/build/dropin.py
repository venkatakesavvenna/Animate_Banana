"""Drop-in third arm: a folder of frame-group SVGs plus narration JSON.

The closed-weight baseline (Gemini 3.1 Pro) was generated outside the pipeline
and arrived as a Drive folder:

    <root>/SVG_files and Narrations/SVGs/<StyleDir>/<id>.html   one <g id="frame_N"
                                                               data-time-start=..>
                                                               per step
    <root>/SVG_files and Narrations/Narration_json/<id>.json   [{frame_id,
                                                               data-time-start,
                                                               data-narration}]
    <root>/Video_Frames/<StyleDir>/<id>/frame_000_t01.25s.png  optional, their
                                                               own render

The files animate with SMIL (`<animate attributeName="opacity" begin=.. dur=..
fill="freeze">` inside each `<g id="frame_N" data-time-start=..>`), so the
document has a real timeline. Frames are taken by seeking it: pause the SVG's
animations, `setCurrentTime` to the MIDPOINT of each step's interval, and
screenshot -- which is exactly how their own Video_Frames were produced
(frame_000_t01.25s for a step spanning 0-2.5s), plus one final frame after the
last step's start. Setting `opacity` attributes by hand does not work: the
SMIL animation owns that attribute and overrides it.

Their pre-rendered PNGs are used when present (ground truth for what the
baseline looks like); otherwise the HTML is rendered here by the same rule.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

STYLE_DIRS = {
    "progressive_reveal": "Progressive_Reveal",
    "alpha_masking": "Alpha_Masking",
    "colour_pop": "Colour_Pop",
    "hopping_bounding_box": "Hopping_Bbox",
    "sliding_bounding_box": "Sliding_Bbox",
}

# A frame group's start time: "3.0", "3.0s", "3".
_SECONDS = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


def _seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    m = _SECONDS.search(str(value))
    return float(m.group(1)) if m else 0.0


def find_svg(root: Path, sample_id: str, style: str) -> Path | None:
    d = STYLE_DIRS.get(style)
    if not d:
        return None
    for ext in ("html", "svg"):
        p = root / "SVG_files and Narrations" / "SVGs" / d / f"{sample_id}.{ext}"
        if p.exists():
            return p
    return None


def find_narration(root: Path, sample_id: str) -> Path | None:
    p = root / "SVG_files and Narrations" / "Narration_json" / f"{sample_id}.json"
    return p if p.exists() else None


def prerendered_frames(root: Path, sample_id: str, style: str) -> list[Path]:
    d = STYLE_DIRS.get(style)
    if not d:
        return []
    folder = root / "Video_Frames" / d / sample_id
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.png"))


def frame_groups(svg_text: str) -> list[tuple[str, float]]:
    """(group id, start seconds) for every <g id="frame_N" ...>, in file order."""
    out = []
    for m in re.finditer(r'<g\b[^>]*\bid="(frame_\d+)"[^>]*>', svg_text):
        tag = m.group(0)
        t = re.search(r'data-time-start="([^"]*)"', tag)
        out.append((m.group(1), _seconds(t.group(1)) if t else 0.0))
    # File order is not guaranteed to be time order; the timeline is.
    out.sort(key=lambda x: (x[1], int(x[0].split("_")[1])))
    return out


def load_steps(narration_path: Path | None, svg_text: str) -> list[dict]:
    """One step per frame group: {frame_id, start, narrative}.

    The JSON carries the narration; the SVG carries the same text in
    `data-narration`, so the SVG is the fallback when the JSON is missing or
    disagrees on frame count. Start times come from the SVG when both exist --
    that is what the render is driven by.
    """
    groups = frame_groups(svg_text)
    by_id = {}
    if narration_path:
        try:
            for item in json.loads(narration_path.read_text(encoding="utf-8")):
                by_id[item.get("frame_id")] = item
        except (ValueError, TypeError):
            by_id = {}
    steps = []
    for gid, start in groups:
        text = (by_id.get(gid) or {}).get("data-narration")
        if text is None:
            m = re.search(r'<g\b[^>]*\bid="%s"[^>]*data-narration="([^"]*)"' % gid, svg_text)
            text = m.group(1) if m else ""
        steps.append({"frame_id": gid, "start": start, "narrative": text})
    return steps


def durations(steps: list[dict], tail_seconds: float = 2.5) -> list[float]:
    """Hold of each step from consecutive start times; the last one holds
    for `tail_seconds` (their own renders end 2.5s after the final start)."""
    out = []
    for i, s in enumerate(steps):
        if i + 1 < len(steps):
            out.append(max(0.5, steps[i + 1]["start"] - s["start"]))
        else:
            out.append(tail_seconds)
    return out


def sample_times(steps: list[dict], tail_seconds: float = 2.5) -> list[float]:
    """Midpoint of each step's interval, then one frame after the last start."""
    starts = [st["start"] for st in steps]
    times = []
    for i, t in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else t + tail_seconds
        times.append((t + nxt) / 2.0)
    times.append(starts[-1] + tail_seconds + 0.1)
    return times


def render_frames(svg_path: Path, out_dir: Path, width: int = 1024,
                  tail_seconds: float = 2.5) -> list[Path]:
    """Seek the SVG's SMIL timeline and screenshot each step's midpoint."""
    from playwright.sync_api import sync_playwright   # only here: build-time dep

    out_dir.mkdir(parents=True, exist_ok=True)
    svg_text = svg_path.read_text(encoding="utf-8")
    steps = load_steps(None, svg_text)
    if not steps:
        return []
    m = re.search(r'viewBox="([^"]+)"', svg_text)
    vw, vh = 800.0, 600.0
    if m:
        parts = m.group(1).replace(",", " ").split()
        if len(parts) == 4:
            vw, vh = float(parts[2]), float(parts[3])
    height = int(round(width * vh / vw))
    html = ("<!doctype html><html><head><style>html,body{margin:0;background:#fff}"
            "svg{display:block;width:%dpx;height:%dpx}</style></head><body>%s</body></html>"
            % (width, height, svg_text))
    paths = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.set_content(html)
        page.evaluate("document.querySelector('svg').pauseAnimations()")
        for i, t in enumerate(sample_times(steps, tail_seconds)):
            page.evaluate("t => { const s = document.querySelector('svg');"
                          " s.setCurrentTime(t); }", t)
            page.wait_for_timeout(30)
            p = out_dir / f"frame_{i:03d}_t{t:05.2f}s.png"
            page.locator("svg").first.screenshot(path=str(p))
            paths.append(p)
        browser.close()
    return paths
