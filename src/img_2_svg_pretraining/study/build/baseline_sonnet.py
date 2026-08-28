"""End-to-end baseline for Experiment 4: figure -> animated SVG in one shot.

The point of this arm is that it is NOT the pipeline. AnimateBanana converts,
parses, sequences, critiques, narrates, designs and critiques again; this asks a
single model for the finished artifact in one call. Comparing them is what makes
RQ4 a question about the pipeline rather than about the underlying model.

    python -m img_2_svg_pretraining.study.build.baseline_sonnet \
        --config src/img_2_svg_pretraining/pipeline/configs/bench_v3_or_svg.yaml \
        --only <id>:<style>,<id>:<style> --out data/baseline_sonnet

Coordinates: Claude Sonnet 5 is in the high-resolution vision tier -- 2576px on
the long edge, and coordinates map 1:1 to image pixels with no scale-factor
math. Every figure in animatebench_v3 is under that (max long edge 2069), so
each is sent at native resolution and the model's coordinates are directly
usable as SVG user units. We therefore hand it the true pixel dimensions and
ask for a viewBox that matches, rather than rescaling anything.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path

import urllib.request

from img_2_svg_pretraining.pipeline.samples import discover_samples
from img_2_svg_pretraining.pipeline.styles import STYLES

OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-5"

SYSTEM = """You produce a single self-contained animated SVG that explains a \
scientific figure.

You are given the figure as an image. Reproduce it faithfully as SVG, then \
animate it so the animation walks a viewer through the figure.

Hard requirements:
- Output ONE complete <svg> document and nothing else. No markdown fence, no \
commentary before or after.
- Set width, height and a viewBox matching the pixel dimensions you are told. \
Coordinates you choose are pixels in that space; place elements where they \
actually appear in the image.
- Animate with CSS @keyframes inside a <style> element in the SVG. Do not use \
SMIL (<animate>), JavaScript, external files, or web fonts.
- Every animation must have a finite total duration and must not loop \
(animation-fill-mode: forwards, no `infinite`).
- Use only text, shapes and paths. Do not embed raster images.
- The finished state must show the complete figure."""

PROMPT = """This figure is {w} x {h} pixels. Use exactly that viewBox.

Rebuild it as an animated SVG in the following style.

ANIMATION STYLE: {style}
{style_desc}

The animation should step through the figure in a sensible order, spending \
roughly {step_secs} seconds per step, for about {total_secs} seconds in total. \
End with the whole figure visible."""


def encode_image(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return mime, base64.b64encode(path.read_bytes()).decode()


def extract_svg(text: str) -> str | None:
    """Pull the SVG out, tolerating a stray fence or preamble.

    The prompt forbids both, but a hard failure on an otherwise-good response
    would silently shrink the baseline arm and make the comparison look better
    than it is.

    Returns None on a response truncated before `</svg>`. That is a genuine
    generation failure and belongs in RQ4's success statistics -- salvaging a
    half-written SVG would put a mangled figure in front of a participant and
    score the baseline on our repair job rather than on its own output.
    """
    m = re.search(r"<svg\b.*?</svg>", text, re.S | re.I)
    if m:
        return m.group(0)
    if re.search(r"<svg\b", text, re.I):
        print("    (response contained <svg> but no closing tag - truncated)")
    return None


def call_model(image: Path, width: int, height: int, style: str,
               style_desc: str, api_key: str, timeout: int = 900) -> dict:
    mime, b64 = encode_image(image)
    body = {
        "model": MODEL,
        "max_tokens": 64000,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": PROMPT.format(
                    w=width, h=height, style=style.replace("_", " "),
                    style_desc=style_desc, step_secs=3, total_secs=40)},
            ]},
        ],
    }
    req = urllib.request.Request(
        OPENROUTER, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.load(r)
    if "error" in payload:
        raise RuntimeError(payload["error"].get("message", "unknown error"))
    return {"text": payload["choices"][0]["message"]["content"],
            "usage": payload.get("usage", {}),
            "seconds": round(time.time() - started, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/animatebench_v3")
    ap.add_argument("--only", required=True,
                    help="comma-separated <sample_id>:<style> pairs")
    ap.add_argument("--out", default="data/baseline_sonnet")
    ap.add_argument("--render", action="store_true",
                    help="also render frames (needs playwright chromium)")
    args = ap.parse_args()

    api_key = os.environ.get("OPEN_ROUTER_KEY")
    if not api_key:
        print("OPEN_ROUTER_KEY is not set", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = {s.id: s for s in discover_samples(Path(args.dataset))}

    from PIL import Image
    ok = failed = 0
    for spec in args.only.split(","):
        sid, _, style = spec.strip().partition(":")
        sample = samples.get(sid)
        if sample is None:
            print(f"  SKIP {sid}: not in dataset")
            failed += 1
            continue
        width, height = Image.open(sample.image_path).size
        desc = STYLES[style].description if style in STYLES else ""

        cell = out / f"{sid}__{style}"
        cell.mkdir(parents=True, exist_ok=True)
        svg_path = cell / "animation.svg"
        if (cell / "raw_response.txt").exists() and not svg_path.exists():
            (cell / "raw_response.txt").unlink()   # retrying; drop the stale failure
        if svg_path.exists():
            print(f"  CACHED {sid} [{style}]")
            ok += 1
            continue

        print(f"  {sid} [{style}] {width}x{height} …", flush=True)
        try:
            result = call_model(sample.image_path, width, height, style, desc, api_key)
        except Exception as exc:                                   # noqa: BLE001
            print(f"    FAIL: {exc}")
            failed += 1
            continue

        svg = extract_svg(result["text"])
        if not svg:
            (cell / "raw_response.txt").write_text(result["text"], encoding="utf-8")
            print("    FAIL: no <svg> in the response (raw kept)")
            failed += 1
            continue

        svg_path.write_text(svg, encoding="utf-8")
        (cell / "meta.json").write_text(json.dumps({
            "sample_id": sid, "animation_style": style, "model": MODEL,
            "width": width, "height": height,
            "usage": result["usage"], "seconds": result["seconds"],
        }, indent=1), encoding="utf-8")
        print(f"    ok  {len(svg)} chars, {result['seconds']}s, "
              f"{result['usage'].get('completion_tokens','?')} out-tokens")
        ok += 1

        if args.render:
            from img_2_svg_pretraining.pipeline.export.svg_frames import render_svg_frames
            try:
                frames, fps = render_svg_frames(svg, cell / "frames", fps=2)
                print(f"    rendered {len(frames)} frames @ {fps}fps")
            except Exception as exc:                               # noqa: BLE001
                print(f"    render FAILED: {exc}")

    print(f"\n{ok} ok, {failed} failed -> {out}")
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
