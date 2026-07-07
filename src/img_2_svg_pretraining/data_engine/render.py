"""Backend-dispatching render: tikz -> viewer/compile.py's latexmk+pdftoppm
path (already used by the benchmark harness), svg -> cairosvg (pure-Python,
no LaTeX toolchain needed -- confirmed `libcairo` is present in this
container's base image, just needed `pip install cairosvg`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cairosvg

from img_2_svg_pretraining.viewer.compile import compile_tikz


@dataclass
class RenderResult:
    ok: bool
    png_path: Path | None
    log: str


def _source_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _render_svg(svg_source: str, cache_dir: Path, dpi: int = 150) -> RenderResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = _source_hash(svg_source)
    png_path = cache_dir / f"{digest}.png"
    if png_path.exists():
        return RenderResult(ok=True, png_path=png_path, log="")
    try:
        cairosvg.svg2png(
            bytestring=svg_source.encode("utf-8"),
            write_to=str(png_path),
            dpi=dpi,
        )
        return RenderResult(ok=True, png_path=png_path, log="")
    except Exception as e:  # cairosvg raises a variety of parse/render errors
        return RenderResult(ok=False, png_path=None, log=f"cairosvg render failed: {e}")


def render(code: str, backend: str, cache_dir: Path, dpi: int = 150) -> RenderResult:
    if backend == "tikz":
        result = compile_tikz(code, cache_dir, dpi=dpi)
        return RenderResult(ok=result.ok, png_path=result.png_path, log=result.log)
    if backend == "svg":
        return _render_svg(code, cache_dir, dpi=dpi)
    raise ValueError(f"Unknown backend '{backend}'. Known: tikz, svg")
