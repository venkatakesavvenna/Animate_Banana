"""Render diagram code to a PNG, whichever target it is written in.

One dispatch point so callers -- the stage-1c critic, the annotation tool's
render endpoint, the inspectors -- stop importing `compile_tikz` directly and
stop being TikZ-only by accident.

Both backends return the same `CompileResult`, so a caller only picks the
target and reads `ok` / `png_path` / `log` exactly as before.
"""
from __future__ import annotations

from pathlib import Path

from ..viewer.compile import CompileResult


def render_code(code: str, target: str, cache_dir: Path, dpi: int = 150,
                timeout: int = 60) -> CompileResult:
    """Rasterize `code` for `target` ("tikz" | "svg")."""
    if target == "tikz":
        # Imported lazily: compile_tikz shells out to latexmk, and keeping it
        # out of module scope means config validation never touches it.
        from ..viewer.compile import compile_tikz
        return compile_tikz(code, Path(cache_dir), dpi=dpi, timeout=timeout)
    if target == "svg":
        from .svg_render import render_svg
        return render_svg(code, Path(cache_dir), dpi=dpi, timeout=timeout)
    raise ValueError(f"unknown target '{target}' (expected 'tikz' or 'svg')")
