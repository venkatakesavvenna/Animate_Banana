"""Rasterize a static SVG document to PNG for preview.

The SVG counterpart to `viewer/compile.py::compile_tikz`, and deliberately
signature- and result-compatible with it: same `CompileResult`, same
content-hash caching, same "return ok=False with the reason" contract rather
than raising. That compatibility is the point -- it lets every caller become
target-aware with a one-line dispatch instead of a rewrite.

Scope is *static* SVG. The animation designer emits CSS `@keyframes`, and
cairosvg has no animation engine and no way to seek a timeline, so animated
frames come from the browser path in `export/svg_render.py` instead. What
this renders is the document's initial state, which is what the diagram
screens want anyway.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..viewer.compile import CompileResult, _source_hash


def render_svg(svg_source: str, cache_dir: Path, dpi: int = 150,
               timeout: int = 60) -> CompileResult:
    """Rasterize an SVG source string to PNG, caching by content hash.

    `timeout` is accepted for signature parity with `compile_tikz` and is
    unused: cairosvg is in-process, so there is no subprocess to bound.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = _source_hash(svg_source)
    png_path = cache_dir / f"{digest}.png"
    if png_path.exists():
        return CompileResult(ok=True, png_path=png_path, log="")

    # Parse first, so malformed markup reports a precise location instead of
    # whatever cairosvg happens to raise. This is also the "compile log" the
    # critic's repair path gets to work from, so it has to be specific.
    try:
        ET.fromstring(svg_source)
    except ET.ParseError as e:
        return CompileResult(ok=False, png_path=None,
                             log=f"SVG is not well-formed XML: {e}")

    try:
        import cairosvg
    except ImportError as e:
        return CompileResult(ok=False, png_path=None,
                             log=f"cairosvg is not installed: {e}")

    try:
        # unsafe=True permits `<image href="/abs/path.png">` to load from
        # disk. Stage 1b splices raster crops in as absolute local paths, so
        # without this every filled raster renders blank -- verified: 6.8 KB
        # of empty boxes versus 45 KB with the imagery.
        #
        # The flag also allows external URL fetches, which is why it is off by
        # default. It is acceptable here because the only documents this
        # renders are ones the pipeline just generated into its own cache, not
        # untrusted input.
        cairosvg.svg2png(bytestring=svg_source.encode("utf-8"),
                         write_to=str(png_path), dpi=dpi, unsafe=True)
    except Exception as e:
        # cairosvg raises a wide range of types for bad geometry, unresolved
        # hrefs and unsupported features; the caller only needs the reason.
        png_path.unlink(missing_ok=True)
        return CompileResult(ok=False, png_path=None,
                             log=f"{type(e).__name__}: {e}")

    if not png_path.exists() or png_path.stat().st_size == 0:
        png_path.unlink(missing_ok=True)
        return CompileResult(ok=False, png_path=None,
                             log="cairosvg produced no output")
    return CompileResult(ok=True, png_path=png_path, log="")
