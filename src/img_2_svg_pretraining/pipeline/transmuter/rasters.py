"""Raster placeholder handling, dispatched by target.

`raster_integrator` used to import `tikz_rasters` unconditionally. That did
not fail on an SVG document -- it did something worse: the TikZ regex matched
nothing, `splice` early-returned, and the stage reported "0 placeholders" as
if that were a finding rather than an unimplemented code path. A stage that
quietly does nothing is the failure mode that has cost the most time on this
project, so an unknown target raises here instead.
"""
from __future__ import annotations

from . import svg_rasters, tikz_rasters

_MODULES = {"tikz": tikz_rasters, "svg": svg_rasters}


def module_for(target: str):
    """The raster implementation for `target`, or a clear error."""
    try:
        return _MODULES[target]
    except KeyError:
        raise ValueError(
            f"no raster placeholder support for target '{target}'; "
            f"implemented: {sorted(_MODULES)}"
        ) from None


def find_placeholders(code: str, target: str) -> list:
    return module_for(target).find_placeholders(code)


def splice(code: str, replacements: dict[str, str], target: str) -> tuple[str, list[str]]:
    return module_for(target).splice(code, replacements)


def document_extent(code: str, target: str) -> tuple[float, float] | None:
    """The drawing's coordinate extent in `bbox()`'s units, or None if unknown.

    Needed because a placeholder's box and a detected region's box are in
    different spaces, and only this makes them comparable. See
    `raster_integrator._expand_to_placeholder`.
    """
    return module_for(target).document_extent(code)
