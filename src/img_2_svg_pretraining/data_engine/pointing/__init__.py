"""Pointing stage: node/raster-region discovery via point-prompted VLMs.

One module per method (see models.py for the registry):
- molmo_point.py -- MolmoPointRunner (allenai/MolmoPoint-8B, pointing specialist)
- molmo2.py       -- Molmo2PointRunner (allenai/Molmo2-8B, general-purpose) + make_point_runner dispatcher
- common.py       -- shared PointResult type and node/raster query prompts
"""
from __future__ import annotations

from img_2_svg_pretraining.data_engine.pointing.common import NODE_QUERY, RASTER_QUERY, PointResult
from img_2_svg_pretraining.data_engine.pointing.molmo2 import Molmo2PointRunner, make_point_runner
from img_2_svg_pretraining.data_engine.pointing.molmo_point import MolmoPointRunner

__all__ = [
    "PointResult",
    "NODE_QUERY",
    "RASTER_QUERY",
    "MolmoPointRunner",
    "Molmo2PointRunner",
    "make_point_runner",
]
