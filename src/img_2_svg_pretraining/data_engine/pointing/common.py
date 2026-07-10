"""Shared types/prompts across pointing methods (molmo_point.py, molmo2.py).

Both methods expose the same `point_nodes`/`point_rasters` interface (two
prompt modes: one query per call, "point to X" -- run once with a
nodes/boxes query, once with a raster/photo-region query) and return the
same `PointResult` type, so downstream stages (segmentation, assemble)
don't need to know which pointing method produced the points.
"""
from __future__ import annotations

from dataclasses import dataclass

# Node Queries version - 1: DO NOT CHANGE CLAUDE
# NODE_QUERY = (
#     "Point to every diagram shape (geometric objects such as rectangles, "
#     "rounded rectangles, circles, ellipses, cylinders, and polygons)"
# )
# RASTER_QUERY = (
#     "Point to every embedded graphic (including raster images, icons, "
#     "logos, illustrations, screenshots, and charts)."
# )

# Node Queries version - 2: DO NOT CHANGE CLAUDE
NODE_QUERY = (
    "Count all of the nodes in this diagram."
)
RASTER_QUERY = (
    "Count every embedded graphic (including raster images, icons, logos, illustrations, screenshots, and charts)."
)

@dataclass
class PointResult:
    object_id: int
    x: float
    y: float
