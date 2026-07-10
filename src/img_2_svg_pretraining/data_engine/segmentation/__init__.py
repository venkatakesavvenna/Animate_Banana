"""Segmentation stage: point/image -> per-node masks.

One module per method (see models.py for the model registry):
- sam3_tracker.py -- Sam3Runner, point-promptable per-object segmentation (facebook/sam3)
- sam2_amg.py     -- Sam2AmgRunner, automatic mask generation, no points needed (facebook/sam2.1-hiera-large)
- classical_cv.py -- point-seeded flood-fill + contour cross-check, no ML/GPU (see its docstring for current status)
"""
from __future__ import annotations

from img_2_svg_pretraining.data_engine.segmentation.sam3_tracker import MaskResult, Sam3Runner, save_mask

__all__ = ["MaskResult", "Sam3Runner", "save_mask"]
