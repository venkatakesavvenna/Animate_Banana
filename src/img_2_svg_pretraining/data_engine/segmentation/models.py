"""Model registry for the segmentation stage.

- `sam3` (`facebook/sam3`): point-promptable per-object segmentation via
  `Sam3TrackerModel`/`Sam3TrackerProcessor` -- see sam3_tracker.py.
- `sam2-amg` (`facebook/sam2.1-hiera-large`): automatic mask generation
  (image in, masks out, no point/text prompts needed) -- see sam2_amg.py.

Unlike `benchmark/models.py` (whose repo ids were confirmed to exist on the
HF Hub before being pinned), the repo ids below were supplied directly by
the user / discovered during exploration and have not been independently
re-verified here beyond confirming they load in this container.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sam3Spec:
    name: str
    hf_repo: str
    trust_remote_code: bool = False
    dtype: str = "bfloat16"


@dataclass
class Sam2Spec:
    name: str
    hf_repo: str
    trust_remote_code: bool = False


SEGMENTATION_MODELS: dict[str, Sam3Spec] = {
    "sam3": Sam3Spec(
        name="sam3",
        hf_repo="facebook/sam3",
    ),
}

AMG_MODELS: dict[str, Sam2Spec] = {
    "sam2-amg": Sam2Spec(
        name="sam2-amg",
        hf_repo="facebook/sam2.1-hiera-large",
    ),
}


def get_segmentation_model(name: str) -> Sam3Spec:
    if name not in SEGMENTATION_MODELS:
        raise KeyError(f"Unknown segmentation model '{name}'. Known: {sorted(SEGMENTATION_MODELS)}")
    return SEGMENTATION_MODELS[name]


def get_amg_model(name: str) -> Sam2Spec:
    if name not in AMG_MODELS:
        raise KeyError(f"Unknown AMG model '{name}'. Known: {sorted(AMG_MODELS)}")
    return AMG_MODELS[name]
