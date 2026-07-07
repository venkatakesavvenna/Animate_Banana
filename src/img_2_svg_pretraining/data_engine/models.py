"""Model registry for the data engine's pointing and segmentation stages.

Separate from `img_2_svg_pretraining.benchmark.models.ModelSpec` because
those models are chat-style image-text-to-text VLMs (image + text prompt ->
text), while pointing/segmentation models have different input/output shapes
(image (+point prompts) -> points/masks) and are loaded via different
`transformers`/model-specific APIs. The edge-discovery and codegen stages
reuse `benchmark.models.MODELS` directly instead of duplicating a registry
here -- see edges.py/codegen.py.

Two pointing models are registered, both selectable via `--pointing-model`
in run_pointing.py/run_data_engine.py (see pointing.py, which dispatches
between their two different output-parsing APIs):
- `molmo-point-8b` (`allenai/MolmoPoint-8B`): pointing-specialist checkpoint;
  points are decoded from special tokens via `model.extract_image_points(...)`
  with per-request metadata from the processor.
- `molmo2-8b` (`allenai/Molmo2-8B`): general-purpose Molmo2 VLM that also
  supports pointing (trained on point/track data among other tasks); points
  are embedded as plain text (`<points ... coords="idx x y ..."/>`) in
  normal generation output, decoded with a regex, no special metadata needed.
  Simpler integration, and may generalize better on diagrams whose nodes
  aren't drawn as clean geometric shapes (MolmoPoint's specialist training
  skews towards natural-image objects).

Unlike `benchmark/models.py` (whose repo ids were confirmed to exist on the
HF Hub before being pinned), the repo ids below were supplied directly by the
user and have NOT been independently verified here -- confirm they load
correctly during first use inside the container.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Molmo2Spec:
    name: str
    hf_repo: str
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"
    dtype: str = "bfloat16"


@dataclass
class Sam3Spec:
    name: str
    hf_repo: str
    trust_remote_code: bool = False
    dtype: str = "bfloat16"


POINTING_MODELS: dict[str, Molmo2Spec] = {
    "molmo-point-8b": Molmo2Spec(
        name="molmo-point-8b",
        hf_repo="allenai/MolmoPoint-8B",
    ),
    "molmo2-8b": Molmo2Spec(
        name="molmo2-8b",
        hf_repo="allenai/Molmo2-8B",
    ),
}

SEGMENTATION_MODELS: dict[str, Sam3Spec] = {
    "sam3": Sam3Spec(
        name="sam3",
        hf_repo="facebook/sam3",
    ),
}


def get_pointing_model(name: str) -> Molmo2Spec:
    if name not in POINTING_MODELS:
        raise KeyError(f"Unknown pointing model '{name}'. Known: {sorted(POINTING_MODELS)}")
    return POINTING_MODELS[name]


def get_segmentation_model(name: str) -> Sam3Spec:
    if name not in SEGMENTATION_MODELS:
        raise KeyError(f"Unknown segmentation model '{name}'. Known: {sorted(SEGMENTATION_MODELS)}")
    return SEGMENTATION_MODELS[name]
