"""Model registry for the pointing stage.

Two pointing models are registered, both selectable via `--pointing-model`
in run_data_engine.py (see molmo2.py, which dispatches between their two
different output-parsing APIs):
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


def get_pointing_model(name: str) -> Molmo2Spec:
    if name not in POINTING_MODELS:
        raise KeyError(f"Unknown pointing model '{name}'. Known: {sorted(POINTING_MODELS)}")
    return POINTING_MODELS[name]
