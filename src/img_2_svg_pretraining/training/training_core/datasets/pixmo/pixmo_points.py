"""pixmo_points — allenai/pixmo-points pointing dataset.

Registry key: pixmo_points

Points are stored in Molmo's [0, 100] coordinate space.
Answer is serialized as space-separated Molmo-format point tokens:
  <point x=X.X y=Y.Y alt=LABEL/>

Both localization_mode="sam" (default) and "autoregressive" use the same
text-only Molmo point format, since pointing does not produce SAM masks.
"""
from itertools import islice
from typing import Callable, Dict, List

from datasets import Dataset, load_dataset
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec, DataArguments

HUB_ID = "allenai/pixmo-points"


def _load_split(hub_id: str, split: str, streaming: bool, sample_limit):
    if streaming:
        streamed = load_dataset(hub_id, split=split, streaming=True)
        rows = list(islice(streamed, sample_limit))
        return Dataset.from_list(rows)
    ds = load_dataset(hub_id, split=split)
    if sample_limit is not None:
        ds = ds.select(range(min(sample_limit, len(ds))))
    return ds


def _format_molmo_points(xs: List[float], ys: List[float], label: str) -> str:
    """Serialize points as Molmo-format tokens. Coords are in [0, 100] space."""
    tokens = [
        f'<point x="{x:.1f}" y="{y:.1f}" alt="{label}"/>'
        for x, y in zip(xs, ys)
    ]
    return " ".join(tokens)


def get_source(
    source: Dict,
    get_sam_source_fn: Callable,
    sam_image_size: int,
    debug_path: str = "",
    localization_mode=None,
):
    image = source["image"]
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"Expected PIL Image, got {type(image)}. "
            "Filter non-PIL samples before calling get_source."
        )

    label = source.get("label") or ""
    points = source.get("points") or {}
    xs = points.get("x") or []
    ys = points.get("y") or []

    question = f"Point to all instances of {label} in the image."
    answer = _format_molmo_points(xs, ys, label)

    source_dict = {
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }
    # Points don't produce SAM masks — always pass empty box list.
    sam_specific_dict = get_sam_source_fn(image, [], is_mask=False, image_size=sam_image_size)
    return source_dict, sam_specific_dict


@DatasetRegistry.register_dataset("pixmo_points")
def get_pixmo_points_dataargs(
    get_sam_source_fn,
    seed=42,
    sam_image_size=1024,
    train=True,
    streaming=False,
    sample_limit=None,
    localization_mode=None,
):
    split = "train"
    ds = _load_split(HUB_ID, split, streaming, sample_limit)
    return DataArguments(
        dataset_path=HUB_ID,
        ds=ds,
        seed=seed,
        get_source=get_source,
        get_source_kwargs={
            "get_sam_source_fn": get_sam_source_fn,
            "sam_image_size": sam_image_size,
            "localization_mode": localization_mode,
        },
        annotation_spec=AnnotationSpec(
            has_localization=True,
            localization_type="point",
        ),
    )
