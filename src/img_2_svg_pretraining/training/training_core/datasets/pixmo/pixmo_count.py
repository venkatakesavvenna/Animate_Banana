"""pixmo_count — allenai/pixmo-count counting dataset.

Registry key: pixmo_count
Splits: train, validation, test

Train split: "Count and point to all {label} in the image."
  Answer includes count + Molmo-format point tokens.

Val/test splits: "How many {label} are in the image?"
  Answer is the count only.
"""
from itertools import islice
from typing import Callable, Dict, List, Optional

from datasets import Dataset, load_dataset
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec, DataArguments

HUB_ID = "allenai/pixmo-count"


def _load_split(
    hub_id: str,
    split: str,
    streaming: bool,
    sample_limit: Optional[int],
):
    if streaming:
        streamed = load_dataset(hub_id, split=split, streaming=True)
        rows = list(islice(streamed, sample_limit))
        return Dataset.from_list(rows)
    ds = load_dataset(hub_id, split=split)
    if sample_limit is not None:
        ds = ds.select(range(min(sample_limit, len(ds))))
    return ds


def _format_molmo_points(xs: List[float], ys: List[float], label: str) -> str:
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
    is_train: bool = True,
):
    image = source["image"]
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"Expected PIL Image, got {type(image)}. "
            "Filter non-PIL samples before calling get_source."
        )

    label = source.get("label") or ""
    count = source.get("count")
    count_str = str(count) if count is not None else "0"

    if is_train:
        points = source.get("points") or {}
        xs = points.get("x") or []
        ys = points.get("y") or []
        point_str = _format_molmo_points(xs, ys, label)
        question = f"Count and point to all {label} in the image."
        answer = f"{count_str}\n{point_str}" if point_str else count_str
    else:
        question = f"How many {label} are in the image?"
        answer = count_str

    source_dict = {
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }
    sam_specific_dict = get_sam_source_fn(image, [], is_mask=False, image_size=sam_image_size)
    return source_dict, sam_specific_dict


@DatasetRegistry.register_dataset("pixmo_count")
def get_pixmo_count_dataargs(
    get_sam_source_fn,
    seed=42,
    sam_image_size=1024,
    train=True,
    streaming=False,
    sample_limit=None,
    localization_mode=None,
):
    if train:
        split = "train"
        is_train = True
    else:
        # prefer validation split; fall back to test
        try:
            split = "validation"
            _load_split(HUB_ID, split, streaming=False, sample_limit=1)
        except Exception:
            split = "test"
        is_train = False

    ds = _load_split(HUB_ID, split, streaming, sample_limit)
    return DataArguments(
        dataset_path=HUB_ID,
        ds=ds,
        seed=seed,
        get_source=get_source,
        get_source_kwargs={
            "get_sam_source_fn": get_sam_source_fn,
            "sam_image_size": sam_image_size,
            "is_train": is_train,
            "localization_mode": localization_mode,
        },
        annotation_spec=AnnotationSpec(
            has_localization=True,
            localization_type="point",
        ),
    )
