"""pixmo_point_explanations — allenai/pixmo-point-explanations dataset.

Registry key: pixmo_point_explanations

Each example has parallel lists of question, parsed_response, and points.
get_source picks the first element of each list.
parsed_response already contains embedded Molmo-format point annotations,
so it is passed through as-is as the answer.
"""
from itertools import islice
from typing import Callable, Dict

from datasets import Dataset, load_dataset
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec, DataArguments

HUB_ID = "allenai/pixmo-point-explanations"


def _load_split(hub_id: str, split: str, streaming: bool, sample_limit):
    if streaming:
        streamed = load_dataset(hub_id, split=split, streaming=True)
        rows = list(islice(streamed, sample_limit))
        return Dataset.from_list(rows)
    ds = load_dataset(hub_id, split=split)
    if sample_limit is not None:
        ds = ds.select(range(min(sample_limit, len(ds))))
    return ds


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

    questions = source.get("question") or []
    parsed_responses = source.get("parsed_response") or []

    question = questions[0] if questions else ""
    answer = parsed_responses[0] if parsed_responses else ""

    source_dict = {
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }
    sam_specific_dict = get_sam_source_fn(image, [], is_mask=False, image_size=sam_image_size)
    return source_dict, sam_specific_dict


@DatasetRegistry.register_dataset("pixmo_point_explanations")
def get_pixmo_point_explanations_dataargs(
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
