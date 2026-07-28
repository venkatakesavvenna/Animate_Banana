"""aokvqa — HuggingFaceM4/A-OKVQA augmented outside knowledge VQA dataset.

Registry key: aokvqa
"""
from itertools import islice
from typing import Callable, Dict

from datasets import Dataset, load_dataset
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec, DataArguments

HUB_ID = "HuggingFaceM4/A-OKVQA"

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _format_choices(choices):
    """Return newline-separated lettered choices string."""
    lines = []
    for i, choice in enumerate(choices):
        letter = _LETTERS[i] if i < len(_LETTERS) else str(i)
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


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

    question = source.get("question") or ""
    choices = source.get("choices") or []
    correct_choice_idx = source.get("correct_choice_idx")
    direct_answers = source.get("direct_answers") or []

    if choices:
        question_text = question + "\n" + _format_choices(choices)
        if correct_choice_idx is not None and correct_choice_idx < len(choices):
            answer = choices[correct_choice_idx]
        elif direct_answers:
            answer = direct_answers[0]
        else:
            answer = ""
    else:
        question_text = question
        answer = direct_answers[0] if direct_answers else ""

    source_dict = {
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question_text}"},
            {"from": "gpt", "value": answer},
        ],
    }
    sam_specific_dict = get_sam_source_fn(image, [], is_mask=False, image_size=sam_image_size)
    return source_dict, sam_specific_dict


@DatasetRegistry.register_dataset("aokvqa")
def get_aokvqa_dataargs(
    get_sam_source_fn,
    seed=42,
    sam_image_size=1024,
    train=True,
    streaming=False,
    sample_limit=None,
    localization_mode=None,
    **dataset_specific_kwargs,
):
    split = "train" if train else "validation"
    ds = _load_split(HUB_ID, split, streaming, sample_limit)

    return DataArguments(
        dataset_path=HUB_ID,
        ds=ds,
        seed=seed,
        get_source=get_source,
        get_source_kwargs={
            "get_sam_source_fn": get_sam_source_fn,
            "sam_image_size": sam_image_size,
        },
        annotation_spec=AnnotationSpec(has_localization=False),
    )
