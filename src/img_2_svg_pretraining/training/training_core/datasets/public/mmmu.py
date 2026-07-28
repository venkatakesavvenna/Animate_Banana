"""mmmu — MMMU/MMMU massive multi-discipline multimodal understanding dataset.

Registry key: mmmu

Loads one or more named subjects via the `subjects` kwarg. Rows with image=None
are filtered out. Default: first 5 subjects for tractability.
"""
from itertools import islice
from typing import Callable, Dict, List, Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec, DataArguments

HUB_ID = "MMMU/MMMU"

_DEFAULT_SUBJECTS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
]

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _format_options(options):
    """Return newline-separated lettered options string."""
    lines = []
    for i, opt in enumerate(options):
        letter = _LETTERS[i] if i < len(_LETTERS) else str(i)
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


def _load_subjects(subjects: List[str], split: str, streaming: bool, sample_limit):
    parts = []
    per_subject_limit = (
        max(1, sample_limit // max(len(subjects), 1)) if sample_limit is not None else None
    )
    for subject in subjects:
        if streaming:
            streamed = load_dataset(HUB_ID, name=subject, split=split, streaming=True)
            rows = list(islice(streamed, per_subject_limit))
            part = Dataset.from_list(rows)
        else:
            part = load_dataset(HUB_ID, name=subject, split=split)
            if per_subject_limit is not None:
                part = part.select(range(min(per_subject_limit, len(part))))
        parts.append(part)
    if not parts:
        return Dataset.from_list([])
    return concatenate_datasets(parts)


def get_source(
    source: Dict,
    get_sam_source_fn: Callable,
    sam_image_size: int,
    debug_path: str = "",
    localization_mode=None,
):
    image = source.get("image")
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"Expected PIL Image, got {type(image)}. "
            "Filter non-PIL samples before calling get_source."
        )

    question = source.get("question") or ""
    answer = source.get("answer") or ""
    options = source.get("options") or []

    if options:
        question_text = question + "\n" + _format_options(options)
    else:
        question_text = question

    source_dict = {
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question_text}"},
            {"from": "gpt", "value": answer},
        ],
    }
    sam_specific_dict = get_sam_source_fn(image, [], is_mask=False, image_size=sam_image_size)
    return source_dict, sam_specific_dict


@DatasetRegistry.register_dataset("mmmu")
def get_mmmu_dataargs(
    get_sam_source_fn,
    seed=42,
    sam_image_size=1024,
    train=True,
    streaming=False,
    sample_limit=None,
    localization_mode=None,
    subjects: Optional[List[str]] = None,
    **dataset_specific_kwargs,
):
    if subjects is None:
        subjects = _DEFAULT_SUBJECTS

    split = "validation" if train else "test"
    ds = _load_subjects(subjects, split, streaming, sample_limit)
    ds = ds.filter(lambda x: x["image"] is not None)

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
