"""pixmo_docs — allenai/pixmo-docs document understanding dataset.

Registry keys:
  pixmo_docs_charts    — name="charts"
  pixmo_docs_diagrams  — name="diagrams"
  pixmo_docs_tables    — name="tables"
  pixmo_docs_other     — name="other"

Splits: train, validation, test

Each example has an image and a questions dict with parallel "question"
and "answer" lists.  get_source picks the first Q/A pair.
"""
from itertools import islice
from typing import Callable, Dict, Optional

from datasets import Dataset, load_dataset
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec, DataArguments

HUB_ID = "allenai/pixmo-docs"

_SUBSET_KEYS = {
    "pixmo_docs_charts": "charts",
    "pixmo_docs_diagrams": "diagrams",
    "pixmo_docs_tables": "tables",
    "pixmo_docs_other": "other",
}


def _load_split(
    hub_id: str,
    name: str,
    split: str,
    streaming: bool,
    sample_limit: Optional[int],
):
    if streaming:
        streamed = load_dataset(hub_id, name=name, split=split, streaming=True)
        rows = list(islice(streamed, sample_limit))
        return Dataset.from_list(rows)
    ds = load_dataset(hub_id, name=name, split=split)
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

    questions_dict = source.get("questions") or {}
    question_list = questions_dict.get("question") or []
    answer_list = questions_dict.get("answer") or []

    question = question_list[0] if question_list else ""
    answer = answer_list[0] if answer_list else ""

    source_dict = {
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }
    sam_specific_dict = get_sam_source_fn(image, [], is_mask=False, image_size=sam_image_size)
    return source_dict, sam_specific_dict


def _make_factory(subset_name: str):
    """Return a DatasetRegistry factory function for the given subset."""

    def _factory(
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
        else:
            # Try validation first, fall back to test.
            try:
                _load_split(HUB_ID, subset_name, "validation", streaming=False, sample_limit=1)
                split = "validation"
            except Exception:
                split = "test"

        ds = _load_split(HUB_ID, subset_name, split, streaming, sample_limit)
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

    return _factory


# Register all four subsets.
for _registry_key, _subset_name in _SUBSET_KEYS.items():
    DatasetRegistry.register_dataset(_registry_key)(_make_factory(_subset_name))
