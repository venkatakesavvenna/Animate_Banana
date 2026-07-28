"""pixmo_cap — allenai/pixmo-cap captioning dataset.

Registry key: pixmo_cap

Modes (controlled via get_source_kwargs["mode"]):
  "captions"             — use the caption field as the answer
  "transcripts"          — use the first element of the transcripts list
  "transcript_and_caption" — caption + "\\n\\n" + first transcript
"""
from itertools import islice
from typing import Callable, Dict

from datasets import Dataset, load_dataset
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec, DataArguments

HUB_ID = "allenai/pixmo-cap"


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
    mode: str = "captions",
):
    image = source["image"]
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"Expected PIL Image, got {type(image)}. "
            "Filter non-PIL samples before calling get_source."
        )

    caption = source.get("caption") or ""
    transcripts = source.get("transcripts") or []
    first_transcript = transcripts[0] if transcripts else ""

    if mode == "transcripts":
        answer = first_transcript
    elif mode == "transcript_and_caption":
        answer = caption + "\n\n" + first_transcript
    else:
        # default: "captions"
        answer = caption

    source_dict = {
        "image": image,
        "conversations": [
            {"from": "human", "value": "<image>\nDescribe this image in detail."},
            {"from": "gpt", "value": answer},
        ],
    }
    sam_specific_dict = get_sam_source_fn(image, [], is_mask=False, image_size=sam_image_size)
    return source_dict, sam_specific_dict


@DatasetRegistry.register_dataset("pixmo_cap")
def get_pixmo_cap_dataargs(
    get_sam_source_fn,
    seed=42,
    sam_image_size=1024,
    train=True,
    streaming=False,
    sample_limit=None,
    localization_mode=None,
    mode="captions",
):
    # pixmo-cap only has a train split
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
            "mode": mode,
        },
        annotation_spec=AnnotationSpec(has_localization=False),
    )
