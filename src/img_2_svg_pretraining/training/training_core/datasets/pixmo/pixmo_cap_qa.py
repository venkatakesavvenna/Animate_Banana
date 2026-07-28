"""pixmo_cap_qa — allenai/pixmo-cap-qa QA dataset.

Registry key: pixmo_cap_qa

Each example has a messages list of alternating user/assistant turns.
get_source picks the first user question and assistant answer.
Each message has role and content fields; content is a list of dicts
with type and text keys.
"""
from itertools import islice
from typing import Callable, Dict

from datasets import Dataset, load_dataset
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec, DataArguments

HUB_ID = "allenai/pixmo-cap-qa"


def _load_split(hub_id: str, split: str, streaming: bool, sample_limit):
    if streaming:
        streamed = load_dataset(hub_id, split=split, streaming=True)
        rows = list(islice(streamed, sample_limit))
        return Dataset.from_list(rows)
    ds = load_dataset(hub_id, split=split)
    if sample_limit is not None:
        ds = ds.select(range(min(sample_limit, len(ds))))
    return ds


def _extract_text(message: Dict) -> str:
    """Extract text from a message dict with a content list."""
    content = message.get("content", [])
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
        # fallback: join all text-like entries
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and "text" in item
        ]
        return " ".join(texts)
    if isinstance(content, str):
        return content
    return ""


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

    messages = source.get("messages") or []
    # Find first user message and the immediately following assistant message.
    question = ""
    answer = ""
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role == "user" and not question:
            question = _extract_text(msg)
        elif role == "assistant" and question and not answer:
            answer = _extract_text(msg)
            break

    source_dict = {
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }
    sam_specific_dict = get_sam_source_fn(image, [], is_mask=False, image_size=sam_image_size)
    return source_dict, sam_specific_dict


@DatasetRegistry.register_dataset("pixmo_cap_qa")
def get_pixmo_cap_qa_dataargs(
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
        },
        annotation_spec=AnnotationSpec(has_localization=False),
    )
