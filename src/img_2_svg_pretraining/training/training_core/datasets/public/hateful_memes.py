"""hateful_memes — Hateful Memes Challenge dataset (Meta AI).

Registry key: hateful_memes

Source: https://ai.meta.com/tools/hatefulmemes/
Requires signing a Data Use Agreement (DUA) before downloading.

Fields: image (path str), text (str), label (int: 0=not hateful, 1=hateful)

Download:
    Hateful Memes requires signing the DUA at https://ai.meta.com/tools/hatefulmemes/
    After downloading, convert to a HF Dataset and save with ``dataset.save_to_disk(data_path)``.

    Alternatively call:
        from img_2_svg_pretraining.training.training_core.datasets.public.hateful_memes import download
        download("/data/hateful_memes")
    which will raise NotImplementedError with instructions.
"""
import os
from typing import Callable, Dict

from datasets import load_from_disk
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec, DataArguments

KEY = "hateful_memes"


def download(data_path: str, n_procs: int = 1) -> None:
    """Raises NotImplementedError — Hateful Memes requires a signed DUA."""
    raise NotImplementedError(
        "Hateful Memes requires signing the DUA at https://ai.meta.com/tools/hatefulmemes/. "
        "After obtaining the data, convert it to a HuggingFace Dataset and call "
        "``dataset.save_to_disk(data_path)``."
    )


def get_source(
    source: Dict,
    get_sam_source_fn: Callable,
    sam_image_size: int,
    debug_path: str = "",
    localization_mode=None,
):
    image = source.get("image")
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected PIL Image, got {type(image)}.")

    text = source.get("text") or ""
    label = source.get("label")
    answer = "Yes" if label == 1 else "No"

    source_dict = {
        "image": image,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nIs this meme hateful? Image text: {text}",
            },
            {"from": "gpt", "value": answer},
        ],
    }
    sam_specific_dict = get_sam_source_fn(image, [], is_mask=False, image_size=sam_image_size)
    return source_dict, sam_specific_dict


@DatasetRegistry.register_dataset(KEY)
def get_hateful_memes_dataargs(
    get_sam_source_fn,
    seed=42,
    sam_image_size=1024,
    train=True,
    sample_limit=None,
    localization_mode=None,
    data_path=None,
):
    if data_path is None:
        raise ValueError(f"data_path is required for {KEY}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{KEY} dataset not found at {data_path}. "
            "Hateful Memes requires signing the DUA at https://ai.meta.com/tools/hatefulmemes/. "
            "After obtaining the data, convert it to a HF Dataset and call "
            "dataset.save_to_disk(data_path)."
        )

    ds_dict = load_from_disk(data_path)
    split = "train" if train else "dev_seen"
    if hasattr(ds_dict, "keys") and split in ds_dict:
        ds = ds_dict[split]
    elif hasattr(ds_dict, "keys"):
        ds = ds_dict[list(ds_dict.keys())[0]]
    else:
        ds = ds_dict

    if sample_limit is not None:
        ds = ds.select(range(min(sample_limit, len(ds))))

    return DataArguments(
        dataset_path=data_path,
        ds=ds,
        seed=seed,
        get_source=get_source,
        get_source_kwargs={
            "get_sam_source_fn": get_sam_source_fn,
            "sam_image_size": sam_image_size,
        },
        annotation_spec=AnnotationSpec(has_localization=False),
    )
