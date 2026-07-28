"""dvqa — Bar chart QA dataset (Kafle et al.).

Registry key: dvqa

Source: https://github.com/kushalkafle/DVQA_dataset
HF mirror (if available): abhilashamangal/DVQA

Download:
    from img_2_svg_pretraining.training.training_core.datasets.public.dvqa import download
    download("/data/dvqa")
"""
import os
from typing import Callable, Dict

from datasets import load_from_disk
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec, DataArguments

KEY = "dvqa"


def download(data_path: str, n_procs: int = 1) -> None:
    """Download DVQA to *data_path* via the HuggingFace Hub.

    If the dataset is not available on HF Hub, you must obtain it manually:
        1. Download from https://github.com/kushalkafle/DVQA_dataset
        2. Prepare a HF Dataset from the raw files and call
           ``dataset.save_to_disk(data_path)``.
    """
    import datasets as hf_datasets

    try:
        ds = hf_datasets.load_dataset("abhilashamangal/DVQA", trust_remote_code=True)
        ds.save_to_disk(data_path)
        print(f"[dvqa] Saved to {data_path}")
    except Exception as exc:
        raise RuntimeError(
            "DVQA is not available on HF Hub. "
            "Download from https://github.com/kushalkafle/DVQA_dataset "
            f"and set data_path to the prepared directory. Original error: {exc}"
        ) from exc


def get_source(
    source: Dict,
    get_sam_source_fn: Callable,
    sam_image_size: int,
    debug_path: str = "",
    localization_mode=None,
):
    image = source["image"]
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected PIL Image, got {type(image)}.")

    question = source.get("question") or ""
    answer = source.get("answer") or ""

    source_dict = {
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": str(answer)},
        ],
    }
    sam_specific_dict = get_sam_source_fn(image, [], is_mask=False, image_size=sam_image_size)
    return source_dict, sam_specific_dict


@DatasetRegistry.register_dataset(KEY)
def get_dvqa_dataargs(
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
            f"Run img_2_svg_pretraining.training.training_core.datasets.public.dvqa.download(data_path) first."
        )

    ds_dict = load_from_disk(data_path)
    split = "train" if train else "validation"
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
