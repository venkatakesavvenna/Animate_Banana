from itertools import islice
from typing import Callable, Dict, List
from datasets import Dataset, load_dataset
import copy
import re

from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry, DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, DataArguments
from img_2_svg_pretraining.training.training_core.datasets.layout.utils import get_paragraph_order, visualise_input_dataset

from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import format_source_sam, sam_collator

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

DATASET_PATH="ds4sd/DocLayNet"
# DATASET_PATH="docling-project/DocLayNet"
mapping = {
    "0": "caption",
    "1": "footnote", 
    "2": "formula",
    "3": "list-item",
    "4": "page-footer",
    "5": "page-header",
    "6": "picture",
    "7": "section-header",
    "8": "table",
    "9": "text",
    "10": "title",
}   
LAYOUT_CLASSES = list(mapping.values())


def _normalize_annotations(source: Dict) -> List[Dict]:
    if "objects" in source:
        return source["objects"]

    if "bboxes" in source and "category_id" in source:
        segmentations = source.get("segmentation") or [None] * len(source["bboxes"])
        areas = source.get("area") or [None] * len(source["bboxes"])
        return [
            {
                "bbox": bbox,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": area,
            }
            for bbox, category_id, segmentation, area in zip(
                source["bboxes"],
                source["category_id"],
                segmentations,
                areas,
            )
        ]

    raise KeyError("DocLayNet sample does not contain 'objects' or 'bboxes'/'category_id' fields")

def obtain_layout_str(annotations: List[Dict]):
    bboxes = []
    # Extract raw bbox list
    bboxes = [ann["bbox"] for ann in annotations]

    # Use your grouping + ordering function
    grouped = get_paragraph_order(bboxes)

    # Flatten back to a single list of boxes in correct reading order
    sorted_boxes = [box for group in grouped for box in group]

    # Now map each sorted box back to its annotation entry
    # (assumes boxes are unique, which is true in layout datasets)
    bbox_to_ann = {tuple(ann["bbox"]): ann for ann in annotations}
    sorted_ann = [bbox_to_ann[tuple(b)] for b in sorted_boxes]

    # ---- Build output string as before ----
    ret_str = ""
    boxes = []
    category_names = []

    for entry in sorted_ann:
        cat = mapping[str(entry["category_id"])]
        box = entry["bbox"]

        # temp = f"<l>{cat}</l><coords>[{box}]</coords>___ [SEG] ___ "
        temp = f"<l>{cat}</l>___ [SEG] ___ "
        ret_str += temp

        boxes.append(box)
        category_names.append(cat)
        # category_ids.append(str(entry["category_id"]))

    return ret_str, boxes, category_names

def get_layout_from_string(decoded_string: str) -> List[str]:
    """
    Extracts all layout class names inside <l>...</l> tags.
    To get the class_names from the string decoded by the VLM/LLM.
    """
    return re.findall(r"<l>(.*?)</l>", decoded_string)

def check_and_raise_error(image):
    """
    If image is path then open the pil image. If image is directly pil image.
    Then check the image size.. if either side is > 3000 then raise an error
    """
    from PIL import Image
    
    # If image is a path, open it
    if isinstance(image, str):
        image = Image.open(image)
    
    # Check image dimensions
    width, height = image.size
    if width > 3000 or height > 3000:
        raise ValueError(f"Image size ({width}x{height}) exceeds maximum allowed dimension of 3000 pixels")

def format_source(source: Dict, get_sam_source_fn: Callable, sam_image_size: int, debug_path:str = "", train=True, localization_mode=None):
    """
    from the dataset item.. we have to get it into this format

    doclaynet Input Format:
    {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=596x794 at 0x7FFB5874B4C0>, 
    'id': 262349, 'annotations': [
        {'segmentation': [[48.81, 744.16, 48.81, 434.61, 59.46, 434.61, 59.46, 744.16, 48.81, 744.16]], 'area': 3296.304103927134, 'iscrowd': 0, 'image_id': 262349, 'bbox': [48.81, 434.61, 10.65, 309.55], 'category_id': 1, 'id': 2557263}, 
        {'segmentation': [[61.46, 48.22, 552.18, 48.22, 552.18, 748.23, 61.46, 748.23, 61.46, 48.22]], 'area': 343511.327655, 'iscrowd': 0, 'image_id': 262349, 'bbox': [61.46, 48.22, 490.72, 700.01], 'category_id': 4, 'id': 2557264}
        ]
    }

    Required Output Format:
    {
      "image": "demo/images/10095.png",
      "conversations": [
        {
          "from": "human",
          "value": "<image>\nPlease give the layout of the document"
        },
        {
          "from": "gpt",
          "value": "Yes"
        }
      ]
    }
    """
    annotations = _normalize_annotations(source)

    ret_str, list_boxes, category_names = obtain_layout_str(annotations)

    def check_and_raise_error(image):
        """
        If image is path then open the pil image. If image is directly pil image.
        Then check the image size.. if either side is > 3000 then raise an error
        """
        from PIL import Image
        
        # If image is a path, open it
        if isinstance(image, str):
            image = Image.open(image)
        
        # Check image dimensions
        width, height = image.size
        if width > 3000 or height > 3000:
            raise ValueError(f"Image size ({width}x{height}) exceeds maximum allowed dimension of 3000 pixels")

    check_and_raise_error(source['image'])

    new_source = {
        "image": source['image'],
        "conversations":[
            {"from": "human", 
            "value": f"<image>\nGive me the layout of the document, use the following classes {LAYOUT_CLASSES}"},
            {"from": "gpt",
            "value": ret_str
            }
        ]
    }

    original_image = source['image']
    sam_specific_source = get_sam_source_fn(original_image, list_boxes, is_mask=False, image_size=sam_image_size)
    
    if debug_path:
        visualise_input_dataset(copy.deepcopy(original_image),debug_path, sam_specific_source["masks"], category_names)

    return new_source, sam_specific_source

def _load_doclaynet_split(data_path: str, split_name: str, streaming: bool, sample_limit: int | None):
    if streaming:
        if sample_limit is None or sample_limit < 2:
            raise ValueError("streaming DocLayNet runs require sample_limit >= 2")
        streamed = load_dataset(data_path, split=split_name, streaming=True)
        rows = list(islice(streamed, sample_limit))
        if len(rows) < 2:
            raise ValueError(f"Only {len(rows)} rows available from {data_path}:{split_name}")
        return Dataset.from_list(rows)

    ds = load_dataset(data_path)
    if split_name in ds:
        return ds[split_name]
    if split_name == "test" and "validation" in ds:
        return ds["validation"]
    raise KeyError(f"Split '{split_name}' not found in dataset {data_path}. Available: {list(ds.keys())}")


@DatasetRegistry.register_dataset("doclaynet")
def get_doclaynet_dataargs(
    get_sam_source_fn,
    debug_path="",
    seed=42,
    sam_image_size=1024,
    data_path=DATASET_PATH,
    train=True,
    streaming=False,
    sample_limit=None,
):
    split_name = "train" if train else "test"
    ds = _load_doclaynet_split(
        data_path=data_path,
        split_name=split_name,
        streaming=streaming,
        sample_limit=sample_limit,
    )
    get_source_kwargs = {
        "get_sam_source_fn": get_sam_source_fn,
        "sam_image_size": sam_image_size,
        "debug_path": debug_path,
    }
    if not train:
        get_source_kwargs["train"] = False

    return DataArguments(
        dataset_path=data_path,
        ds=ds,
        seed=seed,
        get_source=format_source,
        get_source_kwargs=get_source_kwargs,
        extract_from_labels_fn=get_layout_from_string,
        layout_classes=LAYOUT_CLASSES,
    )


# Below part of code is just to run this file independently for debugging purposes
if __name__ == "__main__":
    data_args = DatasetRegistry.get_dataset("doclaynet",
                                            get_sam_source_fn = format_source_sam,
                                            debug_path = "/projects/data/vision-team/srihari_bandarupalli/DocGrounding/src/debug_outputs/dataset_input_debug",
                                            seed=42, sam_image_size=1024)
    
    # importing to autoregister the model
    from ..data_modules.qwen import qwen_data
    def add_new_tokens(tokenizer):
        tokenizer.add_tokens("[SEG]")
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        return tokenizer, seg_token_idx
    
    ds = load_dataset(DATASET_PATH)
    qwen_data_module: DataModule = DataModuleRegistry.get_module("qwen3vl", 
                                        data_args=data_args, change_tokenizer_fn=add_new_tokens, sam_collator = sam_collator)

    # from .utils import split_train_eval
    # train_ds, eval_ds =  split_train_eval(qwen_data_module.Dataset)
    # train_ds[0]
