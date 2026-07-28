from typing import Callable, Dict, List
from datasets import load_dataset
import copy
import re

from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry, DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, DataArguments
from img_2_svg_pretraining.training.training_core.datasets.layout.utils import get_paragraph_order, visualise_input_dataset

from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import format_source_sam, sam_collator

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

DATASET_PATH="ai4bharat/indicdlp"
mapping = {
    "0": "advertisement",
    "1": "answer",
    "2": "author",
    "3": "chapter-title",
    "4": "contact-info",
    "5": "dateline",
    "6": "figure",
    "7": "figure-caption",
    "8": "first-level-question",
    "9": "flag",
    "10": "folio",
    "11": "footer",
    "12": "footnote",
    "13": "formula",
    "14": "header",
    "15": "headline",
    "16": "index",
    "17": "jumpline",
    "18": "options",
    "19": "ordered-list",
    "20": "page-number",
    "21": "paragraph",
    "22": "placeholder-text",
    "23": "quote",
    "24": "reference",
    "25": "second-level-question",
    "26": "section-title",
    "27": "sidebar",
    "28": "sub-headline",
    "29": "sub-ordered-list",
    "30": "sub-section-title",
    "31": "subsub-ordered-list",
    "32": "subsub-section-title",
    "33": "sub-unordered-list",
    "34": "subsub-headline",
    "35": "subsub-unordered-list",
    "36": "table",
    "37": "table-caption",
    "38": "table-of-contents",
    "39": "third-level-question",
    "40": "unordered-list",
    "41": "website-link",
} 

LAYOUT_CLASSES = list(mapping.values())

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

def format_source(source: Dict, get_sam_source_fn: Callable, sam_image_size: int, debug_path:str = "", localization_mode=None):
    """
    from the dataset item.. we have to get it into this format

    indicdlp Input Format:
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
    instances = [
        {"bbox": bbox, "category_id": cid}
        for bbox, cid in zip(source["bboxes"], source["category_ids"])
    ]
    ret_str, list_boxes, category_names = obtain_layout_str(instances)

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

    def check_and_raise_error(image, bboxes):
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
        if width > 4000 or height > 4000:
            raise ValueError(f"Image size ({width}x{height}) exceeds maximum allowed dimension of 3000 pixels")
        
        if len(bboxes) > 50: # 25 per image
            raise ValueError("Too Many Layout Elements [SEG]") 

    check_and_raise_error(source['image'], bboxes=list_boxes)
    
    original_image = source['image']

    sam_specific_source = get_sam_source_fn(original_image, list_boxes, is_mask=False, image_size=sam_image_size)
    
    if debug_path:
        visualise_input_dataset(copy.deepcopy(original_image), debug_path, sam_specific_source["masks"], category_names)

    return new_source, sam_specific_source

@DatasetRegistry.register_dataset("indicdlp")
def get_indicdlp_dataargs(get_sam_source_fn, debug_path="", seed=42, sam_image_size=1024, data_path = DATASET_PATH, train=True):
    ds = load_dataset(data_path)
    if train:
        data_args = DataArguments(
                                dataset_path=data_path, 
                                ds = ds["train"], 
                                seed = seed, 
                                get_source = format_source, 
                                get_source_kwargs={
                                    "get_sam_source_fn": get_sam_source_fn,
                                    "sam_image_size": sam_image_size,
                                    "debug_path": debug_path
                                    # pass debug_path="" or remove it for not printing the inputs.
                                },
                                extract_from_labels_fn = get_layout_from_string,
                                layout_classes = LAYOUT_CLASSES
                                )
    else:
        try:
            data_args = DataArguments(
                                    dataset_path=data_path, 
                                    ds = ds["test"], 
                                    seed = seed, 
                                    get_source = format_source, 
                                    get_source_kwargs={
                                        "get_sam_source_fn": get_sam_source_fn,
                                        "sam_image_size": sam_image_size,
                                        "debug_path": debug_path
                                        # pass debug_path="" or remove it for not printing the inputs.
                                    },
                                    extract_from_labels_fn = get_layout_from_string,
                                    layout_classes = LAYOUT_CLASSES
                                    )
        except Exception as e:
            data_args = DataArguments(
                                    dataset_path=data_path, 
                                    ds = ds["validation"], 
                                    seed = seed, 
                                    get_source = format_source, 
                                    get_source_kwargs={
                                        "get_sam_source_fn": get_sam_source_fn,
                                        "sam_image_size": sam_image_size,
                                        "debug_path": debug_path
                                        # pass debug_path="" or remove it for not printing the inputs.
                                    },
                                    extract_from_labels_fn = get_layout_from_string,
                                    layout_classes = LAYOUT_CLASSES
                                    )
            print("Error is", e)
    return data_args


# Below part of code is just to run this file independently for debugging purposes
if __name__ == "__main__":
    data_args = DatasetRegistry.get_dataset("indicdlp",
                                            get_sam_source_fn = format_source_sam,
                                            debug_path = "/code/debug_output/dataset_input_debug",
                                            seed=42, sam_image_size=1024)
    
    # importing to autoregister the model
    from ..data_modules.qwen import qwen_data
    def add_new_tokens(tokenizer):
        tokenizer.add_tokens("[SEG]")
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        return tokenizer, seg_token_idx
    
    ds = load_dataset(DATASET_PATH)

    qwen_data_module: DataModule = DataModuleRegistry.get_module("qwenvl", 
                                        data_args=data_args, change_tokenizer_fn=add_new_tokens, sam_collator = sam_collator,
                                        model_name="qwen2.5vl", model_path="Qwen/Qwen2.5-VL-7B-Instruct")

    from .utils import split_train_eval
    train_ds, eval_ds =  split_train_eval(qwen_data_module.Dataloader)

    for batch_idx in range(len(train_ds)):
        x = train_ds[batch_idx]