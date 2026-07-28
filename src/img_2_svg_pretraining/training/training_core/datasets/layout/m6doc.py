from typing import Callable, Dict, List
from datasets import load_dataset
import copy
import re
import json
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry, DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, DataArguments
from img_2_svg_pretraining.training.training_core.datasets.layout.utils import get_paragraph_order, visualise_input_dataset

from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import format_source_sam, sam_collator

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

DATASET_PATH="/root/.cache/huggingface/M6_Doc_dataset"
# DATASET_PATH=os.path.expanduser("~/.cache/huggingface/M6_Doc_dataset")

mapping = {
    "0": "_background_",
    "1": "QR code",
    "2": "advertisement",
    "3": "algorithm",
    "4": "answer",
    "5": "author",
    "6": "barcode",
    "7": "bill",
    "8": "blank",
    "9": "bracket",
    "10": "breakout",
    "11": "byline",
    "12": "caption",
    "13": "catalogue",
    "14": "chapter title",
    "15": "code",
    "16": "correction",
    "17": "credit",
    "18": "dateline",
    "19": "drop cap",
    "20": "editor's note",
    "21": "endnote",
    "22": "examinee information",
    "23": "fifth-level title",
    "24": "figure",
    "25": "first-level question number",
    "26": "first-level title",
    "27": "flag",
    "28": "folio",
    "29": "footer",
    "30": "footnote",
    "31": "formula",
    "32": "fourth-level section title",
    "33": "fourth-level title",
    "34": "header",
    "35": "headline",
    "36": "index",
    "37": "inside",
    "38": "institute",
    "39": "jump line",
    "40": "kicker",
    "41": "lead",
    "42": "marginal note",
    "43": "matching",
    "44": "mugshot",
    "45": "option",
    "46": "ordered list",
    "47": "other question number",
    "48": "page number",
    "49": "paragraph",
    "50": "part",
    "51": "play",
    "52": "poem",
    "53": "reference",
    "54": "sealing line",
    "55": "second-level question number",
    "56": "second-level title",
    "57": "section",
    "58": "section title",
    "59": "sidebar",
    "60": "sub section title",
    "61": "subhead",
    "62": "subsub section title",
    "63": "supplementary note",
    "64": "table",
    "65": "table caption",
    "66": "table note",
    "67": "teasers",
    "68": "third-level question number",
    "69": "third-level title",
    "70": "title",
    "71": "translator",
    "72": "underscore",
    "73": "unordered list",
    "74": "weather forecast",
}
LAYOUT_CLASSES = list(mapping.values())

def obtain_layout_str(annotations: List[Dict]):
    bboxes = []
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

def format_source(source: Dict, get_sam_source_fn: Callable, sam_image_size: int, debug_path:str = "", train:bool = True, val: bool = False, localization_mode=None):
    """
    from the dataset item.. we have to get it into this format

    m6doc Input Format:
    {'image_id': <int>,
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=596x794 at 0x7FFB5874B4C0>, 
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
    annotations = source["annotations"]
    ret_str, list_boxes, category_names = obtain_layout_str(annotations)

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

    original_image_path = source['image']["file_name"]
    if train:
        original_image = Image.open(os.path.join(DATASET_PATH, "train2017", original_image_path)).convert("RGB")        # Open the Image in PIL format
    elif val:
        original_image = Image.open(os.path.join(DATASET_PATH, "val2017", original_image_path)).convert("RGB")        # Open the Image in PIL format
    else:
        original_image = Image.open(os.path.join(DATASET_PATH, "test2017", original_image_path)).convert("RGB")        # Open the Image in PIL format

    check_and_raise_error(original_image, list_boxes)

    new_source = {
        "image": original_image,
        "conversations":[
            {"from": "human", 
            "value": f"<image>\nGive me the layout of the document, use the following classes {LAYOUT_CLASSES}"},
            {"from": "gpt",
            "value": ret_str
            }
        ]
    }

    sam_specific_source = get_sam_source_fn(original_image, list_boxes, is_mask=False, image_size=sam_image_size)
    
    if debug_path:
        visualise_input_dataset(copy.deepcopy(original_image),debug_path, sam_specific_source["masks"], category_names)

    return new_source, sam_specific_source

import os, json
from collections import defaultdict
from datasets import Dataset, DatasetDict

from tqdm import tqdm
from collections import defaultdict

def _coco_to_records(coco_dict, desc="Converting COCO"):
    ann_by_image = defaultdict(list)

    for ann in tqdm(coco_dict["annotations"], desc=f"{desc}: annotations"):
        ann_by_image[ann["image_id"]].append(ann)

    records = []
    for img in tqdm(coco_dict["images"], desc=f"{desc}: images"):
        records.append({
            "image_id": img["id"],
            "image": img,
            "annotations": ann_by_image[img["id"]],
        })

    return records

def convert_m6doc_to_huggingface(path: str):
    with open(os.path.join(path, "M6doc_labels", "instances_train2017.json")) as f:
        train_data = json.load(f)

    with open(os.path.join(path, "M6doc_labels", "instances_val2017.json")) as f:
        validation_data = json.load(f)

    with open(os.path.join(path, "M6doc_labels", "instances_test2017.json")) as f:
        test_data = json.load(f)

    train_records = _coco_to_records(train_data)
    validation_records = _coco_to_records(validation_data)
    test_records  = _coco_to_records(test_data)

    return DatasetDict({
        "train": Dataset.from_list(train_records),
        "val": Dataset.from_list(validation_records),
        "test":  Dataset.from_list(test_records),
    })

@DatasetRegistry.register_dataset("m6doc")
def get_m6doc_dataargs(get_sam_source_fn, debug_path="", seed=42, sam_image_size=1024, data_path = DATASET_PATH, train=True):
    ds = convert_m6doc_to_huggingface(data_path)
    if train:
        data_args = DataArguments(
                                dataset_path=data_path, 
                                ds = ds["train"], 
                                seed = seed, 
                                get_source = format_source, 
                                get_source_kwargs={
                                    "get_sam_source_fn": get_sam_source_fn,
                                    "sam_image_size": sam_image_size,
                                    "debug_path": debug_path,
                                    "train": train
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
                                        "debug_path": debug_path,
                                        "train": train
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
    data_args = DatasetRegistry.get_dataset("m6doc",
                                            get_sam_source_fn = format_source_sam,
                                            debug_path = "/code/debug_output/dataset_input_debug_m6doc",
                                            seed=42, sam_image_size=1024)
    
    # importing to autoregister the model
    from ..data_modules.qwen import qwen_data
    def add_new_tokens(tokenizer):
        tokenizer.add_tokens("[SEG]")
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        return tokenizer, seg_token_idx
    
    qwen_data_module: DataModule = DataModuleRegistry.get_module("qwenvl", 
                                        data_args=data_args, change_tokenizer_fn=add_new_tokens, sam_collator = sam_collator, model_name="qwen2.5vl", model_path="Qwen/Qwen2.5-VL-7B-Instruct")

    from .utils import split_train_eval
    train_ds, eval_ds =  split_train_eval(qwen_data_module.Dataloader)

    for batch_idx in range(len(train_ds)):
        x = train_ds[batch_idx]