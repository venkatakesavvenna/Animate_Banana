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

import os, json
from collections import defaultdict
from datasets import Dataset, DatasetDict
from tqdm import tqdm

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

DATASET_PATH="/root/.cache/huggingface/AnnoPage_Dataset"
# DATASET_PATH=os.path.expanduser("~/.cache/huggingface/AnnoPage_Dataset")
mapping = {
    "1": "Chemical formula and equation",
    "2": "Symbol, logo, coat of arms",
    "3": "Exlibris",
    "4": "Photograph",
    "5": "Geometric drawing",
    "6": "Graph",
    "7": "Initial",
    "8": "Caricature and comics",
    "9": "Map",
    "10": "Mathematical expression and equation",
    "11": "Musical notation",
    "12": "Image",
    "13": "Other book decor",
    "14": "Other technical drawing",
    "15": "Decorative inscription",
    "16": "Floor plan",
    "17": "Barcode and QR code",
    "18": "Stamp",
    "19": "Advertisement",
    "20": "Handwritten note",
    "21": "Diagram",
    "22": "Signet",
    "23": "Table",
    "24": "Vignette",
    "25": "Frieze",
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

    annopage Input Format:
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

    original_image_path = source['image']["file_name"]
    if train:
        original_image = Image.open(os.path.join(DATASET_PATH, "images", "train", original_image_path)).convert("RGB")        # Open the Image in PIL format
    elif val:
        original_image = Image.open(os.path.join(DATASET_PATH, "images", "val", original_image_path)).convert("RGB")        # Open the Image in PIL format
    else:
        original_image = Image.open(os.path.join(DATASET_PATH, "images", "test", original_image_path)).convert("RGB")        # Open the Image in PIL format

    sam_specific_source = get_sam_source_fn(original_image, list_boxes, is_mask=False, image_size=sam_image_size)
    
    if debug_path:
        visualise_input_dataset(copy.deepcopy(original_image),debug_path, sam_specific_source["masks"], category_names)

    return new_source, sam_specific_source

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    cx *= img_w
    cy *= img_h
    w  *= img_w
    h  *= img_h

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return [x1, y1, x2, y2]

def _yolo_split_to_records(root, split):
    images_dir = os.path.join(root, "images", split)
    labels_dir = os.path.join(root, "labels", split)

    records = []

    for label_file in tqdm(os.listdir(labels_dir)[:100], desc=f"Loading the {split} set"):
        if not label_file.endswith(".txt"):
            continue

        image_id = label_file.replace(".txt", "")

        # image path (jpg/png)
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(images_dir, image_id + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            continue

        # read image size
        with Image.open(img_path) as img:
            W, H = img.size

        annotations = []
        with open(os.path.join(labels_dir, label_file)) as f:
            for line in f:
                cls, cx, cy, w, h = map(float, line.strip().split())
                bbox = yolo_to_xyxy(cx, cy, w, h, W, H)

                annotations.append({
                    "category_id": int(cls),
                    "bbox": bbox
                })

        records.append({
            "image_id": image_id,
            "image": {
                "file_name": os.path.basename(img_path),
                "width": W,
                "height": H,
            },
            "annotations": annotations
        })

    return records

def convert_annopage_to_huggingface(path: str):
    train_records = _yolo_split_to_records(path, "train")
    test_records  = _yolo_split_to_records(path, "test")

    return DatasetDict({
        "train": Dataset.from_list(train_records),
        "test":  Dataset.from_list(test_records),
    })

@DatasetRegistry.register_dataset("annopage")
def get_annopage_dataargs(get_sam_source_fn, debug_path="", seed=42, sam_image_size=1024, data_path = DATASET_PATH, train=True):
    ds = convert_annopage_to_huggingface(data_path)
    # import pdb; pdb.set_trace()
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
    data_args = DatasetRegistry.get_dataset("annopage",
                                            get_sam_source_fn = format_source_sam,
                                            debug_path = "/code/debug_output/dataset_input_debug_annopage",
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
        # import pdb; pdb.set_trace()