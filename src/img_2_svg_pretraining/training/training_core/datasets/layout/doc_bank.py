from typing import Callable, Dict, List
from datasets import Dataset, DatasetDict
import copy
import re
import json
from PIL import Image
from matplotlib import category
from tqdm import tqdm

from collections import defaultdict
from img_2_svg_pretraining.training.training_core import train
from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry, DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, DataArguments
from img_2_svg_pretraining.training.training_core.datasets.layout.utils import get_paragraph_order, visualise_input_dataset, visualise_promptable_dataset, save_image_with_bboxes

from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import format_source_sam, sam_collator

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

DATASET_PATH=os.path.expanduser("/root/.cache/huggingface/hub/datasets--liminghao1630--DocBank/snapshots/783cc7351436c27c64e4f2910c667d50d8e53070")
DATASET_REGISTRY_NAME = "docbank"
mapping = {}
LAYOUT_CLASSES=[]

def get_layout_from_string(decoded_string: str) -> List[str]:
    """
    Extracts all layout class names inside <l>...</l> tags.
    To get the class_names from the string decoded by the VLM/LLM.
    """
    return re.findall(r"<l>(.*?)</l>", decoded_string)

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
        cat = entry["category"]
        box = entry["bbox"]

        # temp = f"<l>{cat}</l><coords>[{box}]</coords>___ [SEG] ___ "
        
        temp = f"<l>{cat}</l>___ [SEG] ___ "
        ret_str += temp

        boxes.append(box)
        category_names.append(cat)

    return ret_str, boxes, category_names

def format_source(source: Dict, get_sam_source_fn: Callable, sam_image_size: int, debug_path:str = "", train:bool = True, val: bool = False, localization_mode=None):
    """
    from the dataset item.. we have to get it into this format
    {
    "image": x[image_path],
    "annotations":{
                    "bbox_list": [{
                        "bbox": [x1, y1, width, height],
                        "category": category_name
                    }] 
                }
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
        if width > 6000 or height > 6000:
            raise ValueError(f"Image size ({width}x{height}) exceeds maximum allowed dimension of 3000 pixels")
        
        if len(bboxes) > 50: # 25 per image
            raise ValueError("Too Many Layout Elements [SEG]") 

    original_image_path = source['image']
    if train:
        original_image = Image.open(original_image_path).convert("RGB")        # Open the Image in PIL format
    else:
        original_image = Image.open(original_image_path).convert("RGB")        # Open the Image in PIL format

    check_and_raise_error(original_image, list_boxes)

    # Can we format source function such that, we give the classes in the prompt itself.
    new_source = {
        "image": original_image,
        "conversations":[
            {
            "from": "human", 
            "value": f"""<image>\nGive me the layout of the document, use the following classes {LAYOUT_CLASSES}"""},
            {"from": "gpt",
            "value": ret_str
            }
        ]
    }

    sam_specific_source = get_sam_source_fn(original_image, list_boxes, is_mask=False, image_size=sam_image_size)
    
    num_seg_tokens = ret_str.count("[SEG]")
    num_masks = len(sam_specific_source["masks"])

    if num_seg_tokens != num_masks:    
        raise ValueError(f"Number of [SEG] tokens in the answer string is not equal to number of masks obtained from SAM source function. Number of [SEG] tokens: {num_seg_tokens}, Number of masks: {num_masks}, full example: {source}")
    
    if debug_path:
        visualise_promptable_dataset(
            copy.deepcopy(original_image),
            debug_path, 
            sam_specific_source["masks"], 
            question=f"Give me the layout of the document, use the following classes {LAYOUT_CLASSES}",
            ret_str=ret_str,
        )
    return new_source, sam_specific_source

def convert_promptable_to_huggingface(path: str = DATASET_PATH):
    """

    """
    global mapping, LAYOUT_CLASSES
    full_data = {}
    
    path = os.path.join(path, "500K_test.json")
    with open(path, "r") as f:
        data = json.load(f)

    for i in range(len(data['categories'])):
        mapping[data['categories'][i]['id']] = data['categories'][i]['name']
    LAYOUT_CLASSES = list(mapping.values())

    for i in range(len(data['images'])):
        full_data[data['images'][i]["id"]] = {"image": os.path.join(DATASET_PATH, "DocBank_500K_ori_img", data['images'][i]["file_name"]),
                                                "annotations": []
                                            }

    missing_regions = 0
    for i in range(len(data['annotations'])):
        try:
            image_id = data['annotations'][i]['image_id']
            bbox = data['annotations'][i]['bbox']
            category_id = data['annotations'][i]['category_id']
            category_name = mapping[category_id]
            
            full_data[image_id]["annotations"].append({
                "bbox": bbox,
                "category": category_name
            })
        except Exception as e:
            print("Exception", e)
            missing_regions += 1

    train_records = []


    for sample in full_data.keys():
        x = full_data[sample]
        train_records.append(x)
    
    print("Missing Regions are", missing_regions)

    return DatasetDict({
        "train": Dataset.from_list(train_records),
        "test": Dataset.from_list(train_records),
    })

@DatasetRegistry.register_dataset(DATASET_REGISTRY_NAME)
def get_promptable_v1_dataargs(get_sam_source_fn, debug_path="", seed=42, sam_image_size=1024, data_path = DATASET_PATH, train=True):
    ds = convert_promptable_to_huggingface(data_path)
    print(len(ds))
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
    data_args = DatasetRegistry.get_dataset(DATASET_REGISTRY_NAME,
                                            get_sam_source_fn = format_source_sam,
                                            debug_path = "/code/debug_output/docbank",
                                            seed=42, sam_image_size=1024)
    
    # importing to autoregister the model
    from ..data_modules.qwen import qwen_data
    def add_new_tokens(tokenizer):
        tokenizer.add_tokens("[SEG]")
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        return tokenizer, seg_token_idx
    
    qwen_data_module: DataModule = DataModuleRegistry.get_module("qwenvl", 
                                        data_args=data_args, change_tokenizer_fn=add_new_tokens, sam_collator=sam_collator, model_name="qwen2.5vl", model_path="Qwen/Qwen2.5-VL-7B-Instruct")

    from .utils import split_train_eval
    train_ds, eval_ds =  split_train_eval(qwen_data_module.Dataloader)
    
    import random

    seed = 42
    rng = random.Random(seed)

    num_samples = min(30, len(train_ds))
    indices = rng.sample(range(len(train_ds)), num_samples)

    for idx in indices:
        x = train_ds[idx]