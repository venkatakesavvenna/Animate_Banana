from typing import Callable, Dict, List
from datasets import Dataset, DatasetDict
import copy
import re
import json
import numpy as np
from PIL import Image
from matplotlib import category
from tqdm import tqdm

from collections import defaultdict
from img_2_svg_pretraining.training.training_core import train
from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry, DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, DataArguments
from img_2_svg_pretraining.training.training_core.datasets.layout.utils import get_polygon_order, visualise_promptable_dataset

from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import format_source_sam, sam_collator
import os
import xml.etree.ElementTree as ET

os.environ["HF_DATASETS_OFFLINE"] = "1"

DATASET_PATH=os.path.expanduser("/root/.cache/huggingface/PRIMA_DATA")
DATASET_REGISTRY_NAME = "prima"
LAYOUT_CLASSES=[]

def get_layout_from_string(decoded_string: str) -> List[str]:
    """
    Extracts all layout class names inside <l>...</l> tags.
    To get the class_names from the string decoded by the VLM/LLM.
    """
    return re.findall(r"<l>(.*?)</l>", decoded_string)

def _poly_key(polygon) -> tuple:
    """Converts a polygon (list of [x,y] pairs or np.ndarray) to a
    hashable tuple-of-tuples so it can be used as a dict key."""
    arr = np.asarray(polygon, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return tuple(tuple(pt) for pt in arr.tolist())


def obtain_layout_str(annotations: List[Dict]):
    polygons = [ann["bbox"] for ann in annotations]

    # Group and sort polygons in reading order (left-to-right columns,
    # top-to-bottom within each column)
    grouped = get_polygon_order(polygons)

    # Flatten back to a single list of polygons in correct reading order
    sorted_polygons = [poly for group in grouped for poly in group]

    # Map each sorted polygon back to its annotation entry using a
    # hashable tuple-of-tuples key (polygons are lists-of-lists so plain
    # tuple() is not hashable)
    poly_to_ann = {_poly_key(ann["bbox"]): ann for ann in annotations}
    sorted_ann = [poly_to_ann[_poly_key(p)] for p in sorted_polygons]

    # sorted_ann = annotations
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
            question=f"Give me the layout of the document, use the following classes {list(set(category_names))}",
            ret_str=ret_str,
        )
    return new_source, sam_specific_source

def parse_page_xml(xml_path: str):
    """
    Parse a PRImA PAGE XML file and return annotations in
    HuggingFace object detection format.

    Returns:
        annotations: List[Dict]
            [
                {
                    "bbox": ,
                    "category": class_name
                },
                ...
            ]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    namespace = {"ns": root.tag.split("}")[0].strip("{")}

    annotations = []

    region_tags = [
        "TextRegion",
        "ImageRegion",
        "GraphicRegion",
        "SeparatorRegion",
        "TableRegion",
        "ChartRegion",
        "MathsRegion",
        "NoiseRegion",
    ]

    for region_tag in region_tags:
        for region in root.findall(f".//ns:{region_tag}", namespace):

            # Determine class name
            region_type = region.get("type")

            if region_tag == "TextRegion" and region_type:
                class_name = region_type.lower()

            if class_name not in LAYOUT_CLASSES:
                LAYOUT_CLASSES.append(class_name)

            coords = region.find("ns:Coords", namespace)
            if coords is None:
                continue

            points = []
            for point in coords.findall("ns:Point", namespace):
                x = int(point.get("x"))
                y = int(point.get("y"))
                points.append([x, y])

            if not points:
                continue

            annotations.append({
                "bbox": points,
                "category": class_name
            })

    return annotations


def convert_promptable_to_huggingface(path: str = DATASET_PATH):
    """
    Convert full PRImA dataset into HuggingFace detection format.
    """
    train_records = []
    missing_regions = 0

    images_path = os.path.join(path, "Images")

    image_paths = []
    for root_dir, dirs, files in os.walk(images_path):
        for file in files:
            if file.lower().endswith(".tif"):
                image_paths.append(os.path.join(root_dir, file))

    for image_path in image_paths:
        try:
            xml_path = image_path.replace("Images", "XML").replace(".tif", ".xml")

            if not os.path.exists(xml_path):
                base_dir = os.path.dirname(xml_path)
                filename = os.path.basename(xml_path)
                xml_path = os.path.join(base_dir, f"pc-{filename}")

            if not os.path.exists(xml_path):
                print(f"XML file not found for image {image_path}")
                missing_regions += 1
                continue

            annotations = parse_page_xml(xml_path)

            train_records.append({
                "image": image_path,
                "annotations": annotations
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            missing_regions += 1
    
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
                                            debug_path = f"/code/debug_output/{DATASET_REGISTRY_NAME}",
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