from typing import Callable, Dict, List
from datasets import Dataset, DatasetDict
import copy
import re
import json
from PIL import Image
from tqdm import tqdm

from collections import defaultdict
from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry, DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, DataArguments
from img_2_svg_pretraining.training.training_core.datasets.layout.utils import get_paragraph_order, visualise_input_dataset

from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import format_source_sam, sam_collator

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

DATASET_PATH="/root/.cache/huggingface/M6_Doc_dataset"
# DATASET_PATH=os.path.expanduser("~/.cache/huggingface/M6_Doc_dataset")
with open("/code/training_core/mappings/hierlay_v2.json") as fp:
    mapping = json.load(fp)
LAYOUT_CLASSES_GLOBAL = list(mapping.values())

def get_layout_from_string(decoded_string: str) -> List[str]:
    """
    Extracts all layout class names inside <l>...</l> tags.
    To get the class_names from the string decoded by the VLM/LLM.
    """
    return re.findall(r"<l>(.*?)</l>", decoded_string)

def obtain_layout_str(annotations: List[Dict]):
    def clean_annotations(annotations: List[Dict]) -> List[Dict]:
        """Remove boxes that are >50% contained within a larger box."""
        def box_area(bbox):
            return bbox[2] * bbox[3]  # w * h

        def intersection_area(a, b):
            # a, b are (x, y, w, h)
            ax1, ay1 = a[0], a[1]
            ax2, ay2 = a[0] + a[2], a[1] + a[3]
            bx1, by1 = b[0], b[1]
            bx2, by2 = b[0] + b[2], b[1] + b[3]

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            return (ix2 - ix1) * (iy2 - iy1)

        to_remove = set()

        for i, ann_i in enumerate(annotations):
            for j, ann_j in enumerate(annotations):
                if i == j or i in to_remove:
                    continue
                area_i = box_area(ann_i["bbox"])
                area_j = box_area(ann_j["bbox"])
                if area_i >= area_j:
                    continue  # only check if i is the smaller box
                inter = intersection_area(ann_i["bbox"], ann_j["bbox"])
                if area_i > 0 and (inter / area_i) > 0.5:
                    to_remove.add(i)

        return [ann for idx, ann in enumerate(annotations) if idx not in to_remove]

    annotations = clean_annotations(annotations)

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
        cat = entry["category_id"]
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

    hierlay_v2 Input Format:
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
            "value": f"""<image>\nGive me the layout of the document, use the following classes {list(set(category_names))}"""},
            {"from": "gpt",
            "value": ret_str
            }
        ]
    }

    sam_specific_source = get_sam_source_fn(original_image, list_boxes, is_mask=False, image_size=sam_image_size)
    
    if debug_path:
        visualise_input_dataset(copy.deepcopy(original_image),debug_path, sam_specific_source["masks"], category_names)

    return new_source, sam_specific_source

def extract_leaves(node):
    leaves = []
    if isinstance(node, dict) and node.get("type") == "leaf": # If this node itself is a leaf
        leaves.append(node)
    if isinstance(node, dict) and "children" in node: # Recurse into children if present
        for child in node["children"]:
            leaves.extend(extract_leaves(child))
    if isinstance(node, list): # If node is a list, recurse into each item
        for item in node:
            leaves.extend(extract_leaves(item))
    return leaves

def convert_hierlay_v2_to_huggingface(path: str):
    train_records = []

    def process_file(jsonl_path, image_key, region_key="regions"):
        nonlocal train_records
        with open(jsonl_path, "r") as fp:
            lines = fp.readlines()

        missing_regions = 0

        for line in tqdm(lines, desc=f"Loading {jsonl_path}"):
            try:
                x = json.loads(line)
                leaves = x[region_key]

                if region_key == "layout":
                    leaves = leaves["children"]

                annotations = []
                for leaf in leaves:
                    x1, y1, x2, y2 = leaf["bbox"]
                    annotations.append({
                        "category_id": leaf["label"],
                        "bbox": [x1, y1, x2 - x1, y2 - y1]
                    })

                train_records.append({
                    "image": x[image_key],
                    "annotations": annotations
                })
            except Exception as e:
                print("Execption", e)
                missing_regions += 1
        
        print("Missing Regions are", missing_regions)

    process_file(
        # "/fsxvision/raghuveer.r/Final-Set/Flat-Set/results_00000_stage1_structured.jsonl",
        "/home/raghuveer.r/test_stage1_bench/jsonl/results_00000_stage1_structured.jsonl",
        image_key="image_path",
    )
    # process_file(
    #     "/fsxvision/raghuveer.r/Final-Set/results/newspaper/results-2.jsonl",
    #     image_key="image_path",
    #     region_key="layout"
    # )

    return DatasetDict({
        "train": Dataset.from_list(train_records),
        "test": Dataset.from_list(train_records),
    })

@DatasetRegistry.register_dataset("hierlay_v2")
def get_hierlay_v2_dataargs(get_sam_source_fn, debug_path="", seed=42, sam_image_size=1024, data_path = DATASET_PATH, train=True):
    ds = convert_hierlay_v2_to_huggingface(data_path)
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
                                layout_classes = LAYOUT_CLASSES_GLOBAL
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
                                    layout_classes = LAYOUT_CLASSES_GLOBAL
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
                                    layout_classes = LAYOUT_CLASSES_GLOBAL
                                    )
            print("Error is", e)
    return data_args


# Below part of code is just to run this file independently for debugging purposes
if __name__ == "__main__":
    data_args = DatasetRegistry.get_dataset("hierlay_v2",
                                            get_sam_source_fn = format_source_sam,
                                            debug_path = "/code/debug_output/dataset_input_debug_hierlay_v2_new",
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

    # Write LAYOUT_CLASSES_GLOBAL into a dict
    # layout_dict = {str(idx): cls for idx, cls in enumerate(LAYOUT_CLASSES_GLOBAL)}

    # # Write the layout_dict to disk
    # with open("/code/training_core/mappings/hierlay_v2.json", "w", encoding="utf-8") as fp:
    #     json.dump(layout_dict, fp, indent=2, ensure_ascii=False)

    import random

    seed = 42
    rng = random.Random(seed)

    num_samples = min(3000, len(train_ds))
    indices = rng.sample(range(len(train_ds)), num_samples)

    for idx in indices:
        x = train_ds[idx]