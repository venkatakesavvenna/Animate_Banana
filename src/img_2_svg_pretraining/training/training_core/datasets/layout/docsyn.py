from typing import Callable, Dict, List
from datasets import load_dataset
import copy
import re
import io
from PIL import Image

from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry, DatasetRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, DataArguments
from img_2_svg_pretraining.training.training_core.datasets.layout.utils import get_paragraph_order, visualise_input_dataset

from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import format_source_sam, sam_collator

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

DATASET_PATH="juliozhao/DocSynth300K"
mapping = {
    "0": "QR code",
    "1": "advertisement",
    "2": "algorithm",
    "3": "answer",
    "4": "author",
    "5": "barcode",
    "6": "bill",
    "7": "blank",
    "8": "bracket",
    "9": "breakout",
    "10": "byline",
    "11": "caption",
    "12": "catalogue",
    "13": "chapter title",
    "14": "code",
    "15": "correction",
    "16": "credit",
    "17": "dateline",
    "18": "drop cap",
    "19": "editor's note",
    "20": "endnote",
    "21": "examinee information",
    "22": "fifth-level title",
    "23": "figure",
    "24": "first-level question number",
    "25": "first-level title",
    "26": "flag",
    "27": "folio",
    "28": "footer",
    "29": "footnote",
    "30": "formula",
    "31": "fourth-level section title",
    "32": "fourth-level title",
    "33": "header",
    "34": "headline",
    "35": "index",
    "36": "inside",
    "37": "institute",
    "38": "jump line",
    "39": "kicker",
    "40": "lead",
    "41": "marginal note",
    "42": "matching",
    "43": "mugshot",
    "44": "option",
    "45": "ordered list",
    "46": "other question number",
    "47": "page number",
    "48": "paragraph",
    "49": "part",
    "50": "play",
    "51": "poem",
    "52": "reference",
    "53": "sealing line",
    "54": "second-level question number",
    "55": "second-level title",
    "56": "section",
    "57": "section title",
    "58": "sidebar",
    "59": "sub section title",
    "60": "subhead",
    "61": "subsub section title",
    "62": "supplementary note",
    "63": "table",
    "64": "table caption",
    "65": "table note",
    "66": "teasers",
    "67": "third-level question number",
    "68": "third-level title",
    "69": "title",
    "70": "translator",
    "71": "underscore",
    "72": "unordered list",
    "73": "weather forecast",
}
LAYOUT_CLASSES = list(mapping.values())

def parse_annoline_rect_fast(line: str, W: int, H: int):
    parts = list(map(float, line.strip().split()))

    category_id = int(parts[0])
    coords = parts[1:]

    assert len(coords) == 8, "Expected 4-point polygon"

    # extract points
    x1n, y1n = coords[0], coords[1]
    x3n, y3n = coords[4], coords[5]

    # scale to pixels
    x1, y1 = x1n * W, y1n * H
    w, h = x3n * W - x1, y3n * H - y1

    # # safety check (once during debug)
    # if x2 < x1 or y2 < y1:
    #     raise ValueError("Polygon order assumption violated")

    return {
        "category_id": str(category_id),
        "bbox": [x1, y1, w, h],
    }

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

def format_source(source: Dict, get_sam_source_fn: Callable, sam_image_size: int, debug_path:str = "", train=True, localization_mode=None):
    """
    from the dataset item.. we have to get it into this format

    docsyn Input Format:
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

    def check_and_raise_error(image, bboxes=None):
        """
        If image is path then open the pil image. If image is directly pil image.
        Then check the image size.. if either side is > 3000 then raise an error
        """
        
        # If image is a path, open it
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        
        # Check image dimensions
        width, height = image.size
        if width > 4000 or height > 4000:
            raise ValueError(f"Image size ({width}x{height}) exceeds maximum allowed dimension of 3000 pixels")

        if len(bboxes) > 50:
            raise ValueError("Too Many Layout Elements [SEG]")

    check_and_raise_error(source['image_data'])
    original_image = Image.open(io.BytesIO(source['image_data'])).convert("RGB")
    W, H = original_image.size

    annotations = [
        parse_annoline_rect_fast(line, W, H)
        for line in source["anno_string"]
    ]
    ret_str, list_boxes, category_names = obtain_layout_str(annotations)

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

@DatasetRegistry.register_dataset("docsyn")
def get_docsyn_dataargs(get_sam_source_fn, debug_path="", seed=42, sam_image_size=1024, data_path = DATASET_PATH, train=True):
    # https://github.com/huggingface/datasets/pull/7592
    ds = load_dataset(data_path)
    print("loaded data")
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
                                        "debug_path": debug_path,
                                        "train": False
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
    data_args = DatasetRegistry.get_dataset("docsyn",
                                            get_sam_source_fn = format_source_sam,
                                            debug_path = "/code/debug_output/dataset_input_debug_docsyn",
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
        import pdb; pdb.set_trace()