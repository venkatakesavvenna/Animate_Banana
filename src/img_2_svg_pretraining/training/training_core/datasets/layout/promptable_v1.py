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
from img_2_svg_pretraining.training.training_core.datasets.layout.utils import get_paragraph_order, visualise_input_dataset, visualise_promptable_dataset, save_image_with_bboxes

from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import format_source_sam, sam_collator

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

# DATASET_PATH=os.path.expanduser("/home/raghuveer.r/Layout-Promptable/stage1_flat.jsonl")
# DATASET_PATH=os.path.expanduser("/home/raghuveer.r/Layout-Promptable-Bench/stage1_flat.jsonl")
DATASET_PATH=os.path.expanduser("/fsxvision_new/srihari.bandarupalli/DocGrounding/outputs/90k_checkpoint_promptable_finetune/without_teacher_forcing/v1/promptable_v1/stage1_flat-1.jsonl")


def get_layout_from_string(decoded_string: str) -> List[str]:
    """
    Extracts all layout class names inside <l>...</l> tags.
    To get the class_names from the string decoded by the VLM/LLM.
    """
    pass
    return re.findall(r"<l>(.*?)</l>", decoded_string)

def obtain_layout_str(annotations: Dict):
    """
    annotations = {
                "question": "Identify all section headers in the resume."
                "detailed_answer": "string with <box>[x1, y1, x2, y2]</box> and <box>[x1, y1, x2, y2]</box> and so on for all the boxes in the image. The string can also contain other text apart from the box tags.",
                "bbox_list": [ [x1, y1, x2 - x1, y2 - y1], [], ... ] 
                edge cage; ting accountant verification <box>[235,538,1461,724]</box>, the section for sponsoring institution review <box>[234,748,1265,791]</box>, and the legal verification request <box>[236,816,1461,930]</box>. The response section begins
                  with '回复：' <box>[235,955,1458,1139]</box> followed by the detailed explanation from the law firm <box>[233,1164,1459,1211][239,1174,1453,1274][239,1313,1453,1409]</box>."
            }
    """
    detailed_answer = annotations["detailed_answer"]
    question = annotations["question"]

    # Pattern to match individual [x1,y1,x2,y2] coordinates
    coord_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    
    boxes = []
    
    # Pattern to match <box>...</box> tags
    box_pattern = r"<box>(.*?)</box>"
    
    def process_box_content(match):
        box_content = match.group(1)
        # Find all coordinates within this box tag
        coord_matches = list(re.finditer(coord_pattern, box_content))
        
        for coord_match in coord_matches:
            x1, y1, x2, y2 = map(int, coord_match.groups())
            # Convert to [x1, y1, width, height] format
            boxes.append([x1, y1, x2 - x1, y2 - y1])
        
        # Replace all coordinate patterns with ___ [SEG] ___
        replaced_content = re.sub(coord_pattern, " ___ [SEG] ___ ", box_content)
        return f"<box>{replaced_content}</box>"
    
    ret_str = re.sub(box_pattern, process_box_content, detailed_answer)

    # return ret_str, boxes, question
    return ret_str, annotations["bbox_list"], question

def format_source(source: Dict, get_sam_source_fn: Callable, sam_image_size: int, debug_path:str = "", train:bool = True, val: bool = False, localization_mode=None):
    """
    from the dataset item.. we have to get it into this format
    {
    "image": x[image_key],
    "annotations":{
                    "detailed_answer": "string with <box>[x1, y1, x2, y2]</box> and <box>[x1, y1, x2, y2]</box> and so on for all the boxes in the image. The string can also contain other text apart from the box tags.",
                    "bbox_list": [ [x1, y1, x2 - x1, y2 - y1], [], ... ] 
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
    ret_str, list_boxes, question = obtain_layout_str(annotations)

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
            "value": f"""<image>\nGive me the answer along with layout for this question: {question}"""},
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
            question,
            ret_str,
            entire_sample=source["annotations"]["entire_sample"]
        )
    return new_source, sam_specific_source

def convert_promptable_to_huggingface(path: str = DATASET_PATH):
    """
    promptable_v1 Input Format:
    {
    "image_path": "/fsxvision/raghuveer.r/Datasets/Resumes Datasets/formatted/000000098/pages/000000098_page_1.png", 
    "question": "Identify all section headers in the resume.", 
    "detailed_answer": "The section headers include 'Professional Summary' <box>[50,136,235,157]</box>, 'Skills' <box>[51,242,103,260]</box>, 'Work History' <box>[50,390,171,409]</box>, and 'Education' <box>[50,1881,138,1899]</box>.", 
    "answer": [[50, 136, 235, 157], [51, 242, 103, 260], [50, 390, 171, 409], [50, 1881, 138, 1899]], 
    "axis": "Axis3/flat_layout"
    }
    """
    train_records = []

    jsonl_path=path

    with open(jsonl_path, "r") as fp:
        lines = fp.readlines()

    missing_regions = 0

    for line in tqdm(lines, desc=f"Loading {jsonl_path}"):
        try:
            x = json.loads(line)

            annotations = {
                "question": x["question"],
                "detailed_answer": x["detailed_answer"],
                "bbox_list": [],
                "entire_sample": x
            }
            for bbox in x["answer"]:
                x1, y1, x2, y2 = bbox
                annotations["bbox_list"].append([x1, y1, x2 - x1, y2 - y1])
            
            ret_str, list_boxes, question = obtain_layout_str(annotations)
            # if number of "[SEG]" in the ret_str is not equal to number of boxes in sam_specific_source["masks"], then raise an error
            num_seg_tokens = ret_str.count("[SEG]")
            num_masks = len(list_boxes)


            if num_seg_tokens != num_masks:    
                debug = False
                if not debug:
                    continue
                print("Number of [SEG] tokens in the answer string is not equal to number of masks obtained from SAM source function.")
                print("Number of [SEG] tokens: {}, Number of masks: {}".format(num_seg_tokens, num_masks))
                
                # Visualize bboxes on image when there's a mismatch
                try:
                    img = Image.open(x["image_path"]).convert("RGB")
                    visualise_promptable_dataset(
                        copied_original_image=img,
                        debug_path="/code/debug_output/prompt_viz/v1",
                        question=question,
                        ret_str=ret_str,
                        use_bboxes=True,
                        bboxes=x["answer"],
                        bbox_format="xywh",
                        entire_sample=x
                    )
                except Exception as e:
                    print(f"Visualization error: {e}")
                
                import pdb;pdb.set_trace()
                print(annotations["entire_sample"])
            
            else:
                train_records.append({
                    "image": x["image_path"],
                    "annotations": annotations
                })
        except Exception as e:
            print("Exception", e)
            missing_regions += 1
    
    print("Missing Regions are", missing_regions)

    return DatasetDict({
        "train": Dataset.from_list(train_records),
        "test": Dataset.from_list(train_records),
    })

@DatasetRegistry.register_dataset("promptable_v1")
def get_promptable_v1_dataargs(get_sam_source_fn, debug_path="", seed=42, sam_image_size=1024, data_path = DATASET_PATH, train=True):
    ds = convert_promptable_to_huggingface(data_path)
    print(len(ds))
    # import pdb;pdb.set_trace()
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
                                layout_classes = ["UNDEFINED"]
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
                                    layout_classes = ["UNDEFINED"]
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
                                    layout_classes = ["UNDEFINED"]
                                    )
            print("Error is", e)
    return data_args


# Below part of code is just to run this file independently for debugging purposes
if __name__ == "__main__":
    data_args = DatasetRegistry.get_dataset("promptable_v1",
                                            get_sam_source_fn = format_source_sam,
                                            debug_path = "/code/debug_output/promptable_v3",
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

    # Write LAYOUT_CLASSES_GLOBAL into a dict
    # layout_dict = {str(idx): cls for idx, cls in enumerate(LAYOUT_CLASSES_GLOBAL)}

    # # Write the layout_dict to disk
    # with open("/code/training_core/mappings/hierlay_v2.json", "w", encoding="utf-8") as fp:
    #     json.dump(layout_dict, fp, indent=2, ensure_ascii=False)

    # for idx in tqdm(range(len(train_ds))):
    #     x = train_ds[idx]
    # for idx in tqdm(range(len(eval_ds))):
    #     x = eval_ds[idx]
    
    import random

    seed = 42
    rng = random.Random(seed)

    num_samples = min(30, len(train_ds))
    indices = rng.sample(range(len(train_ds)), num_samples)

    for idx in indices:
        x = train_ds[idx]