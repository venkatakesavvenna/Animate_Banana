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


def get_layout_from_string(decoded_string: str) -> List[str]:
    """
    Extracts all layout class names inside <l>...</l> tags.
    To get the class_names from the string decoded by the VLM/LLM.
    """
    return re.findall(r"<l>(.*?)</l>", decoded_string)


def format_source(source: Dict, get_sam_source_fn: Callable, sam_image_size: int, debug_path:str = "", train:bool = True, val: bool = False, localization_mode=None):
    """
    Format source for inference/custom prompts without ground truth annotations.
    
    Input Format:
    {
        "image": <path_to_image or PIL.Image>,
        "prompt": "Your custom prompt text"
    }

    Required Output Format:
    {
      "image": <PIL.Image>,
      "conversations": [
        {
          "from": "human",
          "value": "<image>\n<prompt>"
        },
        {
          "from": "gpt",
          "value": ""
        }
      ]
    }
    """
    prompt = source.get("prompt", "Give me the layout of the document")
    
    # Handle both PIL Image and path
    original_image_path = source['image']
    if isinstance(original_image_path, str):
        original_image = Image.open(original_image_path).convert("RGB")
    else:
        # Already a PIL image
        original_image = original_image_path.convert("RGB")

    def check_and_raise_error(image):
        """
        Check image size constraints
        """
        from PIL import Image
        
        # If image is a path, open it
        if isinstance(image, str):
            image = Image.open(image)
        
        # Check image dimensions
        width, height = image.size
        if width > 6000 or height > 6000:
            raise ValueError(f"Image size ({width}x{height}) exceeds maximum allowed dimension of 6000 pixels")

    check_and_raise_error(original_image)

    # Format as conversation
    new_source = {
        "image": original_image,
        "conversations":[
            {
                "from": "human", 
                "value": f"<image>\n{prompt}"
            },
            {
                "from": "gpt",
                "value": ""  # Empty for inference
            }
        ]
    }

    sam_specific_source =get_sam_source_fn(original_image, [], is_mask=False, image_size=sam_image_size)
    
    return new_source, sam_specific_source


def convert_custom_prompt_to_huggingface(data_list: List[Dict]):
    """
    Convert a list of custom prompt dictionaries to HuggingFace dataset format.
    
    Args:
        data_list: List of dicts with format:
            {
                "image": <path_to_image or PIL.Image>,
                "prompt": "Your custom prompt"
            }
    
    Returns:
        DatasetDict with train and test splits
    """
    train_records = []
    
    for idx, item in enumerate(tqdm(data_list, desc="Processing custom prompts")):
        try:
            # Validate required fields
            if "image" not in item:
                print(f"Skipping item {idx}: missing 'image' field")
                continue
            
            # Use default prompt if not provided
            prompt = item.get("prompt", "Give me the layout of the document")
            
            train_records.append({
                "image": item["image"],
                "prompt": prompt
            })
        except Exception as e:
            print(f"Exception processing item {idx}: {e}")
    
    if not train_records:
        raise ValueError("No valid records created from data_list")
    
    print(f"Successfully processed {len(train_records)} records")

    # Convert list of dicts to dict of lists for older versions of `datasets` library
    dict_records = {k: [dic[k] for dic in train_records] for k in train_records[0]}
    
    return DatasetDict({
        "train": Dataset.from_dict(dict_records),
        "test": Dataset.from_dict(dict_records),
    })


@DatasetRegistry.register_dataset("custom_prompt")
def get_custom_prompt_dataargs(get_sam_source_fn=format_source_sam, debug_path="", seed=42, sam_image_size=1024, data_list=None, train=False):
    """
    Get DataArguments for custom prompt dataset.
    
    Args:
        get_sam_source_fn: SAM source formatting function
        debug_path: Path for debug visualizations
        seed: Random seed
        sam_image_size: SAM image size
        data_list: List of dicts with 'image' and 'prompt' keys
        train: Whether this is for training (should be False for inference)
    
    Returns:
        DataArguments object
    """
    if data_list is None:
        raise ValueError("data_list must be provided for custom_prompt dataset")
    
    ds = convert_custom_prompt_to_huggingface(data_list)
    
    # For custom prompts, we typically use the test split (inference mode)
    split = "train" if train else "test"
    
    data_args = DataArguments(
        dataset_path="custom_prompt",
        ds = ds[split],
        seed = seed,
        get_source = format_source,
        get_source_kwargs={
            "get_sam_source_fn": get_sam_source_fn,
            "sam_image_size": sam_image_size,
            "debug_path": debug_path,
            "train": train
        },
        extract_from_labels_fn = get_layout_from_string,
        layout_classes = ["UNDEFINED"]
    )
    
    return data_args


# Below part of code is just to run this file independently for debugging purposes
if __name__ == "__main__":
    # Example usage
    example_data_list = [
        {
            "image": "/path/to/image1.png",
            "prompt": "Identify all headers in this document"
        },
        {
            "image": "/path/to/image2.png",
            "prompt": "Extract all tables from this page"
        }
    ]
    
    data_args = DatasetRegistry.get_dataset(
        "custom_prompt",
        get_sam_source_fn = format_source_sam,
        debug_path = "/code/debug_output/custom_prompt",
        seed=42,
        sam_image_size=1024,
        data_list=example_data_list,
        train=False
    )
    
    # importing to autoregister the model
    from ..data_modules.qwen import qwen_data
    
    def add_new_tokens(tokenizer):
        tokenizer.add_tokens("[SEG]")
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        return tokenizer, seg_token_idx
    
    qwen_data_module: DataModule = DataModuleRegistry.get_module(
        "qwenvl",
        data_args=data_args,
        change_tokenizer_fn=add_new_tokens,
        sam_collator=sam_collator,
        model_name="qwen2.5vl",
        model_path="Qwen/Qwen2.5-VL-7B-Instruct"
    )

    from .utils import split_train_eval
    train_ds, eval_ds = split_train_eval(qwen_data_module.Dataloader)

    import random
    seed = 42
    rng = random.Random(seed)

    num_samples = min(10, len(eval_ds))
    indices = rng.sample(range(len(eval_ds)), num_samples)

    for idx in indices:
        x = eval_ds[idx]
        print(f"Sample {idx}: {x}")
