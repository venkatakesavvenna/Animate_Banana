import json

import logging
import re
import time
import random

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HF_DATASET

import transformers
from transformers import AutoTokenizer, AutoProcessor, Qwen3VLProcessor
from PIL import Image

# from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import sam_collator

from .qwen_utils import get_rope_index_2, get_rope_index_25, get_rope_index_3


from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule,DataArguments


### All of below code has been imported from https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/data/data_processor.py

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# Model configuration
MODEL_NAME = "qwenvl"
MODEL_FAMILY_NAME = "qwenvlm"

@DataModuleRegistry.register_module(MODEL_FAMILY_NAME)
@DataModuleRegistry.register_module(MODEL_NAME)
def get_qwen_model(
    data_args: DataArguments,
    change_tokenizer_fn: Callable,
    sam_collator: Callable,
    model_name:str,
    model_path: str,
):
    """
    Usage:
    import data_modules.qwen.qwen_data
    dm = DataModuleRegistry.get("qwenvl",
                                data_args, change_tokenizer_fn,
                                model_path="/path/to/my/fine_tuned_qwen"
                            )
    """
    processor: transformers.models.qwen3_vl.Qwen3VLProcessor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer, seg_token_idx = change_tokenizer_fn(processor.tokenizer)

    qwen_dm = DataModule(
        model_name=model_name,
        model_path=model_path,
        processor=processor,
        seg_token_idx = seg_token_idx,
        ignore_idx = IGNORE_INDEX,
        Dataloader=LazySupervisedDataset(processor=processor,data_args=data_args,model_name=model_name),
        Collator=DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer, sam_collator=sam_collator),
        layout_classes=data_args.layout_classes,
        family_name=MODEL_FAMILY_NAME,
        tokenizer_vocab_size=len(processor.tokenizer),
    )
    return qwen_dm

local_rank = 0

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"

# did not modify
def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    return processor

# Modifed to allow PIL images also
def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str) or isinstance(images, Image.Image):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = []
    for img in images:
        if isinstance(img, Image.Image):
            image_pool.append({"type": "image", "image": img})
        else:
            image_pool.append({"type": "image", "image": _make_abs_paths(base_path, img)})

    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        
        if turn["from"] =="human":
            role = "user"
        else:
            role="assistant"

        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages

# did not modify
def preprocess_qwen_visual(
    sources,
    processor: transformers.models.qwen3_vl.Qwen3VLProcessor ,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX) # ignore index for everything except the assistant answer and the end

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels  # (1,seq_len)
    full_result["input_ids"] = input_ids  # (1,seq_len)
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor: transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor, 
                 data_args:DataArguments, model_name:str):
        super(LazySupervisedDataset, self).__init__()
        # dataset_list = data_list(dataset)
        # rank0_print(f"Loading datasets: {data_args}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        # self.model_type = data_args.model_name
        self.model_type = model_name
        if self.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif self.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif self.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {self.model_type} not supported")

        self.list_data_dict: HF_DATASET = data_args.ds

        rank0_print(f"Total training samples: {len(self.list_data_dict)}")

        self.list_data_dict.shuffle(data_args.seed)
        # random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.process_dataset_item: Callable = data_args.get_source
        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    # did not modify below __len__, lengths, modality_lengths, pre_calculated_length, __get__item__
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 1
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                # print("{i}")
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(30):
            try:
                next_index = random.randrange(len(self.list_data_dict))
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        """
        this function gets called by __getitem__
        sources = [self.list_data_dict[i]]
        """
        source, sam_sepcific_source = self.process_dataset_item(source=sources[0], **self.data_args.get_source_kwargs)
        # sam_specific_things added to data_dict later
        data_dict = preprocess_qwen_visual(sources=[source], processor=self.processor)
        # pixel_values and input_ids added to data_dict by Qwen3VLProcessor
        # input_ids and labels are added to data_dict, both of size (1, seq_len)


        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        data_dict={**data_dict, **sam_sepcific_source}
        
        return data_dict    
    
    # did not modify
    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(source, dict):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict

# did not modify
def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    sam_collator: Callable # data_modules/sam/sam_data.py

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        
        """
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        # Keep the batch dimension first so torch.nn.DataParallel scatters
        # position ids by sample instead of splitting the rotary dimension.
        position_ids = pad_and_cat(position_ids).permute(1, 0, 2).contiguous()
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        # input_ids, labels, attention_masks have been padded and added as batched tensors
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        # above things have also been concatenated and added as batched tensors.
        batch = self.sam_collator(instances, batch)
        return batch




def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    # if data_args.data_flatten or data_args.data_packing:
    #     data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
    #     return dict(
    #         train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    #     )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

# What is FlattenedDataCollator??
# DataCollatorForSupervisedDataset:
#   - Standard batching: each example remains independent.
#   - Pads input_ids/labels to (B, S); builds normal attention_mask (1 for non-pad).
#   - Pads/concats position_ids; batches pixel_values/media normally.
#   - Use when not packing sequences or when debugging/training is simpler.

# FlattenedDataCollatorForSupervisedDataset:
#   - Sequence packing: concatenates multiple examples into one long stream.
#   - No per-example padding; input_ids become (B, sum(seq_lengths)).
#   - Builds attention_mask as cumulative sequence lengths (cumsum_seg_lens) so
#     the model can recover segment boundaries.
#   - Concats position_ids along the sequence axis; multimodal tensors also concatenated.
#   - Use when you want higher token throughput and better context utilization.



if __name__ == "__main__":
    pass
