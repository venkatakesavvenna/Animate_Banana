# Give the flexibility to load the standard checkpoint, choose what all test-datasets to run, choosing the metrics would also be good.
import os, glob
import argparse
from typing import Dict, Any, List
from tqdm import tqdm

import torch; from torch.nn.utils.rnn import pad_sequence; import torch.multiprocessing as mp

from img_2_svg_pretraining.training.training_core.inference.utils import ShardedDataset
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer, AutoProcessor

from img_2_svg_pretraining.training.training_core.models.archive.qwen_sam import QwenSam
from img_2_svg_pretraining.training.training_core.registry.registry import DatasetRegistry, DataModuleRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataArguments, DataModule, ModelConfig, VLMArguments, SamModelArguments

from img_2_svg_pretraining.training.training_core.datasets.layout.utils import split_train_eval
from img_2_svg_pretraining.training.training_core.datasets.layout import doclaynet
from img_2_svg_pretraining.training.training_core.data_modules.qwen import qwen_data

# sam specific dataloader/collator
from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import format_source_sam, sam_collator
from img_2_svg_pretraining.training.training_core.validation.compute_metrics import ComputeMetrics
from img_2_svg_pretraining.training.training_core.data_modules.qwen.qwen_data import IGNORE_INDEX
from img_2_svg_pretraining.training.training_core.validation.layout_map import LayoutMAPAccumulator
from img_2_svg_pretraining.training.training_core.inference.inference_metrics import InferenceMetrics

from datasets import load_dataset
from omegaconf import DictConfig
import os
from datetime import datetime

from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration

def add_new_tokens(tokenizer):
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    return tokenizer, seg_token_idx

# -----------------------------
# Truncation helpers (UNCHANGED)
# -----------------------------
def strip_to_prompt(input_ids: torch.Tensor, assistant_hdr_ids):
    ids = input_ids.tolist()
    try:
        start_idx = next(
            i for i in range(len(ids))
            if ids[i:i+len(assistant_hdr_ids)] == assistant_hdr_ids
        )
        truncated = ids[: start_idx + len(assistant_hdr_ids)]
    except StopIteration:
        truncated = ids
    return torch.tensor(truncated, dtype=input_ids.dtype)

def truncate_batch_to_prompt(input_ids: torch.Tensor, tokenizer):
    assistant_hdr_ids = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False
    )
    truncated_seqs = [strip_to_prompt(seq, assistant_hdr_ids) for seq in input_ids]

    truncated_input_ids = pad_sequence(
        truncated_seqs,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = (truncated_input_ids != tokenizer.pad_token_id).long()
    return truncated_input_ids, attention_mask

def truncate_batch_to_prompt_raw(input_ids: torch.Tensor, tokenizer):
    assistant_hdr_ids = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False
    )
    truncated_seqs = []

    for seq in input_ids:
        # 1. Decode full sequence
        text = tokenizer.decode(seq, skip_special_tokens=False)

        # 2. Replace the prompt text
        if "Give me the layout of the document" in text:
            text = text.replace(
                "Give me the layout of the document,",
                "Give me the layout of the document, mention boxes in the [x1, y1, x2, y2] format."
            )

        # 3. Re-tokenize
        new_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # 4. Truncate to assistant prompt
        truncated = strip_to_prompt(new_ids, assistant_hdr_ids)
        truncated_seqs.append(truncated)

    truncated_input_ids = pad_sequence(
        truncated_seqs,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = (truncated_input_ids != tokenizer.pad_token_id).long()

    return truncated_input_ids, attention_mask

def move_to_device(obj, device):
    """
    Recursively move tensors in a nested structure to the given device.
    Supports: torch.Tensor, dict, list, tuple
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    else:
        return obj  # leave other types unchanged

if __name__ == "__main__":
    if False:
        # -----------------------------
        # Setup
        # -----------------------------
        CFG_PATH = "/code/src/img_2_svg_pretraining/training/configs/test_pramana.yaml"
        DEVICE = "cuda:0"

        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()

        cfg = OmegaConf.load(CFG_PATH)

        # -----------------------------
        # Dataset
        # -----------------------------
        data_args: DataArguments = DatasetRegistry.get_dataset(
            "doclaynet",
            get_sam_source_fn=format_source_sam,
            debug_path="",
            seed=cfg.seed,
            sam_image_size=1024,
            train=False
        )

        data_module = DataModuleRegistry.get_module(
            "qwenvl",
            data_args=data_args,
            change_tokenizer_fn=add_new_tokens,
            sam_collator=sam_collator,
            model_name=cfg.model_family_name,
            model_path=cfg.base_model,
        )

        tokenizer = data_module.processor.tokenizer

        eval_loader = DataLoader(
            data_module.Dataloader,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=data_module.Collator,
        )

        # -----------------------------
        # Model
        # -----------------------------
        model = QwenSam.from_pretrained(
            cfg.model.base_model_checkpoint_path,
            config=ModelConfig(
                sam_args=SamModelArguments(
                    tune_image_encoder=False,
                    tune_prompt_encoder=False,
                    tune_mask_decoder=False,
                    checkpoint="/code/checkpoints/sam_vit_h_4b8939.pth",
                ),
                vlm_args=VLMArguments(
                    model_name_or_path=cfg.base_model,
                    tune_mm_llm=False,
                    tune_mm_vision=False,
                    tune_mm_mlp=False,
                    tune_mm_lm_head=False,
                ),
            ),
            data_module=data_module,
        )

        model.eval().to(DEVICE)
        per_batch_peaks = []

        # -----------------------------
        # Inference (WITH truncation)
        # -----------------------------
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Inference")):

                truncated_ids, truncated_mask = truncate_batch_to_prompt(
                    batch["input_ids"],
                    tokenizer
                )

                batch["input_ids"] = truncated_ids.to(DEVICE)
                batch["attention_mask"] = truncated_mask.to(DEVICE)

                batch = move_to_device(batch, DEVICE)

                _ = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    position_ids=batch.get("position_ids", None),
                    images=batch["images"],
                    resize_list=batch["resize_list"],
                    orig_image_size_list=batch["orig_image_size_list"],
                )

                torch.cuda.synchronize()

                peak = torch.cuda.max_memory_allocated() / 1024**3
                per_batch_peaks.append(peak)

                if batch_idx >= 49:
                    break

        print("\n📊 Per-batch Peak GPU Memory (first 50 batches)")
        print(f"  Min peak  : {min(per_batch_peaks):.2f} GB")
        print(f"  Mean peak : {sum(per_batch_peaks)/len(per_batch_peaks):.2f} GB")
        print(f"  Max peak  : {max(per_batch_peaks):.2f} GB")

    else:
        CFG_PATH = "/code/src/img_2_svg_pretraining/training/configs/test_pramana.yaml"
        DEVICE = "cuda:0"

        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()

        cfg = OmegaConf.load(CFG_PATH)

        # -----------------------------
        # Dataset
        # -----------------------------
        data_args: DataArguments = DatasetRegistry.get_dataset(
            "doclaynet",
            get_sam_source_fn=format_source_sam,
            debug_path="",
            seed=cfg.seed,
            sam_image_size=1024,
            train=False
        )

        data_module = DataModuleRegistry.get_module(
            "qwenvl",
            data_args=data_args,
            change_tokenizer_fn=add_new_tokens,
            sam_collator=sam_collator,
            model_name=cfg.model_family_name,
            model_path=cfg.base_model,
        )

        tokenizer = data_module.processor.tokenizer

        eval_loader = DataLoader(
            data_module.Dataloader,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=data_module.Collator,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",cache_dir=None, 
                                                                   dtype=torch.bfloat16)
        model.eval().to(DEVICE)

        per_batch_peaks = []

        # -----------------------------
        # Inference (WITH truncation)
        # -----------------------------
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Inference")):
                truncated_ids, truncated_mask = truncate_batch_to_prompt_raw(
                    batch["input_ids"],
                    tokenizer
                )

                batch["input_ids"] = truncated_ids.to(DEVICE)
                batch["attention_mask"] = truncated_mask.to(DEVICE)

                batch = move_to_device(batch, DEVICE)

                _ = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                )

                torch.cuda.synchronize()

                peak = torch.cuda.max_memory_allocated() / 1024**3
                per_batch_peaks.append(peak)

                if batch_idx >= 49:
                    break

        print("\n📊 Per-batch Peak GPU Memory (first 50 batches)")
        print(f"  Min peak  : {min(per_batch_peaks):.2f} GB")
        print(f"  Mean peak : {sum(per_batch_peaks)/len(per_batch_peaks):.2f} GB")
        print(f"  Max peak  : {max(per_batch_peaks):.2f} GB")