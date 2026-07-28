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
from img_2_svg_pretraining.training.training_core.datasets.layout import doclaynet, indicdlp, m6doc, d4la
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

# the yaml file test_pramana.yaml is as follows:
"""
project_root: /code
test_dataset_root: /code/src/img_2_svg_pretraining/training/datasets/mmdocbench_converted_sampled_augmented_500.jsonl
model_family_name: qwen_vl
base_model: Qwen/Qwen2.5-VL-7B-Instruct
seed: 42

output_dir: /data/outputs
checkpoint_step: 12000
devices: ['cuda:0']

train_images: ${test_dataset_root}/train_images

model:
  base_model_checkpoint_path: ${output_dir}/pramana_final_run_multinode_full_dataset/checkpoint-${checkpoint_step}
  sam_model_checkpoint_path: ${output_dir}/pramana_final_run_multinode_full_dataset/checkpoint-step-${checkpoint_step}/sam_state_dict.pt

data:
  _name_: ${model_family_name}
  dataset_use_dogr: ${test_dataset_root}
  image_root: ${train_images}
  max_pixels: 50176
  min_pixels: 784
  merge_size: 2
  split_ratio: 1
  seed: ${seed}
"""

"""
Also, each evaluate_model() should basically return objects of the form:
{
    "predictions": {
        "text": [
            {
                "mask": np.ndarray (H, W),
                "bbox": [x1, y1, x2, y2],
                "score": 1.0
            }
            ....
        ]
        "table": [...],
        "list": [...],
        "title": [...],
        "figure": [...]
    },
    "ground_truth": {
        "text": [
        {
            "mask": np.ndarray (H,W),
            "bbox": [x1, y1, x2, y2]
        },
        ...
        ],
        "table": [...],
        ...
    }
}
"""

def load_existing_shards(cfg: DictConfig):
    for dataset_name in cfg.eval.datasets:
        dataset_path = f'/code/metrics/{cfg.run_name}/{dataset_name}{cfg.model.base_model_checkpoint_path.replace(cfg.output_dir, "")}'

        # For the given dataset, yield the metrics_calculator
        dm, data_args = build_data_module(cfg, dataset_name)
        metrics_calculator = build_metrics(data_args)

        if os.path.isdir(dataset_path): # Check if directory exists
            shard_files = glob.glob(os.path.join(dataset_path, "*.pt"))
            
            if len(shard_files) == 0:
                return False

            print(f"[INFO] Found {len(shard_files)} existing metric shards. Skipping inference.")

            all_preds, all_gts = [], []
            for f in shard_files:
                data = torch.load(f, map_location="cpu")
                all_preds.extend(data["preds"])
                all_gts.extend(data["gts"])

            results = metrics_calculator(all_preds, all_gts)
            print(results)
        else:
            print(f"Dataset {dataset_name} is not processed yest,,")
    return True

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

def add_new_tokens(tokenizer):
    # tokenizer.add_tokens("[SEG]")
    # seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    tokenizer.add_tokens("[SEG]")       # add the segmentation token
    tokenizer.padding_side = "left"     # Ensure a valid pad token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    return tokenizer, seg_token_idx

def strip_to_prompt(input_ids: torch.Tensor, assistant_hdr_ids: List[int]) -> torch.Tensor:
    """
    Keep tokens up to and including the assistant header, drop everything after.
    """
    ids = input_ids.tolist()
    try:
        start_idx = next(
            i for i in range(len(ids))
            if ids[i:i+len(assistant_hdr_ids)] == assistant_hdr_ids
        )
        truncated = ids[: start_idx + len(assistant_hdr_ids)]
    except StopIteration:
        truncated = ids  # if assistant header not found, keep as is
    return torch.tensor(truncated, dtype=input_ids.dtype) # Stripped.

def truncate_batch_to_prompt(
    input_ids: torch.Tensor,
    tokenizer
):
    """
    input_ids: (B, L)
    returns:
        truncated_input_ids: (B, L')
        attention_mask:      (B, L')
    """
    assistant_hdr_ids = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False
    )
    truncated_seqs = []
    for seq in input_ids:
        truncated = strip_to_prompt(seq, assistant_hdr_ids)
        truncated_seqs.append(truncated.tolist()) # <- list[int] for tokenizer.pad
    
    # pad back to batch
    padded = tokenizer.pad(
        {"input_ids": truncated_seqs},
        padding=True,
        return_tensors="pt"
    )
    # truncated_input_ids = pad_sequence(
    #     truncated_seqs,
    #     batch_first=True,
    #     padding_value=tokenizer.pad_token_id
    # )

    truncated_input_ids = padded["input_ids"]
    attention_mask = padded["attention_mask"]

    # attention_mask = (truncated_input_ids != tokenizer.pad_token_id).long()
    return truncated_input_ids, attention_mask

def build_data_module(cfg, dataset_name):
    data_args = DatasetRegistry.get_dataset(
        dataset_name,
        get_sam_source_fn=format_source_sam,
        debug_path="",
        seed=cfg.seed,
        sam_image_size=1024,
        train=False
    )

    dm = DataModuleRegistry.get_module(
        "qwenvl",
        data_args=data_args,
        change_tokenizer_fn=add_new_tokens,
        sam_collator=sam_collator,
        model_name=cfg.model_family_name,
        model_path=cfg.base_model,
    )

    return dm, data_args

def build_metrics(data_args: DataArguments):
    return InferenceMetrics(
        data_args.extract_from_labels_fn,
        data_args.layout_classes
    )

def evaluate_single_dataset(
    cfg,
    dm: DataModule,
    model: QwenSam,
    metrics: InferenceMetrics,
    rank: int,
    dataset_name: str
):
    device = model.device
    eval_dataset = dm.Dataloader
    collator = dm.Collator

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator
    )

    tokenizer = dm.processor.tokenizer
    all_preds, all_gts = [], []

    for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f"[GPU {rank}] Eval", position=rank, leave=True)):            
        labels = [l.split("<|im_start|>assistant\n")[-1].split("<|im_end|>\n")[0] for l in tokenizer.batch_decode(batch["input_ids"])] # TODO: Why do the labels have a < missing? 
        truncated_input_ids, truncated_attention_mask = truncate_batch_to_prompt(
            batch["input_ids"],
            tokenizer
        )
        batch["input_ids"] = truncated_input_ids.to(device)             # replace batch entries
        batch["attention_mask"] = truncated_attention_mask.to(device)   # replace batch entries
        batch = move_to_device(batch, device)
        try:
            with torch.no_grad():
                preds, pred_masks = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    position_ids=batch.get("position_ids", None),
                    images=batch["images"],
                    resize_list=batch["resize_list"],
                    orig_image_size_list=batch["orig_image_size_list"],
                )
            gt_masks = batch["masks"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            cur_batch_preds, cur_batch_gts = metrics.evaluate_batch(pred_masks, gt_masks,
                                                labels, preds, batch["debug_original_image_list"], 
                                                f'/code/debug/doclaynet_test/{cfg.run_name}/{cfg.model.base_model_checkpoint_path.replace(cfg.output_dir, "")}/{str(timestamp)}_{rank}/', device)
            all_preds.extend(cur_batch_preds)
            all_gts.extend(cur_batch_gts)

        except Exception as e:
            print(
                f"[WARN] Batch {batch_idx} | "
                f"Error: {str(e)}"
            )

    save_dir = f'/code/metrics/{cfg.run_name}/{dataset_name}{cfg.model.base_model_checkpoint_path.replace(cfg.output_dir, "")}'
    os.makedirs(save_dir, exist_ok=True)

    # Save all of them to disk
    torch.save(
        {
            "preds": all_preds,
            "gts": all_gts
        },
        f'{save_dir}/metrics_shard_{rank}.pt'
    )

def worker(rank: int, cfg: DictConfig, devices: List[str]):
    device = devices[rank]

    # ---- GPU binding (CRITICAL) ----
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    torch.cuda.set_device(rank)

    print(f"[Worker {rank}] Using device {device}")

    # ---- Load model + data exactly as before ----
    model_name_or_path = cfg.base_model

    for dataset_name in cfg.eval.datasets:
        dm, data_args = build_data_module(cfg, dataset_name)

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
                    model_name_or_path=model_name_or_path,
                    tune_mm_llm=False,
                    tune_mm_vision=False,
                    tune_mm_mlp=False,
                    tune_mm_lm_head=False,
                ),
            ),
            data_module=dm,
        )
        model.eval().cuda()

        dm.Dataloader = ShardedDataset(
            dm.Dataloader,
            shard_id=rank,
            num_shards=len(devices)
        )

        metrics = build_metrics(data_args)
        
        evaluate_single_dataset(cfg, dm, model, metrics, rank, dataset_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="/code/src/img_2_svg_pretraining/training/configs/test_pramana.yaml")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
        help="List of CUDA devices"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    # EARLY EXIT PATH
    if load_existing_shards(cfg):
        return

    data_args: DataArguments = DatasetRegistry.get_dataset(
        "doclaynet",
        get_sam_source_fn=format_source_sam,
        debug_path="",
        seed=cfg.seed,
        sam_image_size=1024,
        train=False
    )

    metrics_calculator = InferenceMetrics(
        data_args.extract_from_labels_fn,
        data_args.layout_classes
    )

    # 🧠 Otherwise run inference
    mp.spawn(
        worker,
        args=(cfg, args.devices),
        nprocs=len(args.devices),
        join=True
    )

    # EARLY EXIT PATH
    if load_existing_shards(cfg):
        return

if __name__ == "__main__":
    main()