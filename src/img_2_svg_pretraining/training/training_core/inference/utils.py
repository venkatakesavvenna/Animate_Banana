from typing import List, Union
import torch
import numpy as np
import cv2


GENERATE_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "position_ids",
    "token_type_ids",
    "images",
    "resize_list",
    "orig_image_size_list",
    # Molmo-specific VLM image keys (renamed from "images" to avoid SAM conflict)
    "molmo_images",
    "image_input_idx",
    "image_masks",
}


class ShardedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, shard_id, num_shards):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))[shard_id::num_shards]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    
def move_to_device(obj, device):
    """Recursively move tensors in a nested structure to the given device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    else:
        return obj


def strip_to_prompt(input_ids: torch.Tensor, assistant_hdr_ids: List[int]) -> torch.Tensor:
    """Keep tokens up to and including the assistant header, drop everything after."""
    ids = input_ids.tolist()
    try:
        start_idx = next(
            i for i in range(len(ids))
            if ids[i:i+len(assistant_hdr_ids)] == assistant_hdr_ids
        )
        truncated = ids[: start_idx + len(assistant_hdr_ids)]
    except StopIteration:
        truncated = ids  # if assistant header not found, keep as is
    return torch.tensor(truncated, dtype=input_ids.dtype)


def truncate_batch_to_prompt(input_ids: torch.Tensor, tokenizer):
    """
    Truncate batch to prompt only (remove ground truth labels for generation).
    
    Args:
        input_ids: (B, L)
        tokenizer: Tokenizer
    
    Returns:
        truncated_input_ids: (B, L')
        attention_mask: (B, L')
    """
    assistant_hdr_ids = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False
    )
    
    truncated_seqs = []
    for seq in input_ids:
        truncated = strip_to_prompt(seq, assistant_hdr_ids)
        truncated_seqs.append(truncated.tolist())
    
    # Pad back to batch
    padded = tokenizer.pad(
        {"input_ids": truncated_seqs},
        padding=True,
        return_tensors="pt"
    )
    
    truncated_input_ids = padded["input_ids"]
    attention_mask = padded["attention_mask"]
    
    return truncated_input_ids, attention_mask


def truncate_batch_to_prompt_from_labels(batch: dict, ignore_index: int = -100) -> dict:
    """
    Truncate a multimodal batch to prompt-only length using masked labels.

    This is family-agnostic as long as the collator follows the standard
    convention of marking prompt tokens with ``ignore_index`` in ``labels``.
    """
    if "labels" not in batch:
        raise KeyError("Batch does not contain labels; cannot infer prompt length from labels")

    prompt_lengths = [int((labels == ignore_index).sum().item()) for labels in batch["labels"]]
    max_prompt_length = max(prompt_lengths) if prompt_lengths else 0

    truncated = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            truncated[key] = value
            continue

        if key in {"input_ids", "attention_mask", "labels", "token_type_ids"} and value.ndim >= 2:
            truncated[key] = value[:, :max_prompt_length]
        elif key == "position_ids" and value.ndim == 3:
            truncated[key] = value[..., :max_prompt_length]
        else:
            truncated[key] = value

    return truncated


def prepare_generate_batch(batch: dict, tokenizer=None, ignore_index: int = -100) -> dict:
    """
    Prepare a family-agnostic batch for ``VLMSam.generate(...)``.

    Preference order:
    1. Use ``labels`` to infer prompt length for any supported VLM family.
    2. Fall back to tokenizer-based prompt truncation when labels are absent.
    """
    if "labels" in batch:
        prepared = truncate_batch_to_prompt_from_labels(batch, ignore_index=ignore_index)
    else:
        if tokenizer is None:
            raise ValueError("tokenizer is required when batch does not contain labels")
        truncated_input_ids, truncated_attention_mask = truncate_batch_to_prompt(
            batch["input_ids"],
            tokenizer,
        )
        prepared = dict(batch)
        prepared["input_ids"] = truncated_input_ids
        prepared["attention_mask"] = truncated_attention_mask

    return {
        key: value
        for key, value in prepared.items()
        if key in GENERATE_INPUT_KEYS
    }


def decode_supervised_targets(labels: torch.Tensor, tokenizer, ignore_index: int = -100) -> List[str]:
    """
    Decode only supervised target tokens from a label tensor.
    """
    decoded = []
    for row in labels:
        target_ids = row[row != ignore_index]
        decoded.append(tokenizer.decode(target_ids, skip_special_tokens=True))
    return decoded


def to_numpy_mask(
    x: Union[torch.Tensor, np.ndarray, List],
    threshold: float = 0.0
) -> np.ndarray:
    """
    Converts tensor / numpy / list of tensors to a binary numpy uint8 mask.
    
    - Tensor → np.ndarray
    - np.ndarray → np.ndarray
    - List[Tensor or np.ndarray] → concatenated np.ndarray along dim=0
    """

    def _convert_single(item) -> np.ndarray:
        if isinstance(item, torch.Tensor):
            arr = item.detach().cpu().float().numpy()
        else:
            arr = np.asarray(item)

        if arr.dtype != np.uint8:
            arr = (arr > threshold).astype(np.uint8)

        return arr

    # -------- list / tuple case --------
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("to_numpy_mask received an empty list")

        arrays = [_convert_single(item) for item in x]

        # sanity check: all shapes except dim-0 must match
        ref_shape = arrays[0].shape[1:]
        for i, arr in enumerate(arrays):
            if arr.shape[1:] != ref_shape:
                raise ValueError(
                    f"Shape mismatch at index {i}: "
                    f"{arr.shape} vs expected (*, {ref_shape})"
                )

        return np.stack(arrays, axis=0)

    # -------- single tensor / array --------
    return _convert_single(x)

def mask_to_bbox(mask: np.ndarray):
    """
    Compute tight axis-aligned bounding box from binary mask
    using contours. Returns [x1, y1, x2, y2] or None.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None
    cnt = max(contours, key=cv2.contourArea) # take largest connected component
    x, y, w, h = cv2.boundingRect(cnt)
    return [int(x), int(y), int(x + w), int(y + h)]
