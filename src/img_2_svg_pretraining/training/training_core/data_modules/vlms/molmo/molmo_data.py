import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image, ImageOps

from img_2_svg_pretraining.training.training_core.data_modules.vlms.common import (
    IGNORE_INDEX,
    GenericVisionLanguageDataset,
)
from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataArguments, DataModule

_logger = logging.getLogger(__name__)

MODEL_FAMILY_NAME = "molmovlm"


def _extract_images_from_messages(messages: List[Dict]) -> List[Image.Image]:
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") != "image":
                continue
            img = item["image"]
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
                
            try:
                img = ImageOps.exif_transpose(img)
            except (SyntaxError, AttributeError, OSError):
                img.info.pop("exif", None)
                
            images.append(img)
    return images


def _build_molmo_text(messages: List[Dict], add_generation_prompt: bool) -> str:
    """Minimal user/assistant format for Molmo variants without a chat template."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multi-modal content; extract text parts only (images handled separately).
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
        if role == "user":
            parts.append(f" User: {content}")
        elif role == "assistant":
            parts.append(f" Assistant: {content}")
    if add_generation_prompt:
        parts.append(" Assistant:")
    return "\n".join(parts)


def _find_subsequence_occurrences(haystack: List[int], needle: List[int]) -> List[int]:
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return []
    return [i for i in range(n - m + 1) if haystack[i:i + m] == needle]


def _find_answer_boundaries(
    processor, full_ids: torch.Tensor, prompt_text: str, process_fn
) -> tuple:
    """Find where the assistant's answer starts (and, if present, a spurious
    trailing duplicate) inside the WITH-IMAGES token sequence.

    Naively re-tokenizing prompt_text alone (no images) to get its length and
    using that as the mask boundary is wrong: Molmo's processor injects
    hundreds of image-patch tokens when images are present, so a no-image
    prompt_len is nowhere near the real boundary in the image-conditioned
    full sequence. It also isn't safe to just re-tokenize prompt_text WITH
    images and use its length as a prefix offset — the processor is not
    prefix-stable here; feeding it text that already ends in " Assistant:"
    causes it to append a second, spurious " Assistant:" of its own, so the
    re-tokenized "prompt" is longer than the true prefix of the full sequence.

    Instead, search for the literal " Assistant:" role-marker tokens directly
    inside the already-correctly-tokenized full sequence. The first
    occurrence marks the true prompt/answer boundary. The same spurious
    duplicate-marker behavior can also append " Assistant:" again right
    after the answer; if the marker's last occurrence ends within a few
    tokens of the sequence end, that's this artifact, not real answer
    content, so it must be masked too — otherwise the model is trained to
    predict "Assistant:" as a valid continuation after a complete answer,
    which is exactly what produces role-marker repetition loops at inference.

    Falls back to the (less precise) no-image prompt tokenization if the
    marker can't be found at all, e.g. a custom chat template that doesn't
    use this literal role-marker text.

    Returns (prompt_len, trailing_mask_start_or_None).
    """
    marker_ids = processor.tokenizer(" Assistant:", add_special_tokens=False)["input_ids"]
    ids_list = full_ids.tolist()
    occurrences = _find_subsequence_occurrences(ids_list, marker_ids)
    if not occurrences:
        _logger.warning(
            "[molmo_data] Could not locate ' Assistant:' marker in tokenized sequence; "
            "falling back to no-image prompt length (mask boundary may be inaccurate)."
        )
        prompt_inputs = process_fn(text=prompt_text, return_tensors="pt")
        return prompt_inputs["input_ids"].shape[-1], None

    prompt_len = occurrences[0] + len(marker_ids)
    trailing_mask_start = None
    if len(occurrences) > 1:
        last_occ = occurrences[-1]
        seq_len = len(ids_list)
        if seq_len - (last_occ + len(marker_ids)) <= 4:
            trailing_mask_start = last_occ
    return prompt_len, trailing_mask_start


def tokenize_molmo_instance(
    processor,
    messages: List[Dict],
    ignore_index: int,
    max_crops: Optional[int] = None,
) -> Dict:
    """Encode one conversation sample using Molmo's processor.

    Molmo requires apply_chat_template(tokenize=False) + processor(text=..., images=...)
    rather than the tokenize=True path used by Gemma and Qwen.

    max_crops caps how many image crops the processor may emit for a single image
    (Molmo defaults to 12, ~144 tokens each). Uncapped, one high-resolution image
    can dominate a micro-batch's memory since all images in a batch get padded to
    the largest crop count present (see _pad_to_max_crops).
    """
    images = _extract_images_from_messages(messages)
    image_kwargs = {"max_crops": max_crops} if max_crops is not None else {}

    try:
        full_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = processor.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
    except ValueError:
        # Some Molmo variants (e.g. 7B-O with OLMo backbone) have no chat template;
        # build a minimal user/assistant format manually.
        full_text = _build_molmo_text(messages, add_generation_prompt=False)
        prompt_text = _build_molmo_text(messages[:-1], add_generation_prompt=True)

    # Molmo's processor has a custom .process() method that injects the required special
    # token IDs for image patches; the standard __call__ path does not.
    _process_fn = getattr(processor, "process", None) or processor
    full_inputs = _process_fn(
        text=full_text,
        images=images if images else None,
        return_tensors="pt",
        **image_kwargs,
    )

    # process() may return 1-D tensors; normalise to [1, seq_len] for uniform indexing.
    if full_inputs["input_ids"].ndim == 1:
        for k, v in full_inputs.items():
            if isinstance(v, torch.Tensor) and v.ndim == 1:
                full_inputs[k] = v.unsqueeze(0)

    prompt_len, trailing_mask_start = _find_answer_boundaries(
        processor, full_inputs["input_ids"][0], prompt_text, _process_fn
    )
    if trailing_mask_start is not None:
        # Drop the processor's spurious trailing " Assistant:" duplicate from
        # the sequence entirely (not just from labels) — it isn't real answer
        # content, so there's no reason to keep it in input_ids either.
        full_inputs["input_ids"] = full_inputs["input_ids"][:, :trailing_mask_start]
        if "attention_mask" in full_inputs:
            full_inputs["attention_mask"] = full_inputs["attention_mask"][:, :trailing_mask_start]

    # Append an explicit, supervised EOS after the answer. Nothing upstream
    # (chat template, _build_molmo_text, or the processor) adds one, so
    # without this the model never gets a training signal for when to stop —
    # it has no reason to prefer emitting EOS over continuing to generate
    # (e.g. hallucinating another "Assistant: ..." turn) after a complete answer.
    eos_id = processor.tokenizer.eos_token_id
    if eos_id is not None:
        eos_col = full_inputs["input_ids"].new_full((full_inputs["input_ids"].shape[0], 1), eos_id)
        full_inputs["input_ids"] = torch.cat([full_inputs["input_ids"], eos_col], dim=1)
        if "attention_mask" in full_inputs:
            ones_col = full_inputs["attention_mask"].new_ones((full_inputs["attention_mask"].shape[0], 1))
            full_inputs["attention_mask"] = torch.cat([full_inputs["attention_mask"], ones_col], dim=1)

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = ignore_index

    result = {k: v for k, v in full_inputs.items()}
    result["labels"] = labels
    return result


def _pad_to_max_crops(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Stack tensors whose first dimension (n_crops) may differ, padding with pad_value."""
    # Each tensor is [n_crops, ...]; squeeze batch dim if the processor left it in.
    squeezed = [t.squeeze(0) if t.ndim > 1 and t.shape[0] == 1 and t.ndim >= 3 else t for t in tensors]
    # Re-check: if processor returns [1, n_crops, ...] squeeze the leading 1.
    squeezed = [t.squeeze(0) if t.ndim >= 3 and t.shape[0] == 1 else t for t in squeezed]
    max_crops = max(t.shape[0] for t in squeezed)
    padded = []
    for t in squeezed:
        deficit = max_crops - t.shape[0]
        if deficit:
            pad_t = torch.full((deficit,) + t.shape[1:], pad_value, dtype=t.dtype)
            t = torch.cat([t, pad_t], dim=0)
        padded.append(t)
    return torch.stack(padded, dim=0)


def merge_molmo_features(
    features: Sequence[Dict[str, Any]],
    pad_token_id: int,
    ignore_index: int,
    model_max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Batch a list of per-sample Molmo feature dicts into one padded batch."""
    batch: Dict[str, Any] = {}

    _seq_pad = {
        "input_ids": pad_token_id,
        "attention_mask": 0,
        "labels": ignore_index,
        "image_input_idx": -1,
    }
    for key, pad_val in _seq_pad.items():
        tensors = [f[key].squeeze(0) for f in features if key in f]
        if not tensors:
            continue
        batch[key] = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=pad_val
        )

    if "images" in features[0]:
        image_list = [f["images"] for f in features if "images" in f]
        try:
            batch["molmo_images"] = _pad_to_max_crops(image_list, pad_value=0.0)
        except Exception:
            # Fallback: assume all same shape
            batch["molmo_images"] = torch.cat(image_list, dim=0)

    if "image_masks" in features[0]:
        mask_list = [f["image_masks"] for f in features if "image_masks" in f]
        try:
            batch["image_masks"] = _pad_to_max_crops(mask_list, pad_value=0.0)
        except Exception:
            batch["image_masks"] = torch.cat(mask_list, dim=0)

    if model_max_length and "input_ids" in batch:
        for key in ("input_ids", "attention_mask", "labels", "image_input_idx"):
            if key in batch:
                batch[key] = batch[key][:, :model_max_length]

    return batch


class MolmoCollator:
    """Batching collator for the molmovlm family."""

    def __init__(
        self,
        processor,
        tokenizer,
        sam_collator: Callable,
        ignore_index: int = IGNORE_INDEX,
        max_crops: Optional[int] = None,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.sam_collator = sam_collator
        self.ignore_index = ignore_index
        self.max_crops = max_crops

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        features = [
            tokenize_molmo_instance(
                self.processor, inst["messages"], self.ignore_index, max_crops=self.max_crops
            )
            for inst in instances
        ]
        batch = merge_molmo_features(
            features=features,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0,
            ignore_index=self.ignore_index,
            model_max_length=getattr(self.tokenizer, "model_max_length", None),
        )
        return self.sam_collator(instances, batch)


@DataModuleRegistry.register_module(MODEL_FAMILY_NAME)
def get_molmo_data_module(
    data_args: DataArguments,
    change_tokenizer_fn: Callable,
    sam_collator: Callable,
    model_name: str,
    model_path: str,
    molmo_max_crops: Optional[int] = None,
):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if not getattr(processor, "tokenizer", None):
        raise ValueError(
            "molmovlm: AutoProcessor did not expose a .tokenizer attribute. "
            "Ensure trust_remote_code=True and the checkpoint is a Molmo checkpoint."
        )
    processor.tokenizer, seg_token_idx = change_tokenizer_fn(processor.tokenizer)

    return DataModule(
        model_name=model_name,
        model_path=model_path,
        processor=processor,
        seg_token_idx=seg_token_idx,
        ignore_idx=IGNORE_INDEX,
        Dataloader=GenericVisionLanguageDataset(
            data_args=data_args,
            process_dataset_item=data_args.get_source,
        ),
        Collator=MolmoCollator(
            processor=processor,
            tokenizer=processor.tokenizer,
            sam_collator=sam_collator,
            ignore_index=IGNORE_INDEX,
            max_crops=molmo_max_crops,
        ),
        layout_classes=data_args.layout_classes,
        family_name=MODEL_FAMILY_NAME,
        tokenizer_vocab_size=len(processor.tokenizer),
    )
