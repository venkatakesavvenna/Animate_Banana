import random
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"


def apply_processor_image_override(processor, preprocessor_config: Dict[str, Any] | None):
    """Mutate a processor's image preprocessor to follow a swapped encoder config.

    This keeps the decoder-owned data module intact while letting encoder-swap runs
    feed images using the replacement encoder's expected resize and normalization.
    """
    if not preprocessor_config:
        return processor

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return processor

    mean = preprocessor_config.get("image_mean")
    std = preprocessor_config.get("image_std")
    image_size = preprocessor_config.get("image_size")

    if mean is not None and hasattr(image_processor, "image_mean"):
        image_processor.image_mean = list(mean)
    if std is not None and hasattr(image_processor, "image_std"):
        image_processor.image_std = list(std)

    if image_size:
        image_size = int(image_size)

        # Qwen image processors are pixel-budget based rather than height/width based.
        if hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = image_size * image_size
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = image_size * image_size

        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            if "height" in size or "width" in size:
                size["height"] = image_size
                size["width"] = image_size
            if "shortest_edge" in size:
                # Qwen stores pixel budgets here; standard processors store edge length.
                size["shortest_edge"] = (
                    image_size * image_size if hasattr(image_processor, "min_pixels") else image_size
                )
            if "longest_edge" in size:
                size["longest_edge"] = (
                    image_size * image_size if hasattr(image_processor, "max_pixels") else image_size
                )
            image_processor.size = size
        elif size is not None:
            image_processor.size = image_size

        crop_size = getattr(image_processor, "crop_size", None)
        if isinstance(crop_size, dict):
            crop_size["height"] = image_size
            crop_size["width"] = image_size
            image_processor.crop_size = crop_size
        elif crop_size is not None:
            image_processor.crop_size = image_size

        # Molmo: base_image_input_size controls the per-crop resolution for multi-crop
        # tiling. Updating it makes the processor generate crops at the swapped encoder's
        # native size rather than the original CLIP 336×336.
        if hasattr(image_processor, "base_image_input_size"):
            image_processor.base_image_input_size = [image_size, image_size]

    return processor


def build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    images = item.get("image") or []
    if isinstance(images, (str, Image.Image)):
        images = [images]

    image_pool = []
    for image in images:
        if isinstance(image, Image.Image):
            image_pool.append({"type": "image", "image": image})
        else:
            image_pool.append({"type": "image", "image": str((base_path / image).resolve())})

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]

        if role == "user":
            content = []
            for segment in re.split(r"(<image>)", text):
                if segment == DEFAULT_IMAGE_TOKEN:
                    if not image_pool:
                        raise ValueError("Encountered <image> placeholder without a backing image")
                    content.append(image_pool.pop(0))
                elif segment.strip():
                    content.append({"type": "text", "text": segment.strip()})
        else:
            content = [{"type": "text", "text": text}]

        messages.append({"role": role, "content": content})

    if image_pool:
        raise ValueError(f"{len(image_pool)} image(s) remain unused in the source item")

    return messages


class GenericVisionLanguageDataset(Dataset):
    def __init__(self, data_args, process_dataset_item: Callable):
        super().__init__()
        self.list_data_dict = data_args.ds.shuffle(data_args.seed)
        self.data_args = data_args
        self.process_dataset_item = process_dataset_item

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for attempt_idx in range(1):
            try:
                return self._fetch(index)
            except Exception as exc:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {index}. Exception: {exc}")
                time.sleep(1)

        for attempt_idx in range(30):
            try:
                next_index = random.randrange(len(self.list_data_dict))
                return self._fetch(next_index)
            except Exception as exc:
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception: {exc}")

        return self._fetch(index)

    def _fetch(self, index: int) -> Dict[str, Any]:
        source = self.list_data_dict[index]
        source, sam_source = self.process_dataset_item(
            source=source,
            **self.data_args.get_source_kwargs,
        )
        base_path = Path(source.get("data_path", ""))
        return {
            "messages": build_messages(source, base_path),
            **sam_source,
        }


class GenericVisionLanguageCollator:
    def __init__(self, processor, tokenizer, sam_collator: Callable, ignore_index: int = IGNORE_INDEX):
        self.processor = processor
        self.tokenizer = tokenizer
        self.sam_collator = sam_collator
        self.ignore_index = ignore_index

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        features = [self._tokenize_instance(instance) for instance in instances]
        batch = merge_multimodal_features(
            features=features,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0,
            ignore_index=self.ignore_index,
            model_max_length=getattr(self.tokenizer, "model_max_length", None),
        )
        return self.sam_collator(instances, batch)

    def _tokenize_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        messages = instance["messages"]
        if not getattr(self.processor, "chat_template", None):
            raise ValueError("This processor does not define a chat template")
        return tokenize_chat_template_instance(
            processor=self.processor,
            messages=messages,
            ignore_index=self.ignore_index,
        )


def tokenize_chat_template_instance(processor, messages, ignore_index: int) -> Dict[str, Any]:
    prompt_messages = list(messages[:-1])
    full_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_inputs = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    feature = dict(full_inputs)
    labels = full_inputs["input_ids"].clone()
    labels[:, : prompt_inputs["input_ids"].shape[-1]] = ignore_index
    feature["labels"] = labels
    return feature


def merge_multimodal_features(
    features: Sequence[Dict[str, Any]],
    pad_token_id: int,
    ignore_index: int,
    model_max_length: int | None = None,
) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    sequence_keys = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    pad_values = {
        "input_ids": pad_token_id,
        "attention_mask": 0,
        "labels": ignore_index,
        "token_type_ids": 0,
    }

    keys = sorted({key for feature in features for key in feature.keys()})
    for key in keys:
        values = [feature[key] for feature in features if key in feature]
        if key in sequence_keys:
            squeezed = [value.squeeze(0) if value.ndim > 1 else value for value in values]
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                squeezed,
                batch_first=True,
                padding_value=pad_values[key],
            )
        elif key == "pixel_values":
            batch[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = values

    if model_max_length and "input_ids" in batch:
        for key in ("input_ids", "attention_mask", "labels", "token_type_ids"):
            if key in batch:
                batch[key] = batch[key][:, :model_max_length]

    return batch
