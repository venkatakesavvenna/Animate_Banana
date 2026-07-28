from __future__ import annotations

from typing import Callable

from datasets import Dataset
from PIL import Image

from img_2_svg_pretraining.training.training_core.data_modules.sam.sam1 import sam1_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.data_modules.vlms.gemma import gemma_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.data_modules.vlms.molmo import molmo_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.data_modules.vlms.qwen import qwen_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.sam.sam1 import sam1_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.vlm_sam import VLMSam
from img_2_svg_pretraining.training.training_core.models.vlms.gemma import gemma_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.vlms.molmo import molmo_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.vlms.qwen import qwen_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry, SAMDataModuleRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataArguments, ModelConfig, SamModelArguments, VLMArguments

from tests.support.matrix import VLMCheckpointSpec


DEFAULT_USER_PROMPT = "<image>\nDescribe the document layout."
DEFAULT_ASSISTANT_RESPONSE = "<layout>title</layout> [SEG]"


def add_seg_token(tokenizer):
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    return tokenizer, seg_token_idx


def make_synthetic_image(size: tuple[int, int] = (32, 32), color: str = "white") -> Image.Image:
    return Image.new("RGB", size, color=color)


def build_test_data_args(
    image,
    sam_source_fn: Callable,
    sam_image_size: int = 64,
    boxes: list[list[int]] | None = None,
    user_prompt: str = DEFAULT_USER_PROMPT,
    assistant_response: str = DEFAULT_ASSISTANT_RESPONSE,
) -> DataArguments:
    images = image if isinstance(image, list) else [image]
    ds = Dataset.from_list([{"image": item, "boxes": boxes or [[0, 0, 16, 16]]} for item in images])
    return DataArguments(
        dataset_path="synthetic",
        ds=ds,
        seed=7,
        get_source=_test_get_source,
        get_source_kwargs={
            "get_sam_source_fn": sam_source_fn,
            "sam_image_size": sam_image_size,
            "user_prompt": user_prompt,
            "assistant_response": assistant_response,
        },
        extract_from_labels_fn=lambda _text: ["title"],
        layout_classes=["title"],
    )


def _test_get_source(
    source,
    get_sam_source_fn,
    sam_image_size: int = 64,
    user_prompt: str = DEFAULT_USER_PROMPT,
    assistant_response: str = DEFAULT_ASSISTANT_RESPONSE,
):
    image = source["image"]
    boxes = source["boxes"]
    sam_source = get_sam_source_fn(image, boxes, image_size=sam_image_size)
    formatted = {
        "image": image,
        "conversations": [
            {"from": "human", "value": user_prompt},
            {"from": "assistant", "value": assistant_response},
        ],
        "data_path": "",
    }
    return formatted, sam_source


def load_vlm_data_module(
    spec: VLMCheckpointSpec,
    model_ref: str,
    image,
    sam_version: str = "sam1",
    sam_image_size: int = 64,
    boxes: list[list[int]] | None = None,
):
    sam_dm = SAMDataModuleRegistry.get_sam_data_module(sam_version)
    data_args = build_test_data_args(
        image=image,
        sam_source_fn=sam_dm.format_source,
        sam_image_size=sam_image_size,
        boxes=boxes,
    )
    return DataModuleRegistry.get_module(
        spec.family,
        data_args=data_args,
        change_tokenizer_fn=add_seg_token,
        sam_collator=sam_dm.get_collator(),
        model_name=spec.model_family_name,
        model_path=model_ref,
    )


def build_vlm_args(
    spec: VLMCheckpointSpec,
    model_ref: str,
    attn_implementation: str | None = None,
    gradient_checkpointing: bool = False,
    tune_mm_llm: bool = False,
    tune_mm_mlp: bool = False,
    tune_mm_vision: bool = False,
    tune_mm_lm_head: bool = False,
) -> VLMArguments:
    return VLMArguments(
        family=spec.family,
        model_name_or_path=model_ref,
        model_family_name=spec.model_family_name,
        attn_implementation=attn_implementation,
        gradient_checkpointing=gradient_checkpointing,
        tune_mm_llm=tune_mm_llm,
        tune_mm_mlp=tune_mm_mlp,
        tune_mm_vision=tune_mm_vision,
        tune_mm_lm_head=tune_mm_lm_head,
    )


def build_vlm_sam_model(
    spec: VLMCheckpointSpec,
    model_ref: str,
    data_module,
    sam_checkpoint: str,
    attn_implementation: str | None = None,
    gradient_checkpointing: bool = False,
    tune_mm_llm: bool = False,
    tune_mm_mlp: bool = False,
    tune_mm_vision: bool = False,
    tune_mm_lm_head: bool = False,
):
    return VLMSam(
        config=ModelConfig(
            sam_args=SamModelArguments(False, False, True, sam_checkpoint, "sam1"),
            vlm_args=build_vlm_args(
                spec=spec,
                model_ref=model_ref,
                attn_implementation=attn_implementation,
                gradient_checkpointing=gradient_checkpointing,
                tune_mm_llm=tune_mm_llm,
                tune_mm_mlp=tune_mm_mlp,
                tune_mm_vision=tune_mm_vision,
                tune_mm_lm_head=tune_mm_lm_head,
            ),
            vlm_family=spec.family,
            sam_version="sam1",
        ),
        data_module=data_module,
    )
