from typing import Callable

from transformers import AutoProcessor

from img_2_svg_pretraining.training.training_core.data_modules.vlms.common import (
    GenericVisionLanguageCollator,
    GenericVisionLanguageDataset,
    IGNORE_INDEX,
)
from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataArguments, DataModule


MODEL_FAMILY_NAME = "gemmavlm"


@DataModuleRegistry.register_module(MODEL_FAMILY_NAME)
def get_gemma_data_module(
    data_args: DataArguments,
    change_tokenizer_fn: Callable,
    sam_collator: Callable,
    model_name: str,
    model_path: str,
):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if not getattr(processor, "chat_template", None):
        raise ValueError("gemmavlm currently supports Gemma3 chat-template checkpoints only")
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
        Collator=GenericVisionLanguageCollator(
            processor=processor,
            tokenizer=processor.tokenizer,
            sam_collator=sam_collator,
            ignore_index=IGNORE_INDEX,
        ),
        layout_classes=data_args.layout_classes,
        family_name=MODEL_FAMILY_NAME,
        tokenizer_vocab_size=len(processor.tokenizer),
    )
