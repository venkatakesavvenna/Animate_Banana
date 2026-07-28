from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    TrainingArguments,
)

from img_2_svg_pretraining.training.training_core.models.vlms.base import VLMBase
from img_2_svg_pretraining.training.training_core.registry.registry import VLMModelRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import VLMArguments


class QwenModel(VLMBase):
    def __init__(
        self,
        vlm_args: VLMArguments,
        training_args: TrainingArguments = None,
        gradient_checkpointing=True,
        cache_dir=None,
        bf16=True,
    ):
        super().__init__()
        self.qwen: Qwen2_5_VLForConditionalGeneration
        self.qwen, self.model_type = self.get_model(
            model_args=vlm_args,
            cache_dir=cache_dir,
            bf16=bf16,
        )

        self.qwen.config.text_config.use_cache = False

        if gradient_checkpointing:
            if hasattr(self.qwen, "enable_input_require_grads"):
                self.qwen.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, _input, output):
                    output.requires_grad_(True)

                self.qwen.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if training_args:
            self.set_model_training(training_args)
        else:
            self.set_model(vlm_args)

    @property
    def hidden_size(self) -> int:
        return int(self.qwen.config.text_config.hidden_size)

    def resize_token_embeddings(self, vocab_size: int):
        self.qwen.resize_token_embeddings(vocab_size)

    def get_model(self, model_args: VLMArguments, cache_dir, bf16, attn_implementation="flash_attention_2"):
        attn_implementation = model_args.attn_implementation or attn_implementation
        if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if bf16 else None),
            )
            model_type = "qwen3vl"
        elif "qwen3" in model_args.model_name_or_path.lower():
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if bf16 else None),
            )
            model_type = "qwen3vl"
        elif "qwen2.5" in model_args.model_name_or_path.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if bf16 else None),
            )
            model_type = "qwen2.5vl"
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if bf16 else None),
            )
            model_type = "qwen2vl"
        return model, model_type

    def set_model(self, model_args: VLMArguments):
        for _name, parameter in self.qwen.visual.named_parameters():
            parameter.requires_grad = model_args.tune_mm_vision

        for _name, parameter in self.qwen.visual.merger.named_parameters():
            parameter.requires_grad = model_args.tune_mm_mlp

        for _name, parameter in self.qwen.language_model.named_parameters():
            parameter.requires_grad = model_args.tune_mm_llm

        for _name, parameter in self.qwen.lm_head.named_parameters():
            parameter.requires_grad = model_args.tune_mm_lm_head

    def set_model_training(self, training_args: TrainingArguments):
        if training_args.lora_enable:
            from peft import LoraConfig, TaskType, get_peft_model

            print("LoRA enabled")
            for parameter in self.qwen.parameters():
                parameter.requires_grad = False

            lora_config = LoraConfig(
                r=training_args.lora_r or 64,
                lora_alpha=training_args.lora_alpha or 128,
                lora_dropout=training_args.lora_dropout or 0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.qwen = get_peft_model(self.qwen, lora_config)

    def forward(self, **kwargs):
        position_ids = kwargs.get("position_ids")
        if isinstance(position_ids, torch.Tensor) and position_ids.ndim == 3 and position_ids.shape[1] == 3:
            kwargs["position_ids"] = position_ids.permute(1, 0, 2).contiguous()
        return self.qwen(**kwargs)

    def generate(self, **kwargs):
        position_ids = kwargs.get("position_ids")
        if isinstance(position_ids, torch.Tensor) and position_ids.ndim == 3 and position_ids.shape[1] == 3:
            kwargs["position_ids"] = position_ids.permute(1, 0, 2).contiguous()
        return self.qwen.generate(**kwargs)


@VLMModelRegistry.register_model("qwenvl")
@VLMModelRegistry.register_model("qwenvlm")
def get_qwen_vlm_model(
    vlm_args: VLMArguments,
    training_args: TrainingArguments = None,
    gradient_checkpointing=True,
    cache_dir=None,
    bf16=True,
    **_kwargs,
):
    return QwenModel(
        vlm_args=vlm_args,
        training_args=training_args,
        gradient_checkpointing=gradient_checkpointing,
        cache_dir=cache_dir,
        bf16=bf16,
    )
