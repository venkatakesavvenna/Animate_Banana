import torch
from transformers import (
    AutoModelForImageTextToText,
    Gemma3ForConditionalGeneration,
    TrainingArguments,
)

from img_2_svg_pretraining.training.training_core.models.vlms.base import VLMBase
from img_2_svg_pretraining.training.training_core.registry.registry import VLMModelRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import VLMArguments


class GemmaModel(VLMBase):
    def __init__(
        self,
        vlm_args: VLMArguments,
        training_args: TrainingArguments = None,
        gradient_checkpointing=True,
        cache_dir=None,
        bf16=True,
    ):
        super().__init__()
        self.gemma = self.get_model(vlm_args=vlm_args, cache_dir=cache_dir, bf16=bf16)

        if hasattr(self.gemma.config, "text_config") and hasattr(self.gemma.config.text_config, "use_cache"):
            self.gemma.config.text_config.use_cache = False
        elif hasattr(self.gemma.config, "use_cache"):
            self.gemma.config.use_cache = False

        if gradient_checkpointing:
            if hasattr(self.gemma, "enable_input_require_grads"):
                self.gemma.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, _input, output):
                    output.requires_grad_(True)

                self.gemma.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        self.set_model(vlm_args)

    @property
    def hidden_size(self) -> int:
        if hasattr(self.gemma.config, "text_config") and hasattr(self.gemma.config.text_config, "hidden_size"):
            return int(self.gemma.config.text_config.hidden_size)
        return int(self.gemma.config.hidden_size)

    def resize_token_embeddings(self, vocab_size: int):
        self.gemma.resize_token_embeddings(vocab_size)

    def get_model(self, vlm_args: VLMArguments, cache_dir=None, bf16: bool = True):
        attn_implementation = vlm_args.attn_implementation or "eager"
        kwargs = {
            "cache_dir": cache_dir,
            "attn_implementation": attn_implementation,
        }
        if bf16:
            kwargs["dtype"] = torch.bfloat16
        model_name = vlm_args.model_name_or_path.lower()
        if "gemma3" in model_name or "gemma-3" in model_name:
            return Gemma3ForConditionalGeneration.from_pretrained(
                vlm_args.model_name_or_path,
                **kwargs,
            )
        return AutoModelForImageTextToText.from_pretrained(
            vlm_args.model_name_or_path,
            trust_remote_code=True,
            **kwargs,
        )

    def set_model(self, model_args: VLMArguments):
        for _name, parameter in self.gemma.vision_tower.named_parameters():
            parameter.requires_grad = model_args.tune_mm_vision

        if hasattr(self.gemma, "multi_modal_projector"):
            for _name, parameter in self.gemma.multi_modal_projector.named_parameters():
                parameter.requires_grad = model_args.tune_mm_mlp

        if hasattr(self.gemma, "language_model"):
            for _name, parameter in self.gemma.language_model.named_parameters():
                parameter.requires_grad = model_args.tune_mm_llm

        for _name, parameter in self.gemma.lm_head.named_parameters():
            parameter.requires_grad = model_args.tune_mm_lm_head

    def forward(self, **kwargs):
        return self.gemma(**kwargs)

    def generate(self, **kwargs):
        return self.gemma.generate(**kwargs)


@VLMModelRegistry.register_model("gemmavlm")
def get_gemma_vlm_model(
    vlm_args: VLMArguments,
    training_args: TrainingArguments = None,
    gradient_checkpointing=True,
    cache_dir=None,
    bf16=True,
    **_kwargs,
):
    return GemmaModel(
        vlm_args=vlm_args,
        training_args=training_args,
        gradient_checkpointing=gradient_checkpointing,
        cache_dir=cache_dir,
        bf16=bf16,
    )
