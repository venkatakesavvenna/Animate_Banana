"""VLM-only composite model — no SAM segmentation head.

Used when sam_version is 'none'.  Structurally mirrors VLMSam so that the
encoder-swap builder (swap_vision_encoder) works identically: both expose a
.backbone attribute of type VLMBase.

Forward:
  - Pops any SAM-keyed tensors left in kwargs by the noop collator (no-ops).
  - Calls VLM backbone → CE loss * ce_loss_weight.
  - Returns {"loss": ..., "logits": ...} which the base HF Trainer handles
    correctly during both training and eval steps.

Generate:
  - Proxies to backbone.generate(), stripping SAM keys.
  - Returns (List[str], None) — decoded predictions and None for pred_masks,
    matching the VLMSam.generate() return contract so callers can branch on
    ``pred_masks is None`` to skip SAM-specific logic.
"""
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedModel

from img_2_svg_pretraining.training.training_core.models.vlms.base import VLMBase
from img_2_svg_pretraining.training.training_core.registry.registry import VLMModelRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, ModelConfig, VLMArguments

_SAM_KEYS = {
    "images",
    "masks",
    "mask_counts",
    "resize_list",
    "orig_image_size_list",
    "debug_original_image_list",
}

# Keys that are safe to pass to backbone.generate()
_GENERATE_ALLOWED_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "position_ids",
    "token_type_ids",
    # Molmo-specific VLM image keys
    "molmo_images",
    "image_input_idx",
    "image_masks",
}

# Generation-control kwargs. These must be let through as-is (not just
# defaulted) — without this, a caller-specified use_cache/synced_gpus is
# silently dropped and Molmo's manual decode loop falls back to
# use_cache=False, synced_gpus=False regardless of what was actually
# requested: no KV cache (image conditioning is then lost after the first
# generated token — see MolmoModel.generate), and no ZeRO-3 straggler
# protection (a rank whose local batch finishes early can exit the loop
# while other ranks are still mid-generation, deadlocking on the next
# sharded-parameter collective).
_GENERATE_CONTROL_KEYS = {
    "max_new_tokens",
    "do_sample",
    "temperature",
    "pad_token_id",
    "eos_token_id",
    "use_cache",
    "synced_gpus",
}


class VLMOnly(PreTrainedModel):
    """VLM backbone wrapper for CE-loss-only (no segmentation) training."""

    accepts_loss_kwargs = False
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: ModelConfig,
        data_module: DataModule,
        backbone: VLMBase | None = None,
    ):
        super().__init__(config)

        self.processor = data_module.processor
        vlm_gc = bool(config.vlm_args.get("gradient_checkpointing", False))
        self.backbone = backbone or VLMModelRegistry.get_model(
            config.vlm_family,
            vlm_args=VLMArguments(**config.vlm_args),
            training_args=None,
            gradient_checkpointing=vlm_gc,
            cache_dir=None,
            bf16=True,
        )
        self.ce_loss_weight = config.ce_loss_weight
        self._maybe_resize_backbone_embeddings(data_module)

    # ------------------------------------------------------------------
    # Gradient checkpointing delegation (same pattern as VLMSam)
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        backbone_fn = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if backbone_fn is not None:
            backbone_fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            return
        for child in self.backbone.children():
            fn = getattr(child, "gradient_checkpointing_enable", None)
            if fn is not None:
                # Force the flag to True for remote-code models (like Molmo) that 
                # implement gradient checkpointing but forget to set the class attribute
                if not getattr(child, "supports_gradient_checkpointing", False):
                    child.supports_gradient_checkpointing = True
                if not hasattr(child, "gradient_checkpointing"):
                    child.gradient_checkpointing = False
                fn(**({"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs} if gradient_checkpointing_kwargs else {}))

    def gradient_checkpointing_disable(self):
        backbone_fn = getattr(self.backbone, "gradient_checkpointing_disable", None)
        if backbone_fn is not None:
            backbone_fn()
            return
        for child in self.backbone.children():
            fn = getattr(child, "gradient_checkpointing_disable", None)
            if fn is not None:
                fn()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        for key in _SAM_KEYS:
            kwargs.pop(key, None)

        kwargs.setdefault("return_dict", True)
        vlm_out = self.backbone(**kwargs)

        loss = vlm_out.loss * self.ce_loss_weight
        return {"loss": loss, "logits": vlm_out.logits}

    # ------------------------------------------------------------------
    # Generation  (no SAM head — returns text only, masks=None)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, **kwargs) -> tuple:
        """Generate text from the VLM backbone (no segmentation).

        Strips SAM-specific keys and any keys not accepted by the backbone's
        generate() method, then decodes the output token sequences.

        Returns:
            (preds, None) — List[str] of decoded predictions and None for
            pred_masks.  The None sentinel lets callers branch on
            ``pred_masks is None`` to skip mask-related logic, keeping the
            same call-site contract as ``VLMSam.generate()``.
        """
        self.eval()

        # Strip SAM keys
        for key in _SAM_KEYS:
            kwargs.pop(key, None)

        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")

        generate_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in (_GENERATE_ALLOWED_KEYS | _GENERATE_CONTROL_KEYS) and value is not None
        }
        generate_kwargs.setdefault("do_sample", False)
        generate_kwargs.setdefault("max_new_tokens", 512)
        generate_kwargs.setdefault(
            "pad_token_id",
            self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
        )
        eos_token_ids = self._build_eos_token_ids()
        if eos_token_ids:
            generate_kwargs.setdefault(
                "eos_token_id",
                eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0],
            )

        generate_outputs = self.backbone.generate(**generate_kwargs)

        # Handle both raw tensor and GenerateOutput objects
        sequences = generate_outputs if isinstance(generate_outputs, torch.Tensor) else generate_outputs.sequences

        # generated tokens always start at input_ids.shape[1] regardless of
        # padding side — attention_mask.sum() gives the actual non-padded length
        # which, for left-padded batches, includes tail-of-prompt tokens (e.g.
        # " Assistant:") in the decode output for shorter samples.
        if input_ids is not None and isinstance(input_ids, torch.Tensor):
            input_seq_len = input_ids.shape[1]
            continuations = [sequences[i, input_seq_len:] for i in range(sequences.shape[0])]
        else:
            continuations = [sequences[i, :] for i in range(sequences.shape[0])]
        preds = self.processor.tokenizer.batch_decode(continuations, skip_special_tokens=True)

        return preds, None  # None = no pred_masks

    def _build_eos_token_ids(self) -> List[int]:
        eos_ids: List[int] = []
        tokenizer = self.processor.tokenizer
        if tokenizer.eos_token_id is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        for token in ("<|im_end|>", "<end_of_turn>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == tokenizer.unk_token_id:
                continue
            token_id = int(token_id)
            if token_id not in eos_ids:
                eos_ids.append(token_id)
        return eos_ids

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_resize_backbone_embeddings(self, data_module: DataModule):
        if data_module.tokenizer_vocab_size is None:
            return
        self.backbone.resize_token_embeddings(data_module.tokenizer_vocab_size)

