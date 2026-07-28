from typing import Dict, List

import torch
from torch import nn
from transformers import PreTrainedModel

from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import unpack_mask_batch
from img_2_svg_pretraining.training.training_core.models.vlms.base import VLMBase
from img_2_svg_pretraining.training.training_core.models.sam.base import SAMModelBase
from img_2_svg_pretraining.training.training_core.registry.registry import SAMModelRegistry, VLMModelRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, ModelConfig, SamModelArguments, VLMArguments


class VLMSam(PreTrainedModel):
    accepts_loss_kwargs = False
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: ModelConfig,
        data_module: DataModule,
        backbone: VLMBase | None = None,
        sam_head: SAMModelBase | None = None,
    ):
        super().__init__(config)

        self.processor = data_module.processor
        vlm_gradient_checkpointing = bool(config.vlm_args.get("gradient_checkpointing", False))
        self.backbone = backbone or VLMModelRegistry.get_model(
            config.vlm_family,
            vlm_args=VLMArguments(**config.vlm_args),
            training_args=None,
            gradient_checkpointing=vlm_gradient_checkpointing,
            cache_dir=None,
            bf16=True,
        )
        self.sam_head = sam_head or SAMModelRegistry.get_sam(
            config.sam_version,
            model_args=SamModelArguments(**config.sam_args),
        )

        in_dim = self.backbone.hidden_size
        out_dim = self.sam_head.prompt_embed_dim

        text_fc_layout = [
            nn.Linear(in_dim, in_dim, dtype=torch.bfloat16),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim, dtype=torch.bfloat16),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs_layout = nn.ModuleList([nn.Sequential(*text_fc_layout)])

        for parameter in self.text_hidden_fcs_layout.parameters():
            parameter.requires_grad = True

        self.seg_token_idx = data_module.seg_token_idx
        self.ce_loss_weight = config.ce_loss_weight
        self.dice_loss_weight = config.dice_loss_weight
        self.bce_loss_weight = config.bce_loss_weight
        self.mask_weight = 0.75

        self._maybe_resize_backbone_embeddings(data_module)

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

    def forward(self, **kwargs):
        sam_inputs = self._extract_sam_inputs(kwargs)
        input_ids = kwargs["input_ids"]
        seg_token_mask = input_ids == self.seg_token_idx

        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("return_dict", True)
        vlm_out = self.backbone(**kwargs)

        ce_loss = vlm_out.loss * self.ce_loss_weight
        output_hidden_states = vlm_out.hidden_states

        last_hidden_state_layout = self.text_hidden_fcs_layout[0](output_hidden_states[-1])
        pred_embeddings_layout = self.extract_pred_embeddings(last_hidden_state_layout, seg_token_mask)

        pred_masks_layout, bce, dice = self.sam_head(
            images=sam_inputs["images"],
            masks=sam_inputs["masks"],
            vlm_seg_hidden_states=pred_embeddings_layout,
            resize_list=sam_inputs["resize_list"],
            label_list=sam_inputs["orig_image_size_list"],
            mask_counts=sam_inputs["mask_counts"],
        )
        total_mask_loss = self.dice_loss_weight * dice + self.bce_loss_weight * bce if bce is not None and dice is not None else None
        total_loss = ce_loss + (total_mask_loss * self.mask_weight) if total_mask_loss is not None else ce_loss

        return_dict: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_loss": (total_mask_loss * self.mask_weight) if total_mask_loss is not None else torch.tensor(0.0, device=ce_loss.device),
        }

        if not self.training:
            gt_masks = unpack_mask_batch(
                sam_inputs["masks"],
                mask_counts=sam_inputs["mask_counts"],
                image_size_list=sam_inputs["orig_image_size_list"],
            )
            return_dict.update(
                {
                    "logits": vlm_out.logits,
                    "pred_masks": pred_masks_layout,
                    "gt_masks": gt_masks,
                    "debug_original_images": sam_inputs["debug_original_image_list"],
                }
            )
        return return_dict

    def extract_pred_embeddings(self, last_hidden_state_local, seg_token_mask):
        assert last_hidden_state_local.size()[0] == seg_token_mask.size()[0]
        pred_embeddings = []
        for index in range(len(last_hidden_state_local)):
            pred_embeddings.append(last_hidden_state_local[index][seg_token_mask[index]])
        return pred_embeddings

    @torch.no_grad()
    def generate(self, **kwargs):
        self.eval()

        sam_inputs = self._extract_sam_inputs(kwargs)
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs.get("attention_mask")

        generate_kwargs = self._build_generate_kwargs(kwargs)
        generate_outputs = self.backbone.generate(**generate_kwargs)

        sequences = generate_outputs.sequences
        prompt_lengths = self._get_prompt_lengths(input_ids=input_ids, attention_mask=attention_mask)
        continuation_mask = self._build_continuation_seg_mask(sequences, prompt_lengths)
        continuations = self._slice_continuations(sequences, prompt_lengths)
        preds = self.processor.tokenizer.batch_decode(continuations, skip_special_tokens=True)

        generated_hidden_states = self._extract_generated_hidden_states(
            hidden_states=generate_outputs.hidden_states,
            target_length=continuation_mask.shape[1],
            batch_size=sequences.shape[0],
            hidden_size=self.backbone.hidden_size,
            device=sequences.device,
            dtype=self.text_hidden_fcs_layout[0][0].weight.dtype,
        )

        last_hidden_state_layout = self.text_hidden_fcs_layout[0](generated_hidden_states)
        pred_embeddings_layout = self.extract_pred_embeddings(last_hidden_state_layout, continuation_mask)

        pred_masks_layout, _, _ = self.sam_head(
            images=sam_inputs["images"],
            masks=None,
            vlm_seg_hidden_states=pred_embeddings_layout,
            resize_list=sam_inputs["resize_list"],
            label_list=sam_inputs["orig_image_size_list"],
            mask_counts=None,
        )
        return preds, pred_masks_layout

    def _maybe_resize_backbone_embeddings(self, data_module: DataModule):
        if data_module.tokenizer_vocab_size is None:
            return
        self.backbone.resize_token_embeddings(data_module.tokenizer_vocab_size)

    @staticmethod
    def _extract_sam_inputs(kwargs: Dict[str, torch.Tensor | List[torch.Tensor]]):
        sam_keys = {
            "images",
            "masks",
            "mask_counts",
            "resize_list",
            "orig_image_size_list",
            "debug_original_image_list",
        }
        sam_inputs = {}
        for key in sam_keys:
            sam_inputs[key] = kwargs.pop(key, None)
        return sam_inputs

    def _build_generate_kwargs(self, kwargs: Dict[str, torch.Tensor | List[torch.Tensor]]):
        allowed_keys = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "token_type_ids",
            # Molmo-specific VLM image keys (renamed from "images" to avoid SAM conflict)
            "molmo_images",
            "image_input_idx",
            "image_masks",
        }
        # Generation-control kwargs must be let through as-is, not just
        # defaulted — otherwise a caller-specified use_cache/synced_gpus is
        # silently dropped (see the matching comment in vlm_only.py).
        control_keys = {
            "max_new_tokens", "do_sample", "temperature",
            "pad_token_id", "eos_token_id", "use_cache", "synced_gpus",
        }
        generate_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in (allowed_keys | control_keys) and value is not None
        }
        generate_kwargs.setdefault("do_sample", False)
        generate_kwargs.setdefault("max_new_tokens", 512)
        generate_kwargs.setdefault("output_hidden_states", True)
        generate_kwargs.setdefault("return_dict_in_generate", True)
        generate_kwargs.setdefault("output_logits", True)
        generate_kwargs.setdefault("pad_token_id", self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id)
        eos_token_ids = self._build_eos_token_ids()
        if eos_token_ids:
            generate_kwargs.setdefault("eos_token_id", eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0])
        return generate_kwargs

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

    @staticmethod
    def _get_prompt_lengths(input_ids: torch.LongTensor, attention_mask: torch.Tensor | None) -> List[int]:
        # generated tokens always start at input_ids.shape[1]; using
        # attention_mask.sum() would include tail-of-prompt tokens for
        # shorter (padded) samples in a left-padded batch.
        return [input_ids.shape[1]] * input_ids.shape[0]

    def _build_continuation_seg_mask(self, sequences: torch.LongTensor, prompt_lengths: List[int]) -> torch.BoolTensor:
        continuation_masks = []
        for index, prompt_len in enumerate(prompt_lengths):
            continuation_masks.append(sequences[index, prompt_len:] == self.seg_token_idx)
        return torch.stack(continuation_masks, dim=0) if continuation_masks else sequences.new_zeros((0, 0), dtype=torch.bool)

    @staticmethod
    def _slice_continuations(sequences: torch.LongTensor, prompt_lengths: List[int]) -> List[torch.LongTensor]:
        return [sequences[index, prompt_len:] for index, prompt_len in enumerate(prompt_lengths)]

    @staticmethod
    def _extract_generated_hidden_states(
        hidden_states,
        target_length: int,
        batch_size: int,
        hidden_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if target_length == 0:
            return torch.zeros((batch_size, 0, hidden_size), device=device, dtype=dtype)
        if hidden_states is None:
            raise ValueError("generate() did not return hidden states; set output_hidden_states=True")

        generated_steps = []
        for step_hidden_states in hidden_states[1:]:
            last_hidden_state = step_hidden_states[-1]
            if last_hidden_state.ndim != 3:
                raise ValueError(f"Expected generated hidden states to be rank 3, got {last_hidden_state.ndim}")
            if last_hidden_state.shape[1] != 1:
                last_hidden_state = last_hidden_state[:, -1:, :]
            generated_steps.append(last_hidden_state.to(dtype=dtype))

        if generated_steps:
            generated_hidden = torch.cat(generated_steps, dim=1)
        else:
            generated_hidden = torch.zeros((batch_size, 0, hidden_size), device=device, dtype=dtype)

        if generated_hidden.shape[1] < target_length:
            pad = torch.zeros(
                (batch_size, target_length - generated_hidden.shape[1], hidden_size),
                device=device,
                dtype=generated_hidden.dtype,
            )
            generated_hidden = torch.cat([generated_hidden, pad], dim=1)
        elif generated_hidden.shape[1] > target_length:
            generated_hidden = generated_hidden[:, :target_length, :]

        return generated_hidden
