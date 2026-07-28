import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint

_logger = logging.getLogger(__name__)
from transformers import AutoModelForCausalLM, TrainingArguments

from img_2_svg_pretraining.training.training_core.models.vlms.base import VLMBase
from img_2_svg_pretraining.training.training_core.registry.registry import VLMModelRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import VLMArguments
from img_2_svg_pretraining.training.training_core.vision_encoders.adapter_utils import cast_tensor_to_module_dtype


def _enable_block_checkpointing(block: nn.Module) -> None:
    """Wrap a MolmoBlock's forward so it runs under activation checkpointing.

    Molmo-O/1B's remote-code decoder never reads any gradient_checkpointing flag
    (its forward always calls `block(...)` directly), so HF's generic
    gradient_checkpointing_enable() on the outer model is a silent no-op for these
    variants. We patch each block instance directly instead.
    """
    if not hasattr(block, "_uncheckpointed_forward"):
        block._uncheckpointed_forward = block.forward
    orig_forward = block._uncheckpointed_forward

    def checkpointed_forward(x, attention_bias=None, position_ids=None, layer_past=None, use_cache=False):
        if block.training and not use_cache:
            return torch.utils.checkpoint.checkpoint(
                orig_forward, x, attention_bias, position_ids, layer_past, use_cache,
                use_reentrant=False,
            )
        return orig_forward(x, attention_bias=attention_bias, position_ids=position_ids, layer_past=layer_past, use_cache=use_cache)

    block.forward = checkpointed_forward


def _disable_block_checkpointing(block: nn.Module) -> None:
    orig_forward = getattr(block, "_uncheckpointed_forward", None)
    if orig_forward is not None:
        block.forward = orig_forward


class MolmoModel(VLMBase):
    """VLMBase wrapper for the Molmo family (7B-D, 7B-O, MolmoE-1B).

    All variants use trust_remote_code=True.  The -D variant (allenai/Molmo-7B-D-0924)
    has a Qwen2-7B LLM backbone; -O (allenai/Molmo-7B-O-0924) and MolmoE-1B use OLMo
    backbones.  We probe several config attributes to read hidden_size correctly across
    all three.
    """

    def __init__(
        self,
        vlm_args: VLMArguments,
        training_args: TrainingArguments = None,
        gradient_checkpointing: bool = True,
        cache_dir=None,
        bf16: bool = True,
    ):
        super().__init__()
        self.molmo = self._load(vlm_args, cache_dir=cache_dir, bf16=bf16)

        # Disable KV-cache for training.
        cfg = self.molmo.config
        for sub in (getattr(cfg, "text_config", None), cfg):
            if sub is not None and hasattr(sub, "use_cache"):
                sub.use_cache = False
                break

        # Do not force input embedding outputs to require grads for Molmo. The
        # remote-code forward path injects image features into token embeddings
        # in-place, and turning those embeddings into leaf grad tensors causes
        # autograd failures on the D variant. Trainer-driven gradient
        # checkpointing still works through model.gradient_checkpointing_enable().

        self.set_model(vlm_args)

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if self._patch_decoder_blocks(enable=True):
            return
        fn = getattr(self.molmo, "gradient_checkpointing_enable", None)
        if fn is not None:
            if not getattr(self.molmo, "supports_gradient_checkpointing", False):
                self.molmo.supports_gradient_checkpointing = True
            fn(**({"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs} if gradient_checkpointing_kwargs else {}))

    def gradient_checkpointing_disable(self):
        if self._patch_decoder_blocks(enable=False):
            return
        fn = getattr(self.molmo, "gradient_checkpointing_disable", None)
        if fn is not None:
            fn()

    def _patch_decoder_blocks(self, enable: bool) -> bool:
        """Patch Molmo-O/1B's OLMo-style decoder blocks directly.

        Returns True if this checkpoint (custom `Molmo` class with
        `transformer.blocks`) was found and patched. Returns False for the -D
        variant (standard Qwen2 backbone), which already supports
        gradient_checkpointing_enable/disable natively via HF.
        """
        inner = getattr(self.molmo, "model", None)
        blocks = getattr(getattr(inner, "transformer", None), "blocks", None)
        if blocks is None:
            return False
        for block in blocks:
            if enable:
                _enable_block_checkpointing(block)
            else:
                _disable_block_checkpointing(block)
        return True

    # ------------------------------------------------------------------
    # VLMBase interface
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        cfg = self.molmo.config
        # Try text_config first (some multimodal HF configs nest the LLM config here).
        for sub in (getattr(cfg, "text_config", None), cfg):
            if sub is None:
                continue
            for attr in ("hidden_size", "d_model"):
                if hasattr(sub, attr):
                    return int(getattr(sub, attr))
        raise AttributeError(
            f"Cannot determine hidden_size from Molmo config: {cfg}"
        )

    def resize_token_embeddings(self, vocab_size: int):
        try:
            self.molmo.resize_token_embeddings(vocab_size)
        except Exception:
            # Molmo remote-code checkpoints do not consistently implement HF resize
            # semantics across D/O variants, so we always verify and patch manually.
            pass

        self._ensure_input_embeddings_capacity(vocab_size)
        self._ensure_output_embeddings_capacity(vocab_size)
        self._update_vocab_size_config(vocab_size)

    def forward(self, **kwargs):
        # The collator stores Molmo VLM images as "molmo_images" to avoid the naming
        # conflict with SAM's "images" key in VLMSam._extract_sam_inputs().
        if "molmo_images" in kwargs:
            kwargs["images"] = kwargs.pop("molmo_images")
        if "images" in kwargs:
            kwargs["images"] = self._cast_images_for_vision_backbone(kwargs["images"])
        return self.molmo(**kwargs)

    def generate(self, **kwargs):
        """Run Molmo generation via a manual auto-regressive decode loop.

        We deliberately bypass HF GenerationMixin.generate() because under
        DeepSpeed ZeRO-3 the wrapped model forward signature loses all
        Molmo-specific kwargs and HF _validate_model_kwargs raises a
        ValueError on every rank before a single token is generated.

        This loop calls self.molmo() (forward) directly, which always works
        correctly regardless of wrapper depth.
        """
        input_ids       = kwargs.get("input_ids")
        attention_mask  = kwargs.get("attention_mask")
        _images         = kwargs.get("molmo_images")
        images          = _images if _images is not None else kwargs.get("images")
        if images is not None:
            images = self._cast_images_for_vision_backbone(images)
        image_masks     = kwargs.get("image_masks")
        image_input_idx = kwargs.get("image_input_idx")
        max_new_tokens  = int(kwargs.get("max_new_tokens", 512))
        do_sample       = bool(kwargs.get("do_sample", False))
        temperature     = float(kwargs.get("temperature", 1.0))
        pad_token_id    = kwargs.get("pad_token_id")
        eos_token_id    = kwargs.get("eos_token_id")
        use_cache       = bool(kwargs.get("use_cache", False))
        synced_gpus     = bool(kwargs.get("synced_gpus", False))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        device     = next(self.molmo.parameters()).device
        batch_size = input_ids.shape[0]

        # Build position_ids (Molmo convention: cumsum of attention_mask)
        if attention_mask is not None:
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.long), dim=-1) - 1,
                min=0,
            )
        else:
            position_ids = None

        # Step 0: forward with images.
        # Under ZeRO-3, self.molmo() triggers AllGather on all ranks.  If one rank
        # throws here (e.g. OOM on a problematic image), it must signal all others
        # to bail out together — otherwise they hang waiting for the failed rank.
        _dist = None
        if synced_gpus:
            try:
                import torch.distributed as _dist_mod
                if _dist_mod.is_available() and _dist_mod.is_initialized():
                    _dist = _dist_mod
            except Exception:
                pass

        step0_ok = True
        out0 = None
        try:
            with torch.no_grad():
                out0 = self.molmo(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    images=images,
                    image_masks=image_masks,
                    image_input_idx=image_input_idx,
                    use_cache=use_cache,
                    last_logits_only=True,
                )
        except Exception:
            step0_ok = False
            _logger.exception(
                "[MolmoModel.generate] Step-0 forward failed — this rank's "
                "generation will return the unmodified input instead of a caption."
            )

        if _dist is not None:
            ok_tensor = torch.tensor(int(step0_ok), device=device)
            _dist.all_reduce(ok_tensor, op=_dist.ReduceOp.MIN)
            if ok_tensor.item() == 0:
                # At least one rank failed — all ranks return input unchanged
                return input_ids
        elif not step0_ok:
            return input_ids

        logits_step     = out0.logits[:, -1, :]
        past_key_values = out0.past_key_values if use_cache else None

        if do_sample and temperature != 1.0:
            logits_step = logits_step / temperature
        if do_sample:
            next_token = torch.multinomial(torch.softmax(logits_step, dim=-1), 1)
        else:
            next_token = logits_step.argmax(dim=-1, keepdim=True)

        generated = next_token.clone()
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            for eid in eos_token_id:
                done |= (next_token.squeeze(-1) == eid)

        cur_position_ids   = position_ids[:, -1:] + 1 if position_ids is not None else None
        cur_attention_mask = attention_mask

        for _ in range(max_new_tokens - 1):
            if not synced_gpus and done.all():
                break

            if cur_attention_mask is not None:
                cur_attention_mask = torch.cat(
                    [cur_attention_mask,
                     cur_attention_mask.new_ones((batch_size, 1))],
                    dim=-1,
                )

            with torch.no_grad():
                if use_cache:
                    step_out = self.molmo(
                        input_ids=next_token,
                        attention_mask=cur_attention_mask,
                        position_ids=cur_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        last_logits_only=True,
                    )
                else:
                    full_ids  = torch.cat([input_ids, generated], dim=-1)
                    full_mask = cur_attention_mask
                    full_pos  = (
                        torch.clamp(
                            torch.cumsum(full_mask.to(torch.long), dim=-1) - 1,
                            min=0,
                        ) if full_mask is not None else None
                    )
                    # Without a KV cache, each step recomputes the full sequence from
                    # scratch, so the image kwargs must be passed on every step, not
                    # just step 0 — otherwise the model loses all visual grounding
                    # after the first generated token and free-runs on text alone.
                    step_out = self.molmo(
                        input_ids=full_ids,
                        attention_mask=full_mask,
                        position_ids=full_pos,
                        images=images,
                        image_masks=image_masks,
                        image_input_idx=image_input_idx,
                        use_cache=False,
                        last_logits_only=True,
                    )

            logits_step = step_out.logits[:, -1, :]
            if do_sample and temperature != 1.0:
                logits_step = logits_step / temperature
            if do_sample:
                next_token = torch.multinomial(torch.softmax(logits_step, dim=-1), 1)
            else:
                next_token = logits_step.argmax(dim=-1, keepdim=True)

            if past_key_values is not None:
                past_key_values = step_out.past_key_values
            if cur_position_ids is not None:
                cur_position_ids = cur_position_ids + 1
            if pad_token_id is not None:
                next_token = next_token.masked_fill(done.unsqueeze(-1), pad_token_id)

            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None:
                for eid in eos_token_id:
                    done |= (next_token.squeeze(-1) == eid)

        return torch.cat([input_ids, generated], dim=-1)


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, vlm_args: VLMArguments, cache_dir=None, bf16: bool = True):
        kwargs = {"trust_remote_code": True, "cache_dir": cache_dir}
        if bf16:
            kwargs["torch_dtype"] = torch.bfloat16
        return AutoModelForCausalLM.from_pretrained(
            vlm_args.model_name_or_path, **kwargs
        )

    def set_model(self, model_args: VLMArguments):
        # Vision backbone — present in all Molmo variants.
        vb = self._get_vision_backbone()
        if vb is not None:
            for p in vb.parameters():
                p.requires_grad = model_args.tune_mm_vision
            # Projector lives inside vision_backbone for Molmo.
            if hasattr(vb, "image_projector"):
                for p in vb.image_projector.parameters():
                    p.requires_grad = model_args.tune_mm_mlp

        # LLM trunk — attribute name differs between -D (Qwen2) and -O / MolmoE (OLMo).
        for attr in ("model", "language_model", "transformer"):
            if hasattr(self.molmo, attr):
                for p in getattr(self.molmo, attr).parameters():
                    p.requires_grad = model_args.tune_mm_llm
                break

        if hasattr(self.molmo, "lm_head"):
            for p in self.molmo.lm_head.parameters():
                p.requires_grad = model_args.tune_mm_lm_head

    def _get_vision_backbone(self):
        if hasattr(self.molmo, "vision_backbone"):
            return self.molmo.vision_backbone
        if hasattr(self.molmo, "model") and hasattr(self.molmo.model, "vision_backbone"):
            return self.molmo.model.vision_backbone
        return None

    def _get_output_embeddings_module(self):
        if hasattr(self.molmo, "get_output_embeddings"):
            module = self.molmo.get_output_embeddings()
            if module is not None:
                return module
        if hasattr(self.molmo, "lm_head"):
            return self.molmo.lm_head
        if hasattr(self.molmo, "model") and hasattr(self.molmo.model, "transformer"):
            transformer = self.molmo.model.transformer
            if hasattr(transformer, "ff_out"):
                return transformer.ff_out
        return None

    def _set_output_embeddings_module(self, module: nn.Module):
        if hasattr(self.molmo, "set_output_embeddings"):
            self.molmo.set_output_embeddings(module)
            return
        if hasattr(self.molmo, "lm_head"):
            self.molmo.lm_head = module
            return
        if hasattr(self.molmo, "model") and hasattr(self.molmo.model, "transformer"):
            transformer = self.molmo.model.transformer
            if hasattr(transformer, "ff_out"):
                transformer.ff_out = module
                return
        raise AttributeError("Unable to set Molmo output embeddings module")

    def _current_input_vocab_size(self) -> int | None:
        emb_wrapper = self.molmo.get_input_embeddings()
        if hasattr(emb_wrapper, "weight"):
            return int(emb_wrapper.weight.shape[0])

        inner = getattr(emb_wrapper, "embedding", None)
        if inner is None or not hasattr(inner, "weight"):
            return None

        total_vocab = int(inner.weight.shape[0])
        extra = getattr(emb_wrapper, "new_embedding", None)
        if extra is not None and hasattr(extra, "weight"):
            total_vocab += int(extra.weight.shape[0])
        return total_vocab

    def _ensure_input_embeddings_capacity(self, vocab_size: int):
        current_vocab = self._current_input_vocab_size()
        if current_vocab is not None and current_vocab >= vocab_size:
            return

        emb_wrapper = self.molmo.get_input_embeddings()
        if hasattr(emb_wrapper, "weight"):
            old_weight = emb_wrapper.weight
            old_vocab, hidden = old_weight.shape
            new_embedding = nn.Embedding(
                vocab_size,
                hidden,
                dtype=old_weight.dtype,
                device=old_weight.device,
            )
            new_embedding.weight.data[:old_vocab] = old_weight.data
            new_embedding.weight.data[old_vocab:] = old_weight.data[-1:]
            if hasattr(self.molmo, "set_input_embeddings"):
                self.molmo.set_input_embeddings(new_embedding)
            return

        inner = getattr(emb_wrapper, "embedding", None)
        if inner is None or not hasattr(inner, "weight"):
            return

        base_vocab, hidden = inner.weight.shape
        needed_extra = max(vocab_size - base_vocab, 0)
        extra = getattr(emb_wrapper, "new_embedding", None)
        current_extra = int(extra.weight.shape[0]) if extra is not None and hasattr(extra, "weight") else 0
        if current_extra >= needed_extra:
            return

        new_extra = nn.Embedding(
            needed_extra,
            hidden,
            dtype=inner.weight.dtype,
            device=inner.weight.device,
        )
        if current_extra:
            new_extra.weight.data[:current_extra] = extra.weight.data
            fill_row = extra.weight.data[-1:]
        else:
            fill_row = inner.weight.data[-1:]
        new_extra.weight.data[current_extra:] = fill_row
        emb_wrapper.new_embedding = new_extra

    def _ensure_output_embeddings_capacity(self, vocab_size: int):
        out_module = self._get_output_embeddings_module()
        if out_module is None or not hasattr(out_module, "weight"):
            return
        current_vocab, hidden = out_module.weight.shape
        if current_vocab >= vocab_size:
            return

        new_head = nn.Linear(
            hidden,
            vocab_size,
            bias=out_module.bias is not None,
            dtype=out_module.weight.dtype,
            device=out_module.weight.device,
        )
        new_head.weight.data[:current_vocab] = out_module.weight.data
        new_head.weight.data[current_vocab:] = out_module.weight.data[-1:]
        if out_module.bias is not None:
            new_head.bias.data[:current_vocab] = out_module.bias.data
            new_head.bias.data[current_vocab:] = out_module.bias.data[-1:]
        self._set_output_embeddings_module(new_head)

    def _update_vocab_size_config(self, vocab_size: int):
        for cfg in (getattr(self.molmo, "config", None), getattr(getattr(self.molmo, "model", None), "config", None)):
            if cfg is None:
                continue
            # Only grow; never shrink. OLMo pads embedding_size for efficiency (e.g.
            # actual lm_head may have 100352 rows while vocab_size=100284) — shrinking
            # the config causes the remote-code forward's view(-1, embedding_size) to
            # mismatch the lm_head output width.
            if hasattr(cfg, "vocab_size") and vocab_size > cfg.vocab_size:
                cfg.vocab_size = vocab_size
            if hasattr(cfg, "embedding_size") and vocab_size > cfg.embedding_size:
                cfg.embedding_size = vocab_size

    def _cast_images_for_vision_backbone(self, images: torch.Tensor) -> torch.Tensor:
        vision_backbone = self._get_vision_backbone()
        if vision_backbone is None:
            return images

        target_module = vision_backbone
        if hasattr(vision_backbone, "image_vit"):
            target_module = vision_backbone.image_vit
        return cast_tensor_to_module_dtype(images, target_module)


@VLMModelRegistry.register_model("molmovlm")
def get_molmo_vlm_model(
    vlm_args: VLMArguments,
    training_args: TrainingArguments = None,
    gradient_checkpointing: bool = True,
    cache_dir=None,
    bf16: bool = True,
    **_kwargs,
):
    return MolmoModel(
        vlm_args=vlm_args,
        training_args=training_args,
        gradient_checkpointing=gradient_checkpointing,
        cache_dir=cache_dir,
        bf16=bf16,
    )
