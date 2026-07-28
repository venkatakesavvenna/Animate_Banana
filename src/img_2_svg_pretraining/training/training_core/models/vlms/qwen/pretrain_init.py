"""Qwen2/2.5/3 VL pre-training weight initialization.

Config schema (all fields optional):

  pretrain_init:
    # Vision encoder
    ve_init_mode: from_vlm        # from_vlm | raw_encoder | swap_from_vlm
    ve_checkpoint: google/siglip-so400m-patch14-384   # raw_encoder: HF ID or path
    ve_source_vlm: null            # swap_from_vlm: Qwen-family VLM to extract visual from
                                   #   e.g. Qwen/Qwen3-VL-7B for SigLIP2

    # Connector (visual.merger)
    reset_connector: true

    # LLM
    lm_init_mode: from_vlm        # from_vlm | raw_llm
    lm_checkpoint: Qwen/Qwen2.5-7B  # raw_llm: Qwen2ForCausalLM / Qwen2.5 base model

Notes:
  - raw_encoder maps SiglipVisionModel keys → Qwen2VisionTransformer.
    QKV is fused in Qwen (attn.qkv) but separate in SigLIP (q/k/v_proj).
    SigLIP positional embeddings are not transferred (Qwen uses RoPE).
    Layers are mapped by index; if counts differ, shared prefix is loaded.
  - swap_from_vlm loads qwen.visual (excluding merger) from another Qwen VLM.
    Use this to transplant SigLIP2 from Qwen3-VL into Qwen2.5-VL.
  - raw_llm: Qwen2ForCausalLM keys map 1:1 to Qwen2.5-VL text model.
    Vocabulary size mismatches are handled by truncation/zero-padding.
"""

import logging
import re

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def init_pretrain_weights(qwen_backbone, pretrain_cfg):
    hf_model = qwen_backbone.qwen  # Qwen2_5_VLForConditionalGeneration (or variant)

    # ── Vision encoder ───────────────────────────────────────────────────────
    ve_init_mode = pretrain_cfg.get("ve_init_mode", "from_vlm")
    ve_checkpoint = pretrain_cfg.get("ve_checkpoint")
    if ve_checkpoint and ve_init_mode == "from_vlm":
        ve_init_mode = "raw_encoder"
    ve_source_vlm = pretrain_cfg.get("ve_source_vlm")

    if ve_init_mode == "raw_encoder":
        if not ve_checkpoint:
            raise ValueError("pretrain_init.ve_checkpoint is required when ve_init_mode=raw_encoder")
        _load_siglip_into_visual(hf_model, ve_checkpoint)
    elif ve_init_mode == "swap_from_vlm":
        if not ve_source_vlm:
            raise ValueError("pretrain_init.ve_source_vlm is required when ve_init_mode=swap_from_vlm")
        _swap_visual_from_vlm(hf_model, ve_source_vlm)
    else:
        _logger.info("[PretrainInit/Qwen] ve_init_mode=from_vlm — keeping base_model visual weights.")

    # ── Connector (merger) ───────────────────────────────────────────────────
    if pretrain_cfg.get("reset_connector", True):
        _reset_qwen_merger(hf_model)

    # ── LLM ──────────────────────────────────────────────────────────────────
    lm_init_mode = pretrain_cfg.get("lm_init_mode", "from_vlm")
    lm_checkpoint = pretrain_cfg.get("lm_checkpoint")
    if lm_checkpoint and lm_init_mode == "from_vlm":
        lm_init_mode = "raw_llm"

    if lm_init_mode == "raw_llm":
        if not lm_checkpoint:
            raise ValueError("pretrain_init.lm_checkpoint is required when lm_init_mode=raw_llm")
        _load_qwen_text_model(hf_model, lm_checkpoint)
    else:
        _logger.info("[PretrainInit/Qwen] lm_init_mode=from_vlm — keeping base_model LLM weights.")

    _logger.info("[PretrainInit/Qwen] done.")


# ---------------------------------------------------------------------------
# SigLIP → Qwen2VisionTransformer
# ---------------------------------------------------------------------------

def _load_siglip_into_visual(hf_model, siglip_ckpt: str):
    """Load SiglipVisionModel weights into hf_model.visual (Qwen2VisionTransformer).

    Key differences handled:
      - QKV: SigLIP has separate q/k/v_proj; Qwen has fused attn.qkv  → cat(q, k, v)
      - Positional embeddings: SigLIP learned; Qwen uses RoPE           → skip
      - patch_embed: SigLIP Conv2d → Qwen Conv2d                        → direct copy
    """
    _logger.info("[PretrainInit/Qwen] Loading SigLIP from %s", siglip_ckpt)
    from transformers import SiglipVisionModel, SiglipModel
    _local = siglip_ckpt.startswith("/")
    try:
        siglip = SiglipVisionModel.from_pretrained(siglip_ckpt, local_files_only=_local)
        siglip_sd = {k: v for k, v in siglip.state_dict().items()}
        prefix = "vision_model."
    except Exception:
        siglip = SiglipModel.from_pretrained(siglip_ckpt, local_files_only=_local)
        siglip_sd = {k: v for k, v in siglip.state_dict().items() if k.startswith("vision_model.")}
        prefix = "vision_model."

    visual = _get_visual(hf_model)
    if visual is None:
        raise RuntimeError("Cannot find visual in Qwen model")
    qwen_visual_sd = visual.state_dict()

    new_sd = {}
    n_mapped = 0

    def _try_copy(src_key, dst_key, transform=None):
        nonlocal n_mapped
        full_src = prefix + src_key
        if full_src not in siglip_sd or dst_key not in qwen_visual_sd:
            return
        src = siglip_sd[full_src]
        if transform is not None:
            src = transform(src)
        if src.shape != qwen_visual_sd[dst_key].shape:
            _logger.warning("[PretrainInit/Qwen] shape mismatch %s → %s: %s vs %s",
                            dst_key, dst_key, tuple(src.shape), tuple(qwen_visual_sd[dst_key].shape))
            return
        new_sd[dst_key] = src.clone()
        n_mapped += 1

    # Patch embedding (Conv2d → Conv2d; shapes must match)
    _try_copy("embeddings.patch_embedding.weight", "patch_embed.proj.weight")
    _try_copy("embeddings.patch_embedding.bias",   "patch_embed.proj.bias")

    # Per-layer mapping
    siglip_layer_indices = sorted({
        int(m.group(1))
        for k in siglip_sd
        for m in [re.match(r"vision_model\.encoder\.layers\.(\d+)\.", k)]
        if m
    })
    qwen_layer_indices = sorted({
        int(m.group(1))
        for k in qwen_visual_sd
        for m in [re.match(r"blocks\.(\d+)\.", k)]
        if m
    })
    n_layers = min(len(siglip_layer_indices), len(qwen_layer_indices))
    if len(siglip_layer_indices) != len(qwen_layer_indices):
        _logger.warning(
            "[PretrainInit/Qwen] SigLIP has %d layers, Qwen visual has %d — mapping first %d",
            len(siglip_layer_indices), len(qwen_layer_indices), n_layers,
        )

    for idx in range(n_layers):
        si = siglip_layer_indices[idx]
        qi = qwen_layer_indices[idx]

        # Layer norms
        _try_copy(f"encoder.layers.{si}.layer_norm1.weight", f"blocks.{qi}.norm1.weight")
        _try_copy(f"encoder.layers.{si}.layer_norm1.bias",   f"blocks.{qi}.norm1.bias")
        _try_copy(f"encoder.layers.{si}.layer_norm2.weight", f"blocks.{qi}.norm2.weight")
        _try_copy(f"encoder.layers.{si}.layer_norm2.bias",   f"blocks.{qi}.norm2.bias")

        # Fused QKV: cat(q, k, v) dim=0
        for suffix, dst_suffix in [("weight", "weight"), ("bias", "bias")]:
            q_k = f"{prefix}encoder.layers.{si}.self_attn.q_proj.{suffix}"
            k_k = f"{prefix}encoder.layers.{si}.self_attn.k_proj.{suffix}"
            v_k = f"{prefix}encoder.layers.{si}.self_attn.v_proj.{suffix}"
            dst_k = f"blocks.{qi}.attn.qkv.{suffix}"
            if q_k in siglip_sd and k_k in siglip_sd and v_k in siglip_sd and dst_k in qwen_visual_sd:
                fused = torch.cat([siglip_sd[q_k], siglip_sd[k_k], siglip_sd[v_k]], dim=0)
                if fused.shape == qwen_visual_sd[dst_k].shape:
                    new_sd[dst_k] = fused
                    n_mapped += 1
                else:
                    _logger.warning("[PretrainInit/Qwen] QKV shape mismatch at layer %d: %s vs %s",
                                    qi, tuple(fused.shape), tuple(qwen_visual_sd[dst_k].shape))

        # Output projection
        _try_copy(f"encoder.layers.{si}.self_attn.out_proj.weight", f"blocks.{qi}.attn.proj.weight")
        _try_copy(f"encoder.layers.{si}.self_attn.out_proj.bias",   f"blocks.{qi}.attn.proj.bias")

        # FFN
        _try_copy(f"encoder.layers.{si}.mlp.fc1.weight", f"blocks.{qi}.mlp.fc1.weight")
        _try_copy(f"encoder.layers.{si}.mlp.fc1.bias",   f"blocks.{qi}.mlp.fc1.bias")
        _try_copy(f"encoder.layers.{si}.mlp.fc2.weight", f"blocks.{qi}.mlp.fc2.weight")
        _try_copy(f"encoder.layers.{si}.mlp.fc2.bias",   f"blocks.{qi}.mlp.fc2.bias")

    # SigLIP post_layernorm → Qwen2VisionTransformer has no global post-norm; skip.
    # SigLIP learned position_embedding → Qwen uses RoPE; skip.

    missing, unexpected = visual.load_state_dict(new_sd, strict=False)
    merger_missing = [k for k in missing if k.startswith("merger.")]
    non_merger_missing = [k for k in missing if not k.startswith("merger.")]
    _logger.info("[PretrainInit/Qwen] SigLIP → Qwen visual: %d mapped | %d non-merger missing | %d merger missing",
                 n_mapped, len(non_merger_missing), len(merger_missing))
    if non_merger_missing:
        _logger.info("[PretrainInit/Qwen] non-merger missing (will keep VLM init): %s", non_merger_missing[:10])

    del siglip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Swap visual from another Qwen-family VLM (e.g. Qwen3-VL SigLIP2 → Qwen2.5-VL)
# ---------------------------------------------------------------------------

def _swap_visual_from_vlm(hf_model, source_vlm_path: str):
    """Extract visual ViT weights (excluding merger) from another Qwen VLM."""
    _logger.info("[PretrainInit/Qwen] swap_from_vlm: loading source from %s", source_vlm_path)
    from transformers import AutoModelForCausalLM
    _local = source_vlm_path.startswith("/")
    src = AutoModelForCausalLM.from_pretrained(
        source_vlm_path,
        torch_dtype=torch.bfloat16,
        local_files_only=_local,
        trust_remote_code=True,
    )
    src_visual = _get_visual(src)
    if src_visual is None:
        raise RuntimeError(f"Cannot find visual in source VLM at {source_vlm_path}")

    # Copy all visual weights except merger (merger will be reset afterward)
    src_sd = {
        k: v.cpu()
        for k, v in src_visual.state_dict().items()
        if not k.startswith("merger.")
    }
    del src
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tgt_visual = _get_visual(hf_model)
    missing, unexpected = tgt_visual.load_state_dict(src_sd, strict=False)
    merger_missing = [k for k in missing if k.startswith("merger.")]
    non_merger_missing = [k for k in missing if not k.startswith("merger.")]
    _logger.info("[PretrainInit/Qwen] swap_from_vlm: %d loaded | %d non-merger missing | %d merger (expected)",
                 len(src_sd) - len(non_merger_missing), len(non_merger_missing), len(merger_missing))


# ---------------------------------------------------------------------------
# Reset merger (connector)
# ---------------------------------------------------------------------------

def _reset_qwen_merger(hf_model):
    """Re-initialize visual.merger with N(0, 0.02)."""
    visual = _get_visual(hf_model)
    if visual is None:
        raise RuntimeError("Cannot find visual in Qwen model")
    merger = getattr(visual, "merger", None)
    if merger is None:
        _logger.warning("[PretrainInit/Qwen] No merger found — skipping connector reset.")
        return
    n = 0
    for param in merger.parameters():
        if param.dim() >= 2:
            nn.init.normal_(param, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(param)
        n += 1
    _logger.info("[PretrainInit/Qwen] merger reset → N(0, 0.02): %d parameter tensors", n)


# ---------------------------------------------------------------------------
# Qwen2ForCausalLM → Qwen2.5-VL text model
# ---------------------------------------------------------------------------

def _load_qwen_text_model(hf_model, lm_ckpt: str):
    """Load Qwen2/2.5 base LLM weights into the Qwen2.5-VL text model.

    Both share the same Qwen2 transformer architecture — keys map 1:1.
    Vocabulary size mismatches are handled by truncation or zero-padding.
    """
    _logger.info("[PretrainInit/Qwen] Loading Qwen LLM from %s", lm_ckpt)
    from transformers import AutoModelForCausalLM
    _local = lm_ckpt.startswith("/")
    lm = AutoModelForCausalLM.from_pretrained(
        lm_ckpt,
        torch_dtype=torch.bfloat16,
        local_files_only=_local,
    )
    lm_sd = {k: v.cpu() for k, v in lm.state_dict().items()}
    del lm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    new_sd = {}
    n_mapped = 0

    def _copy(src_k, dst_k, dst_sd):
        nonlocal n_mapped
        if src_k not in lm_sd or dst_k not in dst_sd:
            return False
        new_sd[dst_k] = lm_sd[src_k].clone()
        n_mapped += 1
        return True

    # Gather target state dicts from the text model and lm_head
    target_model = hf_model.model
    target_sd = target_model.state_dict()
    lm_head_sd = hf_model.lm_head.state_dict() if hasattr(hf_model, "lm_head") else {}

    # Embedding: handle vocab size mismatch
    if "model.embed_tokens.weight" in lm_sd and "embed_tokens.weight" in target_sd:
        src = lm_sd["model.embed_tokens.weight"]
        dst_rows = target_sd["embed_tokens.weight"].shape[0]
        src_rows = src.shape[0]
        if src_rows >= dst_rows:
            new_sd["embed_tokens.weight"] = src[:dst_rows].clone()
        else:
            tmp = target_sd["embed_tokens.weight"].clone()
            tmp[:src_rows] = src
            new_sd["embed_tokens.weight"] = tmp
            _logger.warning("[PretrainInit/Qwen] embed_tokens: src %d rows < dst %d rows — zero-padded tail",
                            src_rows, dst_rows)
        n_mapped += 1

    # Final norm
    _copy("model.norm.weight", "norm.weight", target_sd)

    # All transformer layers — direct key copy (strip "model." prefix from lm_sd)
    layer_indices = sorted({
        int(m.group(1))
        for k in lm_sd
        for m in [re.match(r"model\.layers\.(\d+)\.", k)]
        if m
    })
    for i in layer_indices:
        for suffix in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.k_proj.weight", "self_attn.k_proj.bias",
            "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            _copy(f"model.layers.{i}.{suffix}", f"layers.{i}.{suffix}", target_sd)

    missing, unexpected = target_model.load_state_dict(new_sd, strict=False)
    _logger.info("[PretrainInit/Qwen] Qwen LLM → text model: %d mapped | %d missing | %d unexpected",
                 n_mapped, len(missing), len(unexpected))
    if missing:
        _logger.warning("[PretrainInit/Qwen] missing keys: %s", missing[:10])

    # LM head
    if "lm_head.weight" in lm_sd and "weight" in lm_head_sd:
        src = lm_sd["lm_head.weight"]
        dst_rows = lm_head_sd["weight"].shape[0]
        src_rows = src.shape[0]
        if src_rows >= dst_rows:
            lm_head_new = {"weight": src[:dst_rows].clone()}
        else:
            tmp = lm_head_sd["weight"].clone()
            tmp[:src_rows] = src
            lm_head_new = {"weight": tmp}
        missing_h, _ = hf_model.lm_head.load_state_dict(lm_head_new, strict=False)
        _logger.info("[PretrainInit/Qwen] lm_head loaded (vocab %d → %d) | missing=%d",
                     src_rows, dst_rows, len(missing_h))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_visual(hf_model):
    """Return the Qwen2VisionTransformerPretrainModel from any Qwen VL variant."""
    if hasattr(hf_model, "visual"):
        return hf_model.visual
    m = getattr(hf_model, "model", None)
    if m is not None and hasattr(m, "visual"):
        return m.visual
    return None
