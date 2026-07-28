"""Molmo pre-training weight initialization.

Config schema (all fields optional):

  pretrain_init:
    # Vision encoder
    ve_init_mode: from_vlm        # from_vlm | raw_encoder | swap_from_vlm
    ve_checkpoint: openai/clip-vit-large-patch14-336   # raw_encoder: HF ID or path
    ve_source_vlm: null            # swap_from_vlm: VLM checkpoint to extract ViT from

    # Connector
    reset_connector: true

    # LLM
    lm_init_mode: from_vlm        # from_vlm | raw_llm
    lm_checkpoint: allenai/OLMo-7B-1024-preview        # raw_llm: HF ID or path

Backward-compat aliases: clip_checkpoint → ve_checkpoint, lm_checkpoint alone implies raw_llm.
Note: raw_encoder for Molmo only supports CLIP-architecture models (ViT-L/14@336).
"""

import logging
import re

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def init_pretrain_weights(molmo_backbone, pretrain_cfg):
    hf_model = molmo_backbone.molmo  # MolmoForCausalLM

    # ── Vision encoder ───────────────────────────────────────────────────────
    ve_init_mode = pretrain_cfg.get("ve_init_mode", "from_vlm")
    # backward compat: clip_checkpoint alone implies raw_encoder
    ve_checkpoint = pretrain_cfg.get("ve_checkpoint") or pretrain_cfg.get("clip_checkpoint")
    if ve_checkpoint and ve_init_mode == "from_vlm":
        ve_init_mode = "raw_encoder"
    ve_source_vlm = pretrain_cfg.get("ve_source_vlm")

    if ve_init_mode == "raw_encoder":
        if not ve_checkpoint:
            raise ValueError("pretrain_init.ve_checkpoint is required when ve_init_mode=raw_encoder")
        _load_clip_into_vit(hf_model, ve_checkpoint)
    elif ve_init_mode == "swap_from_vlm":
        if not ve_source_vlm:
            raise ValueError("pretrain_init.ve_source_vlm is required when ve_init_mode=swap_from_vlm")
        _swap_vit_from_vlm(hf_model, ve_source_vlm)
    else:
        _logger.info("[PretrainInit] ve_init_mode=from_vlm — keeping base_model ViT weights.")

    # ── Connector ────────────────────────────────────────────────────────────
    if pretrain_cfg.get("reset_connector", True):
        _reset_connector(hf_model)

    # ── LLM ──────────────────────────────────────────────────────────────────
    lm_init_mode = pretrain_cfg.get("lm_init_mode", "from_vlm")
    # backward compat: lm_checkpoint alone implies raw_llm
    lm_checkpoint = pretrain_cfg.get("lm_checkpoint")
    if lm_checkpoint and lm_init_mode == "from_vlm":
        lm_init_mode = "raw_llm"

    if lm_init_mode == "raw_llm":
        if not lm_checkpoint:
            raise ValueError("pretrain_init.lm_checkpoint is required when lm_init_mode=raw_llm")
        _load_olmo_into_transformer(hf_model, lm_checkpoint)
    else:
        _logger.info("[PretrainInit] lm_init_mode=from_vlm — keeping base_model LLM weights.")

    _logger.info("[PretrainInit] done.")


# ---------------------------------------------------------------------------
# CLIP → Molmo image_vit
# ---------------------------------------------------------------------------

def _load_clip_into_vit(hf_model, clip_ckpt: str):
    """Load CLIP ViT-L/14@336 weights into hf_model.model.vision_backbone.image_vit."""
    _logger.info("[PretrainInit] Loading CLIP from %s", clip_ckpt)

    # Load full CLIP model (handles both CLIPModel and CLIPVisionModel checkpoints).
    # local_files_only only when given an absolute path — model IDs use HF hub cache.
    from transformers import CLIPModel, CLIPVisionModel
    _local = clip_ckpt.startswith("/")
    try:
        clip = CLIPVisionModel.from_pretrained(clip_ckpt, local_files_only=_local)
        clip_sd = {k: v for k, v in clip.state_dict().items()}
        prefix = "vision_model."
    except Exception:
        clip = CLIPModel.from_pretrained(clip_ckpt, local_files_only=_local)
        clip_sd = {k: v for k, v in clip.state_dict().items() if k.startswith("vision_model.")}
        prefix = "vision_model."

    vb = _get_vision_backbone(hf_model)
    if vb is None or not hasattr(vb, "image_vit"):
        raise RuntimeError("Cannot find model.vision_backbone.image_vit in Molmo model")
    vit = vb.image_vit
    molmo_vit_sd = vit.state_dict()

    new_sd = {}
    n_mapped = 0

    # ── Embeddings ───────────────────────────────────────────────────────────
    def _try_copy(clip_key, molmo_key, reshape_fn=None):
        nonlocal n_mapped
        full_clip_key = prefix + clip_key
        if full_clip_key not in clip_sd or molmo_key not in molmo_vit_sd:
            return
        src = clip_sd[full_clip_key]
        if reshape_fn is not None:
            src = reshape_fn(src)
        dst_shape = molmo_vit_sd[molmo_key].shape
        if src.shape != dst_shape:
            _logger.warning("[PretrainInit] shape mismatch %s: clip %s vs molmo %s — skipping",
                            molmo_key, tuple(src.shape), tuple(dst_shape))
            return
        new_sd[molmo_key] = src.clone()
        n_mapped += 1

    # class_embedding: CLIP (1024,) → Molmo (1024,)  — direct copy
    _try_copy("embeddings.class_embedding", "class_embedding")

    # positional_embedding: CLIP Embedding.weight (577, 1024) → Molmo (577, 1024)
    _try_copy("embeddings.position_embedding.weight", "positional_embedding")

    # patch_embedding: CLIP Conv2d weight (1024, 3, 14, 14) → Molmo Linear weight (1024, 588)
    _try_copy(
        "embeddings.patch_embedding.weight",
        "patch_embedding.weight",
        reshape_fn=lambda w: w.reshape(w.shape[0], -1),
    )

    # pre-layer norm
    _try_copy("pre_layrnorm.weight", "pre_ln.weight")
    _try_copy("pre_layrnorm.bias",   "pre_ln.bias")

    # ── Per-layer ────────────────────────────────────────────────────────────
    layer_indices = sorted({
        int(m.group(1))
        for k in clip_sd
        for m in [re.match(r"vision_model\.encoder\.layers\.(\d+)\.", k)]
        if m
    })

    CLIP_TO_MOLMO = {
        "self_attn.q_proj": "attention.wq",
        "self_attn.k_proj": "attention.wk",
        "self_attn.v_proj": "attention.wv",
        "self_attn.out_proj": "attention.wo",
        "layer_norm1": "attention_norm",
        "layer_norm2": "ffn_norm",
        "mlp.fc1": "feed_forward.w1",
        "mlp.fc2": "feed_forward.w2",
    }

    for i in layer_indices:
        for clip_sfx, molmo_sfx in CLIP_TO_MOLMO.items():
            for tensor_sfx in ("weight", "bias"):
                _try_copy(
                    f"encoder.layers.{i}.{clip_sfx}.{tensor_sfx}",
                    f"transformer.resblocks.{i}.{molmo_sfx}.{tensor_sfx}",
                )

    # Note: CLIP post_layernorm has no counterpart in Molmo ViT — skip.

    missing, unexpected = vit.load_state_dict(new_sd, strict=False)
    _logger.info("[PretrainInit] CLIP → Molmo ViT: %d tensors mapped | %d missing | %d unexpected",
                 n_mapped, len(missing), len(unexpected))
    if missing:
        _logger.info("[PretrainInit] ViT still-missing keys (will keep Molmo init): %s", missing)

    del clip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Reset connector to random weights
# ---------------------------------------------------------------------------

def _reset_connector(hf_model):
    """Re-initialize image_pooling_2d and image_projector with N(0, 0.02)."""
    vb = _get_vision_backbone(hf_model)
    if vb is None:
        raise RuntimeError("Cannot find vision_backbone")

    for subname in ("image_pooling_2d", "image_projector"):
        sub = getattr(vb, subname, None)
        if sub is None:
            continue
        n = 0
        for param in sub.parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(param)
            n += 1
        _logger.info("[PretrainInit] %s reset → N(0, 0.02): %d parameter tensors", subname, n)


# ---------------------------------------------------------------------------
# OLMo → Molmo transformer
# ---------------------------------------------------------------------------

def _load_olmo_into_transformer(hf_model, lm_ckpt: str):
    """Load OLMo-7B-1024-preview weights into hf_model.model.transformer."""
    _logger.info("[PretrainInit] Loading OLMo from %s", lm_ckpt)
    from transformers import AutoModelForCausalLM

    _local = lm_ckpt.startswith("/")
    olmo = AutoModelForCausalLM.from_pretrained(
        lm_ckpt,
        torch_dtype=torch.bfloat16,
        local_files_only=_local,
        trust_remote_code=True,
    )
    olmo_sd = {k: v.cpu() for k, v in olmo.state_dict().items()}

    # Detect key format
    has_hf_format = any(k.startswith("model.layers.") for k in olmo_sd)
    has_native_format = any(k.startswith("model.transformer.blocks.") for k in olmo_sd)

    xfmr = _get_transformer(hf_model)
    if xfmr is None:
        raise RuntimeError("Cannot find model.transformer in Molmo model")
    molmo_sd = xfmr.state_dict()

    new_sd = {}
    if has_hf_format:
        _logger.info("[PretrainInit] Detected HF OlmoForCausalLM key format")
        n_mapped = _map_olmo_hf_to_molmo(olmo_sd, molmo_sd, new_sd)
    elif has_native_format:
        _logger.info("[PretrainInit] Detected native OLMo key format — direct copy")
        n_mapped = 0
        for k in molmo_sd:
            if k in olmo_sd and olmo_sd[k].shape == molmo_sd[k].shape:
                new_sd[k] = olmo_sd[k]
                n_mapped += 1
    else:
        raise ValueError(
            f"Unrecognized OLMo key format. Sample keys: {list(olmo_sd.keys())[:5]}"
        )

    missing, unexpected = xfmr.load_state_dict(new_sd, strict=False)
    _logger.info("[PretrainInit] OLMo → Molmo transformer: %d tensors mapped | %d missing | %d unexpected",
                 n_mapped, len(missing), len(unexpected))
    non_image_missing = [k for k in missing if "new_embedding" not in k]
    if non_image_missing:
        _logger.warning("[PretrainInit] Unexpected missing keys: %s", non_image_missing[:15])

    del olmo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _map_olmo_hf_to_molmo(olmo_sd: dict, molmo_sd: dict, new_sd: dict) -> int:
    """Map HF OlmoForCausalLM state dict → Molmo transformer state dict.

    Handles fused projections:
      Q + K + V  (separate in OLMo HF) → att_proj (fused QKV in Molmo)
      gate + up  (separate in OLMo HF) → ff_proj  (fused gate|up in Molmo)

    Returns number of tensors mapped.
    """
    n_mapped = 0

    def _copy(src_k, dst_k, transform=None):
        nonlocal n_mapped
        if src_k not in olmo_sd or dst_k not in molmo_sd:
            return False
        src = olmo_sd[src_k]
        if transform is not None:
            src = transform(src)
        if src.shape != molmo_sd[dst_k].shape:
            _logger.warning("[PretrainInit] shape mismatch %s → %s: %s vs %s",
                            src_k, dst_k, tuple(src.shape), tuple(molmo_sd[dst_k].shape))
            return False
        new_sd[dst_k] = src.clone()
        n_mapped += 1
        return True

    # ── Embeddings ───────────────────────────────────────────────────────────
    # OLMo embed_tokens → Molmo wte.embedding (base vocab rows only)
    if "model.embed_tokens.weight" in olmo_sd and "wte.embedding" in molmo_sd:
        src = olmo_sd["model.embed_tokens.weight"]
        dst_rows = molmo_sd["wte.embedding"].shape[0]
        src_rows = src.shape[0]
        if src_rows >= dst_rows:
            new_sd["wte.embedding"] = src[:dst_rows].clone()
        else:
            # OLMo vocab is smaller — copy what fits; remaining rows keep Molmo init
            tmp = molmo_sd["wte.embedding"].clone()
            tmp[:src_rows] = src
            new_sd["wte.embedding"] = tmp
            _logger.warning("[PretrainInit] OLMo embed_tokens has %d rows, Molmo wte.embedding needs %d",
                            src_rows, dst_rows)
        n_mapped += 1

    # wte.new_embedding intentionally NOT copied — these are the 5 image tokens
    # that don't exist in OLMo; they are already randomly initialized by the
    # Molmo architecture constructor.

    # Final layer norm
    _copy("model.norm.weight", "ln_f.weight")

    # LM head
    if "lm_head.weight" in olmo_sd and "ff_out.weight" in molmo_sd:
        src = olmo_sd["lm_head.weight"]
        dst_rows = molmo_sd["ff_out.weight"].shape[0]
        src_rows = src.shape[0]
        if src_rows >= dst_rows:
            new_sd["ff_out.weight"] = src[:dst_rows].clone()
        else:
            tmp = molmo_sd["ff_out.weight"].clone()
            tmp[:src_rows] = src
            new_sd["ff_out.weight"] = tmp
        n_mapped += 1

    # ── Per-layer blocks ──────────────────────────────────────────────────────
    layer_indices = sorted({
        int(m.group(1))
        for k in olmo_sd
        for m in [re.match(r"model\.layers\.(\d+)\.", k)]
        if m
    })

    for i in layer_indices:
        blk = f"blocks.{i}"

        # OLMo-7B-1024-preview is post-norm: norm after each residual.
        # post_attention_layernorm (after attn residual) → attn_norm
        # post_feedforward_layernorm (after FFN residual) → ff_norm
        _copy(f"model.layers.{i}.post_attention_layernorm.weight", f"{blk}.attn_norm.weight")
        _copy(f"model.layers.{i}.post_feedforward_layernorm.weight", f"{blk}.ff_norm.weight")

        # Fused QKV: cat(Q, K, V) on dim 0 → att_proj
        q_k = f"model.layers.{i}.self_attn.q_proj.weight"
        k_k = f"model.layers.{i}.self_attn.k_proj.weight"
        v_k = f"model.layers.{i}.self_attn.v_proj.weight"
        dst_k = f"{blk}.att_proj.weight"
        if q_k in olmo_sd and k_k in olmo_sd and v_k in olmo_sd and dst_k in molmo_sd:
            fused = torch.cat([olmo_sd[q_k], olmo_sd[k_k], olmo_sd[v_k]], dim=0)
            if fused.shape == molmo_sd[dst_k].shape:
                new_sd[dst_k] = fused
                n_mapped += 1
            else:
                _logger.warning("[PretrainInit] att_proj shape mismatch at layer %d: fused %s vs molmo %s",
                                i, tuple(fused.shape), tuple(molmo_sd[dst_k].shape))

        # Attention output
        _copy(f"model.layers.{i}.self_attn.o_proj.weight", f"{blk}.attn_out.weight")

        # Per-head QK norms
        _copy(f"model.layers.{i}.self_attn.q_norm.weight", f"{blk}.q_norm.weight")
        _copy(f"model.layers.{i}.self_attn.k_norm.weight", f"{blk}.k_norm.weight")

        # Fused gate+up: Molmo stores [up, gate] on dim 0 → ff_proj
        gate_k = f"model.layers.{i}.mlp.gate_proj.weight"
        up_k   = f"model.layers.{i}.mlp.up_proj.weight"
        dst_k  = f"{blk}.ff_proj.weight"
        if gate_k in olmo_sd and up_k in olmo_sd and dst_k in molmo_sd:
            fused = torch.cat([olmo_sd[up_k], olmo_sd[gate_k]], dim=0)
            if fused.shape == molmo_sd[dst_k].shape:
                new_sd[dst_k] = fused
                n_mapped += 1
            else:
                _logger.warning("[PretrainInit] ff_proj shape mismatch at layer %d: fused %s vs molmo %s",
                                i, tuple(fused.shape), tuple(molmo_sd[dst_k].shape))

        # FFN down projection
        _copy(f"model.layers.{i}.mlp.down_proj.weight", f"{blk}.ff_out.weight")

    return n_mapped


# ---------------------------------------------------------------------------
# Swap ViT from another Molmo-family VLM checkpoint
# ---------------------------------------------------------------------------

def _swap_vit_from_vlm(hf_model, source_vlm_path: str):
    """Extract image_vit weights from another Molmo VLM and load them."""
    _logger.info("[PretrainInit] swap_from_vlm: loading source VLM from %s", source_vlm_path)
    from transformers import AutoModelForCausalLM
    _local = source_vlm_path.startswith("/")
    src = AutoModelForCausalLM.from_pretrained(
        source_vlm_path,
        torch_dtype=torch.bfloat16,
        local_files_only=_local,
        trust_remote_code=True,
    )
    src_vb = _get_vision_backbone(src)
    if src_vb is None or not hasattr(src_vb, "image_vit"):
        raise RuntimeError(f"Cannot find image_vit in source VLM at {source_vlm_path}")
    src_sd = {k: v.cpu() for k, v in src_vb.image_vit.state_dict().items()}
    del src
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tgt_vit = _get_vision_backbone(hf_model).image_vit
    missing, unexpected = tgt_vit.load_state_dict(src_sd, strict=False)
    _logger.info("[PretrainInit] swap_from_vlm ViT: %d tensors loaded | missing=%d unexpected=%d",
                 len(src_sd) - len(missing), len(missing), len(unexpected))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_vision_backbone(hf_model):
    if hasattr(hf_model, "vision_backbone"):
        return hf_model.vision_backbone
    m = getattr(hf_model, "model", None)
    if m is not None and hasattr(m, "vision_backbone"):
        return m.vision_backbone
    return None


def _get_transformer(hf_model):
    m = getattr(hf_model, "model", None)
    if m is not None and hasattr(m, "transformer"):
        return m.transformer
    if hasattr(hf_model, "transformer"):
        return hf_model.transformer
    return None
