"""Gemma3 VLM pre-training weight initialization.

Config schema (all fields optional):

  pretrain_init:
    # Vision encoder
    ve_init_mode: from_vlm        # from_vlm | raw_encoder | swap_from_vlm
    ve_checkpoint: google/siglip-so400m-patch14-384   # raw_encoder: HF ID or path
    ve_source_vlm: null            # swap_from_vlm: Gemma3-family VLM to extract vision_tower from

    # Connector (multi_modal_projector)
    reset_connector: true

    # LLM
    lm_init_mode: from_vlm        # from_vlm | raw_llm
    lm_checkpoint: null            # raw_llm: Gemma3ForCausalLM path, or multimodal checkpoint

Notes:
  - raw_encoder: SiglipVisionModel keys map 1:1 to Gemma3 vision_tower (both use
    vision_model.* prefix). Position embeddings are skipped on size mismatch
    (standalone SigLIP 384px has 576 positions; Gemma3 896px has 4096).
    All encoder layer weights transfer directly when hidden_size matches (1152).
  - swap_from_vlm: extract vision_tower state dict from another Gemma3 VLM checkpoint
    and load into this model. Use for cross-size transplants (e.g. 27B SigLIP → 4B).
  - raw_llm: loads Gemma3ForCausalLM if checkpoint is text-only; if the checkpoint is
    a multimodal Gemma3 (e.g. google/gemma-3-4b-it), language_model is extracted
    automatically. Keys are identical between standalone and VLM text models — no
    key remapping needed. Vocab size mismatches are truncated/zero-padded.
  - Gemma3 has no standalone text-only instruct checkpoint separate from the multimodal
    IT checkpoint. PT replication therefore typically uses ve_init_mode=from_vlm +
    lm_init_mode=from_vlm + reset_connector=true (connector learned from scratch).
"""

import logging

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def init_pretrain_weights(gemma_backbone, pretrain_cfg):
    hf_model = gemma_backbone.gemma  # Gemma3ForConditionalGeneration

    # ── Vision encoder ───────────────────────────────────────────────────────
    ve_init_mode = pretrain_cfg.get("ve_init_mode", "from_vlm")
    ve_checkpoint = pretrain_cfg.get("ve_checkpoint")
    if ve_checkpoint and ve_init_mode == "from_vlm":
        ve_init_mode = "raw_encoder"
    ve_source_vlm = pretrain_cfg.get("ve_source_vlm")

    if ve_init_mode == "raw_encoder":
        if not ve_checkpoint:
            raise ValueError("pretrain_init.ve_checkpoint is required when ve_init_mode=raw_encoder")
        _load_siglip_into_vision_tower(hf_model, ve_checkpoint)
    elif ve_init_mode == "swap_from_vlm":
        if not ve_source_vlm:
            raise ValueError("pretrain_init.ve_source_vlm is required when ve_init_mode=swap_from_vlm")
        _swap_vision_tower_from_vlm(hf_model, ve_source_vlm)
    else:
        _logger.info("[PretrainInit/Gemma] ve_init_mode=from_vlm — keeping base_model vision_tower weights.")

    # ── Connector (multi_modal_projector) ────────────────────────────────────
    if pretrain_cfg.get("reset_connector", True):
        _reset_gemma_projector(hf_model)

    # ── LLM ──────────────────────────────────────────────────────────────────
    lm_init_mode = pretrain_cfg.get("lm_init_mode", "from_vlm")
    lm_checkpoint = pretrain_cfg.get("lm_checkpoint")
    if lm_checkpoint and lm_init_mode == "from_vlm":
        lm_init_mode = "raw_llm"

    if lm_init_mode == "raw_llm":
        if not lm_checkpoint:
            raise ValueError("pretrain_init.lm_checkpoint is required when lm_init_mode=raw_llm")
        _load_gemma3_text_model(hf_model, lm_checkpoint)
    else:
        _logger.info("[PretrainInit/Gemma] lm_init_mode=from_vlm — keeping base_model LLM weights.")

    _logger.info("[PretrainInit/Gemma] done.")


# ---------------------------------------------------------------------------
# SigLIP → Gemma3 vision_tower
# ---------------------------------------------------------------------------

def _load_siglip_into_vision_tower(hf_model, siglip_ckpt: str):
    """Load SiglipVisionModel weights into hf_model.vision_tower.

    Both use the same vision_model.* key prefix, so keys map 1:1.
    Position embeddings are skipped when sizes differ (standalone 384px → 576
    positions vs Gemma3 896px → 4096 positions). All other weights transfer
    directly when hidden_size matches (1152 for SigLIP-SO400M and Gemma3-4B).
    """
    _logger.info("[PretrainInit/Gemma] Loading SigLIP from %s", siglip_ckpt)
    from transformers import SiglipVisionModel, SiglipModel
    _local = siglip_ckpt.startswith("/")
    try:
        siglip = SiglipVisionModel.from_pretrained(siglip_ckpt, local_files_only=_local)
        src_sd = {k: v.cpu() for k, v in siglip.state_dict().items()}
    except Exception:
        siglip = SiglipModel.from_pretrained(siglip_ckpt, local_files_only=_local)
        src_sd = {k: v.cpu() for k, v in siglip.state_dict().items()
                  if k.startswith("vision_model.")}
    del siglip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vt = hf_model.vision_tower
    tgt_sd = vt.state_dict()
    new_sd = {}
    n_mapped = 0
    n_skipped_pos = 0

    for k, v in src_sd.items():
        if k not in tgt_sd:
            continue
        if "position_embedding" in k and v.shape != tgt_sd[k].shape:
            n_skipped_pos += 1
            continue
        if v.shape != tgt_sd[k].shape:
            _logger.warning("[PretrainInit/Gemma] shape mismatch %s: %s vs %s",
                            k, tuple(v.shape), tuple(tgt_sd[k].shape))
            continue
        new_sd[k] = v.clone()
        n_mapped += 1

    missing, _ = vt.load_state_dict(new_sd, strict=False)
    _logger.info("[PretrainInit/Gemma] SigLIP → vision_tower: %d mapped | "
                 "%d pos_emb skipped (size mismatch) | %d missing",
                 n_mapped, n_skipped_pos, len(missing))
    if missing:
        _logger.info("[PretrainInit/Gemma] missing (keep VLM init): %s", missing[:5])


# ---------------------------------------------------------------------------
# Swap vision_tower from another Gemma3 VLM
# ---------------------------------------------------------------------------

def _swap_vision_tower_from_vlm(hf_model, source_vlm_path: str):
    """Extract vision_tower from another Gemma3 VLM checkpoint and load here."""
    _logger.info("[PretrainInit/Gemma] swap_from_vlm: loading source from %s", source_vlm_path)
    from transformers import Gemma3ForConditionalGeneration, AutoModelForImageTextToText
    _local = source_vlm_path.startswith("/")
    ckpt_lower = source_vlm_path.lower()
    try:
        if "gemma3" in ckpt_lower or "gemma-3" in ckpt_lower:
            src = Gemma3ForConditionalGeneration.from_pretrained(
                source_vlm_path, torch_dtype=torch.bfloat16, local_files_only=_local)
        else:
            src = AutoModelForImageTextToText.from_pretrained(
                source_vlm_path, torch_dtype=torch.bfloat16,
                trust_remote_code=True, local_files_only=_local)
    except Exception as e:
        raise RuntimeError(f"Failed to load source VLM: {e}") from e

    if not hasattr(src, "vision_tower"):
        raise RuntimeError(f"Source VLM at {source_vlm_path} has no vision_tower attribute")
    src_sd = {k: v.cpu() for k, v in src.vision_tower.state_dict().items()}
    del src
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    missing, unexpected = hf_model.vision_tower.load_state_dict(src_sd, strict=False)
    _logger.info("[PretrainInit/Gemma] swap_from_vlm: %d loaded | %d missing | %d unexpected",
                 len(src_sd) - len(missing), len(missing), len(unexpected))


# ---------------------------------------------------------------------------
# Reset connector
# ---------------------------------------------------------------------------

def _reset_gemma_projector(hf_model):
    """Re-initialize multi_modal_projector: N(0, 0.02) for 2D, zeros for 1D."""
    proj = getattr(hf_model, "multi_modal_projector", None)
    if proj is None:
        _logger.warning("[PretrainInit/Gemma] No multi_modal_projector found — skipping.")
        return
    n = 0
    for param in proj.parameters():
        if param.dim() >= 2:
            nn.init.normal_(param, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(param)
        n += 1
    _logger.info("[PretrainInit/Gemma] multi_modal_projector reset → N(0, 0.02): %d tensors", n)


# ---------------------------------------------------------------------------
# Gemma3ForCausalLM → language_model
# ---------------------------------------------------------------------------

def _load_gemma3_text_model(hf_model, lm_ckpt: str):
    """Load Gemma3 text model weights into hf_model.language_model.

    Handles both Gemma3ForCausalLM (text-only) and Gemma3ForConditionalGeneration
    (multimodal) source checkpoints — language_model is extracted from the latter.
    Keys are identical between standalone and VLM language_model; no remapping needed.
    Vocab size mismatches (multimodal adds image tokens) are truncated/zero-padded.
    """
    _logger.info("[PretrainInit/Gemma] Loading Gemma3 LLM from %s", lm_ckpt)
    _local = lm_ckpt.startswith("/")
    from transformers import AutoModelForCausalLM, Gemma3ForConditionalGeneration, AutoModelForImageTextToText

    # Try text-only first; fall back to extracting from multimodal checkpoint
    try:
        lm = AutoModelForCausalLM.from_pretrained(
            lm_ckpt, torch_dtype=torch.bfloat16, local_files_only=_local)
        lm_sd = {k: v.cpu() for k, v in lm.state_dict().items()}
        del lm
        _logger.info("[PretrainInit/Gemma] Loaded as Gemma3ForCausalLM.")
    except Exception:
        ckpt_lower = lm_ckpt.lower()
        try:
            if "gemma3" in ckpt_lower or "gemma-3" in ckpt_lower:
                vlm = Gemma3ForConditionalGeneration.from_pretrained(
                    lm_ckpt, torch_dtype=torch.bfloat16, local_files_only=_local)
            else:
                vlm = AutoModelForImageTextToText.from_pretrained(
                    lm_ckpt, torch_dtype=torch.bfloat16,
                    trust_remote_code=True, local_files_only=_local)
            lm_sd = {k: v.cpu() for k, v in vlm.language_model.state_dict().items()}
            del vlm
            _logger.info("[PretrainInit/Gemma] Extracted language_model from multimodal checkpoint.")
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM from {lm_ckpt}: {e}") from e

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tgt = hf_model.language_model
    tgt_sd = tgt.state_dict()
    new_sd = {}
    n_mapped = 0

    for k, v in lm_sd.items():
        if k not in tgt_sd:
            continue
        src_shape = v.shape
        dst_shape = tgt_sd[k].shape
        if src_shape == dst_shape:
            new_sd[k] = v.clone()
            n_mapped += 1
        elif v.dim() == 2 and src_shape[1] == dst_shape[1]:
            # Row-count mismatch: vocab size (multimodal adds image tokens)
            n_rows = dst_shape[0]
            if src_shape[0] >= n_rows:
                new_sd[k] = v[:n_rows].clone()
            else:
                tmp = tgt_sd[k].clone()
                tmp[:src_shape[0]] = v
                new_sd[k] = tmp
                _logger.warning("[PretrainInit/Gemma] %s: src %d rows < dst %d — zero-padded tail",
                                k, src_shape[0], n_rows)
            n_mapped += 1
        else:
            _logger.warning("[PretrainInit/Gemma] shape mismatch %s: %s vs %s",
                            k, tuple(src_shape), tuple(dst_shape))

    missing, _ = tgt.load_state_dict(new_sd, strict=False)
    _logger.info("[PretrainInit/Gemma] Gemma3 LLM → language_model: %d mapped | %d missing",
                 n_mapped, len(missing))
    if missing:
        _logger.warning("[PretrainInit/Gemma] missing keys (keep VLM init): %s", missing[:10])
