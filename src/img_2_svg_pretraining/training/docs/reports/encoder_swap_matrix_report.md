# Encoder × Decoder Swap Matrix — Smoke Test Report

**Run ID:** `20260625T131905Z` (base) + `20260625T142345Z` (openvision)
**Date:** 2026-06-25
**W&B Project:** `img-2-svg-pretraining-encoder-swap-matrix`
**Total jobs:** 80 (4 decoders × 10 encoders × 2 modes)
**Result:** ✅ **80/80 passed**

---

## Summary

| Decoder | Passed | Failed |
|---|---|---|
| molmo7b-d | 20/20 | 0 |
| molmo7b-o | 20/20 | 0 |
| qwen25 | 20/20 | 0 |
| gemma3 | 20/20 | 0 |
| **Total** | **80/80** | **0** |

---

## molmo7b-d

| Encoder | sam1 loss | vlm_only loss |
|---|---|---|
| native | 15.225 | 14.646 |
| clip | 4.615 | 4.394 |
| siglip | 4.529 | 4.491 |
| siglip2 | 4.486 | 4.754 |
| metaclip | 4.733 | 4.502 |
| metaclip2 | 4.289 | 4.459 |
| openvision | **4.195** | **4.395** |
| extracted-molmo7bo | 4.279 | 4.065 |
| extracted-qwen25 | 4.594 | 4.417 |
| extracted-gemma3 | 4.305 | 4.199 |

---

## molmo7b-o

| Encoder | sam1 loss | vlm_only loss |
|---|---|---|
| native | 16.327 | 15.711 |
| clip | 4.705 | 4.509 |
| siglip | 13.612 | 12.916 |
| siglip2 | 13.799 | 12.760 |
| metaclip | 14.946 | 14.188 |
| metaclip2 | 14.102 | 12.868 |
| openvision | **13.462** | **12.624** |
| extracted-molmo7bd | 14.765 | 13.843 |
| extracted-qwen25 | 14.789 | 14.213 |
| extracted-gemma3 | 17.133 | 15.540 |

---

## qwen25

| Encoder | sam1 loss | vlm_only loss |
|---|---|---|
| native | 3.498 | 2.980 |
| clip | 1.475 | 0.983 |
| siglip | 3.711 | 3.198 |
| siglip2 | 3.738 | 3.150 |
| metaclip | 3.827 | 3.155 |
| metaclip2 | 4.344 | 3.155 |
| openvision | **3.629** | **3.145** |
| extracted-molmo7bo | 3.657 | 3.057 |
| extracted-molmo7bd | 4.290 | 3.221 |
| extracted-gemma3 | 3.978 | 3.083 |

---

## gemma3

| Encoder | sam1 loss | vlm_only loss |
|---|---|---|
| native | 3.482 | 2.787 |
| clip | 1.392 | 1.211 |
| siglip | 3.865 | 3.059 |
| siglip2 | 4.119 | 3.499 |
| metaclip | 3.988 | 3.825 |
| metaclip2 | 4.027 | 3.575 |
| openvision | **4.110** | **3.469** |
| extracted-molmo7bo | 3.636 | 2.672 |
| extracted-molmo7bd | 3.390 | 3.079 |
| extracted-qwen25 | 4.428 | 3.573 |

---

## Fixes Applied

### 1. Cross-Molmo encoder swap broadcast bug (`775b2b7`)
`image_masks` in `_molmo_patchify_images` was shape `[B,1]` (2D per-crop). Molmo-7B-O's `pad_and_partial_pad` path expects `[B,T,N_patches]` (3D per-patch). The 2D mask broadcasted `image_features` from `[B,1,N,D]` → `[B,B,N,D]`, causing a downstream matmul shape mismatch. Fixed to `torch.ones((batch, 1, grid_h * grid_w))`.

**Affected jobs (previously failing, now passing):**
- `molmo7b-d` + `extracted-molmo7bo`: sam1=4.279, vlm_only=4.065
- `molmo7b-o` + `extracted-molmo7bd`: sam1=14.765, vlm_only=13.843

### 2. OpenVision `hf-hub:` argument order (`a59ae02`)
`open_clip.create_model_from_pretrained` was called as `(arch, "hf-hub:...")`. The `hf-hub:` string must be passed as `model_name` (first arg), not `pretrained`. Dropped the redundant `arch`/`_infer_arch` logic entirely.

**Affected jobs (previously failing, now passing):** all 8 openvision jobs across all decoders.

---

## Smoke Test Criteria

Each passing job produces:
- Finite, non-NaN training loss after 3 steps
- Validation artifact PNG saved under `outputs/.../validation_steps/.../3/0/`
- W&B run logged to project `img-2-svg-pretraining-encoder-swap-matrix`

All 80 jobs satisfy all three criteria.
