"""
Smoke test: Qwen2.5-VL visual encoder swapped into Gemma3 decoder (VLMSam composite).

Steps:
  1. Build Gemma3 VLMSam.
  2. Extract Qwen2.5-VL's visual encoder to /tmp/qwen_ve_extract/.
  3. Swap the extracted encoder into Gemma3's backbone.
  4. Run 3 training steps using a real Gemma data module batch (correct image tokens).
"""
from __future__ import annotations

import sys
import tempfile
import time

import torch
from torch.optim import AdamW

sys.path.insert(0, "/code")

QWEN_CKPT = (
    "/fsxvision_new/anirudh.srinivasan/hf_cache/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
GEMMA_CKPT = (
    "/fsxvision_new/anirudh.srinivasan/hf_cache/hub/"
    "models--google--gemma-3-4b-it/snapshots/"
    "093f9f388b31de276ce2de164bdc2081324b9767"
)
SAM_CKPT = "/fsxvision_new/srihari.bandarupalli/DocGrounding/checkpoints/sam_vit_h_4b8939.pth"


def main():
    device = torch.device("cuda:0")
    print(f"[cross-family] device: {device}", flush=True)

    t0 = time.time()

    # --- step 1: build Gemma3 VLMSam ---
    print("[cross-family] building Gemma3 VLMSam ...", flush=True)
    from img_2_svg_pretraining.training.training_core.builders.build_vlm_sam import build_vlm_sam
    model, _ = build_vlm_sam(
        vlm_family="gemmavlm",
        vlm_checkpoint=GEMMA_CKPT,
        model_family_name="gemma3",
        sam_version="sam1",
        sam_checkpoint=SAM_CKPT,
        attn_implementation="eager",
        bf16=True,
    )
    print(f"[cross-family] Gemma VLMSam built. hidden_size={model.backbone.hidden_size}", flush=True)

    # --- step 2: extract Qwen visual encoder ---
    ve_dir = tempfile.mkdtemp(prefix="qwen_ve_extract_")
    print(f"[cross-family] extracting Qwen2.5-VL visual encoder to {ve_dir} ...", flush=True)
    from img_2_svg_pretraining.training.training_core.builders.extract_vision_encoder import extract_vision_encoder
    enc = extract_vision_encoder(
        vlm_family="qwenvlm",
        vlm_checkpoint=QWEN_CKPT,
        model_family_name="qwen2.5vl",
        encoder_name="qwen_visual",
        output_dir=ve_dir,
    )
    print(f"[cross-family] extracted encoder embed_dim={enc.embed_dim}", flush=True)

    # --- step 3: swap into Gemma backbone ---
    print("[cross-family] swapping encoder into Gemma backbone ...", flush=True)
    from img_2_svg_pretraining.training.training_core.builders.swap_vision_encoder import swap_vision_encoder
    model.backbone, adapter = swap_vision_encoder(model.backbone, enc, "gemmavlm")
    print(f"[cross-family] swap done. adapter={adapter}", flush=True)

    model.to(device)

    # --- step 4: build a real Gemma batch (uses Gemma processor → correct image tokens) ---
    from tests.support.builders import build_vlm_sam_model, load_vlm_data_module, make_synthetic_image
    from tests.support.matrix import GEMMA3_SPEC

    data_module = load_vlm_data_module(
        spec=GEMMA3_SPEC,
        model_ref=GEMMA_CKPT,
        image=make_synthetic_image(),
        sam_image_size=1024,
    )
    batch_raw = data_module.Collator([data_module.Dataloader[0]])
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_raw.items()}

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=1e-6)

    model.train()
    print("[cross-family] running 3 training steps ...", flush=True)
    for step in range(1, 4):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs["loss"]
        assert torch.isfinite(loss), f"step {step}: non-finite loss {loss.item()}"
        loss.backward()
        optimizer.step()
        print(
            f"[cross-family] step {step}/3 | loss={loss.item():.4f} "
            f"ce={outputs['ce_loss'].item():.4f} "
            f"mask={outputs['mask_loss'].item():.4f}",
            flush=True,
        )

    elapsed = time.time() - t0
    print(f"[cross-family] PASSED in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
