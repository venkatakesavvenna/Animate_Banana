# SAM3 mask fine-tuning

Fine-tunes `facebook/sam3` (`Sam3TrackerModel`) on human-accepted point ->
mask instances from the annotation tool (`annotation_tool/`), to adapt SAM3
to this project's diagram-node visual style.

## Data caveat

All source masks have `mask_source == "sam3_auto"` — they are SAM3's own
predictions that a human reviewer accepted as correct, not independently
hand-drawn ground truth. This is still a legitimate fine-tune (human
acceptance filtered out bad predictions; the model learns this domain's
specific node/shape conventions), but it is a domain-adaptation/distillation
signal, not fully independent supervision. Keep this in mind when
interpreting results.

## Run order

```bash
# 1. Morphological fill (closing, then conditional opening) over every
#    accepted+masked instance in data/annotations/.
python -m img_2_svg_pretraining.data_engine.sam_finetuning.prepare_masks

# 2. Random-20 visual QA contact sheet -- LOOK AT THIS before training.
python -m img_2_svg_pretraining.data_engine.sam_finetuning.qa_contact_sheet --seed 42

# 3. Train (single GPU; dataset/trainable-param count too small to justify DDP).
#    Logs to Weights & Biases by default (WANDB_API_KEY must be in the env --
#    source the repo-root .env first); pass --no-wandb to skip.
set -a && source .env && set +a
CUDA_VISIBLE_DEVICES=5 python -m img_2_svg_pretraining.data_engine.sam_finetuning.train \
    --run-name <name> --epochs 30 --batch-size 4
```

Final checkpoint: `checkpoints/<name>/final/` (also `checkpoints/<name>/best/`,
saved whenever val loss improves). Both are full HF-style `Sam3TrackerModel`
checkpoints (frozen vision-encoder weights included), reloadable exactly
like existing inference code:

```python
from transformers import Sam3TrackerModel
model = Sam3TrackerModel.from_pretrained("checkpoints/<name>/final").to("cuda").eval()
```

Drop-in compatible with `data_engine/segmentation/sam3_tracker.py::Sam3Runner`
and `annotation_tool/sam3_backend.py::Sam3InteractiveBackend` — just point
`hf_repo`/`spec.hf_repo` at the checkpoint directory instead of
`facebook/sam3`.

## Fine-tuning strategy

- **Frozen:** `vision_encoder` (454M params).
- **Trained:** `prompt_encoder` + `mask_decoder` + `shared_image_embedding`
  (~4.2M params, confirmed via direct inspection). Full fine-tune, no
  LoRA/peft — the trainable module is already small enough that LoRA (which
  exists to cheaply adapt *large* modules) adds no benefit, and `peft` isn't
  a repo dependency. Rationale: with only ~1,549 masks across 335 images,
  training the 454M-param encoder risks catastrophic forgetting/overfitting
  for no clear upside.
- **Loss:** Dice + BCE between predicted mask logits (native 288x288
  decoder-output resolution) and the filled GT mask, resized to match.
  Optional `--iou-loss-weight` (default 0, off) adds an MSE term between
  predicted and true IoU, per the original SAM paper — left off by default
  to keep the first runs simple to debug.
- **Split:** image-level train/val (default 30 val images / ~305 train),
  written to `splits.json` for reproducibility. A 30-image val set is
  directional only, not a rigorous benchmark, given the small data scale.
- **GPU:** single GPU (`CUDA_VISIBLE_DEVICES=5`) — confirmed sufficient at
  this data/model scale; no DDP wiring built.
- **Logging:** Weights & Biases (project `sam3-mask-finetuning`), plus a
  local `checkpoints/<name>/log.csv` always written as a backup. The run URL
  is printed at startup and saved to `checkpoints/<name>/wandb_url.txt`.

## Files

- `prepare_masks.py` — closing (+ conditional opening) over accepted masks; writes `masks_processed/` + `manifest.jsonl`.
- `qa_contact_sheet.py` — random-N triptych QA sheet (raw crop / original mask / filled mask).
- `dataset.py` — `Sam3MaskDataset`, image-level train/val split (`splits.json`).
- `train.py` — training loop: freeze vision encoder, Dice+BCE loss, checkpoint save.
- `masks_processed/`, `checkpoints/`, `qa/`, `splits.json` — generated artifacts, gitignored.
