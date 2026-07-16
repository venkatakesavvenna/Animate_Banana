"""SAM3 point-prompted mask fine-tuning on human-accepted annotations.

Pipeline: prepare_masks.py (morphological fill + manifest) ->
qa_contact_sheet.py (random-20 visual check, human gate) ->
dataset.py (train/val split, torch Dataset) -> train.py (decoder-only
fine-tune, checkpoint save). See README.md for run order and rationale.
"""
from __future__ import annotations
