"""Contract tests for the inference layer.

Verifies:
1. InferenceResult dataclass structure and helper methods.
2. read_sam_version_from_checkpoint correctly reads from config.json.
3. save_outputs.py handles has_masks=False (VLM-only) and has_masks=True (VLM+SAM).
4. run_inference in fastapi.py handles pred_masks=None (VLM-only) without crash.
5. InferenceRunner and InferenceResult are importable from fastapi.py (re-export).
"""
from __future__ import annotations

import importlib
import json
import os
import tempfile
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from img_2_svg_pretraining.training.training_core.inference.contract import InferenceResult
from img_2_svg_pretraining.training.training_core.inference.factory import read_sam_version_from_checkpoint


# ---------------------------------------------------------------------------
# 1. InferenceResult
# ---------------------------------------------------------------------------


class TestInferenceResult:
    def test_vlm_only_result_has_no_masks(self):
        result = InferenceResult(text="hello", detections=None, has_masks=False)
        assert result.has_masks is False
        assert result.pred_masks_np() == []
        assert result.pred_classes() == []
        assert result.pred_bboxes() == []

    def test_vlm_only_result_with_text_classes(self):
        detections = [
            {"cls_name": "title", "bbox": None},
            {"cls_name": "paragraph", "bbox": [10, 20, 100, 80]},
        ]
        result = InferenceResult(text="<layout>title</layout>", detections=detections, has_masks=False)
        assert result.pred_classes() == ["title", "paragraph"]
        assert result.pred_bboxes() == [None, [10, 20, 100, 80]]
        assert result.pred_masks_np() == []  # no masks even though detections present

    def test_vlm_sam_result_has_masks(self):
        mask = np.ones((64, 64), dtype=np.uint8)
        detections = [
            {"cls_name": "figure", "bbox": [0, 0, 64, 64], "mask_np": mask},
        ]
        result = InferenceResult(text="<layout>figure</layout> [SEG]", detections=detections, has_masks=True)
        assert result.has_masks is True
        assert len(result.pred_masks_np()) == 1
        assert result.pred_masks_np()[0].shape == (64, 64)

    def test_default_fields(self):
        result = InferenceResult(text="")
        assert result.detections is None
        assert result.has_masks is False


# ---------------------------------------------------------------------------
# 2. read_sam_version_from_checkpoint
# ---------------------------------------------------------------------------


class TestReadSamVersionFromCheckpoint:
    def test_reads_sam1_from_config(self, tmp_path):
        cfg = {"model_type": "vlm_sam", "sam_version": "sam1", "vlm_family": "qwenvlm"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        assert read_sam_version_from_checkpoint(str(tmp_path)) == "sam1"

    def test_reads_none_from_config(self, tmp_path):
        cfg = {"model_type": "vlm_sam", "sam_version": "none"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        assert read_sam_version_from_checkpoint(str(tmp_path)) == "none"

    def test_raises_file_not_found_when_config_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config.json"):
            read_sam_version_from_checkpoint(str(tmp_path))

    def test_raises_key_error_when_sam_version_absent(self, tmp_path):
        cfg = {"model_type": "vlm_sam"}  # no sam_version
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        with pytest.raises(KeyError, match="sam_version"):
            read_sam_version_from_checkpoint(str(tmp_path))


# ---------------------------------------------------------------------------
# 3. save_outputs — mask-optional guards
# ---------------------------------------------------------------------------


class TestSaveOutputs:
    """Verify that Save_Outputs handles both has_masks=True and has_masks=False."""

    def _make_saver(self):
        from img_2_svg_pretraining.training.training_core.inference.save_outputs import Save_Outputs
        return Save_Outputs(extract_layout_fn=lambda s: ["title"])

    def test_vlm_only_writes_text_json(self, tmp_path):
        saver = self._make_saver()
        saver.evaluate_single_image(
            pm=None,
            gm=None,
            gt_labels=["title"],
            pred_labels=["title"],
            orig=None,
            img_idx=0,
            threshold=0.0,
            debug_save_dir=str(tmp_path),
            preds_str="<layout>title</layout>",
            labels_str="<layout>title</layout>",
            has_masks=False,
        )
        out = tmp_path / "0" / "detections.json"
        assert out.exists(), "detections.json must be written for VLM-only"
        payload = json.loads(out.read_text())
        assert "pred_str" in payload
        assert "pred_classes" in payload
        assert not (tmp_path / "0" / "pred_masks.npy").exists(), "no masks should be saved"

    def test_vlm_sam_writes_masks_json(self, tmp_path):
        saver = self._make_saver()
        pm = torch.ones(1, 8, 8)  # non-zero mask
        gm = torch.ones(1, 8, 8)
        saver.evaluate_single_image(
            pm=pm,
            gm=gm,
            gt_labels=["title"],
            pred_labels=["title"],
            orig=None,
            img_idx=0,
            threshold=0.0,
            debug_save_dir=str(tmp_path),
            preds_str="<layout>title</layout> [SEG]",
            labels_str="<layout>title</layout> [SEG]",
            has_masks=True,
        )
        assert (tmp_path / "0" / "pred_masks.npy").exists()
        assert (tmp_path / "0" / "gt_masks.npy").exists()
        assert (tmp_path / "0" / "detections.json").exists()

    def test_vlm_only_batch_no_crash(self, tmp_path):
        saver = self._make_saver()
        saver.evaluate_batch(
            pred_masks=None,
            gt_masks=None,
            labels=["gt text"],
            preds=["pred text"],
            original_images=None,
            debug_save_dir=str(tmp_path),
            has_masks=False,
        )
        # Should not raise; text json is written
        assert (tmp_path / "0" / "detections.json").exists()


# ---------------------------------------------------------------------------
# 4. run_inference handles pred_masks=None
# ---------------------------------------------------------------------------


class _MockModel:
    """Model stub that mimics VLMOnly.generate() returning (preds, None)."""
    def __init__(self, return_masks: bool):
        self._return_masks = return_masks

    def generate(self, **kwargs) -> Tuple:
        if self._return_masks:
            return ["pred text"], [torch.zeros(1, 8, 8)]
        return ["pred text"], None


class _MockDataArgs:
    extract_from_labels_fn = lambda self, x: ["title"]  # noqa: E731


class _MockTokenizer:
    pass


class _MockProcessor:
    tokenizer = _MockTokenizer()


class _MockDM:
    processor = _MockProcessor()


def test_run_inference_vlm_only_no_crash():
    """run_inference must not crash when pred_masks is None."""
    from img_2_svg_pretraining.training.training_core.inference import fastapi

    model = _MockModel(return_masks=False)
    data_args = _MockDataArgs()
    data_module = _MockDM()

    batch = {"input_ids": torch.tensor([[1, 2, 3]])}

    # Patch helpers to avoid full data pipeline
    with (
        patch.object(fastapi, "move_to_device", side_effect=lambda b, d: b),
        patch.object(fastapi, "prepare_generate_batch", side_effect=lambda b, tokenizer: b),
    ):
        preds, detections = fastapi.run_inference(model, batch, data_module, data_args, "cpu")

    assert preds == ["pred text"]
    assert len(detections) == 1
    for det in detections[0]:
        assert "mask_np" not in det  # VLM-only has no masks
        assert "cls_name" in det


def test_run_inference_vlm_sam_includes_masks():
    """run_inference must populate mask_np when pred_masks are returned."""
    from img_2_svg_pretraining.training.training_core.inference import fastapi
    from img_2_svg_pretraining.training.training_core.validation.eval_utils import to_numpy_mask

    model = _MockModel(return_masks=True)
    data_args = _MockDataArgs()
    data_module = _MockDM()
    batch = {"input_ids": torch.tensor([[1, 2, 3]])}

    with (
        patch.object(fastapi, "move_to_device", side_effect=lambda b, d: b),
        patch.object(fastapi, "prepare_generate_batch", side_effect=lambda b, tokenizer: b),
    ):
        preds, detections = fastapi.run_inference(model, batch, data_module, data_args, "cpu")

    assert preds == ["pred text"]
    assert len(detections) == 1
    for det in detections[0]:
        assert "mask_np" in det


# ---------------------------------------------------------------------------
# 5. Re-exports from fastapi.py
# ---------------------------------------------------------------------------


def test_fastapi_re_exports_inference_runner():
    import img_2_svg_pretraining.training.training_core.inference.fastapi as m
    assert hasattr(m, "InferenceRunner"), "InferenceRunner must be re-exported"
    assert hasattr(m, "InferenceResult"), "InferenceResult must be re-exported"
