"""Unit tests for VLMOnly.generate().

Mirrors test_vlm_sam_generate.py — uses a stubbed backbone to verify that:

1. VLMOnly.generate() returns ``(preds: List[str], None)`` — correct contract.
2. Batch dimension is preserved (len(preds) == batch_size).
3. The ``None`` sentinel is exactly ``None``, not an empty list or tensor.
4. SAM-specific keys in the input are silently dropped (no TypeError).

No GPU required; all stubs run on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pytest
import torch
from transformers import PretrainedConfig

from img_2_svg_pretraining.training.training_core.models.vlm_only import VLMOnly
from img_2_svg_pretraining.training.training_core.models.vlms.base import VLMBase
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, ModelConfig, VLMArguments


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub used by VLMOnly.generate()."""

    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 999

    def convert_tokens_to_ids(self, token: str) -> Optional[int]:
        return None  # treat all special tokens as absent

    def batch_decode(self, sequences: List[torch.LongTensor], skip_special_tokens: bool = True) -> List[str]:
        return [f"decoded_{i}" for i in range(len(sequences))]


@dataclass
class _FakeProcessor:
    tokenizer: _FakeTokenizer = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = _FakeTokenizer()


class _FakeGenerateOutput:
    """Minimal GenerateOutput stub matching transformers convention."""

    def __init__(self, sequences: torch.LongTensor):
        self.sequences = sequences


class _FakeBackbone(VLMBase):
    """Tiny deterministic backbone — generate() returns fake token sequences."""

    BATCH_SIZE_ATTR = "_forced_batch_size"
    NEW_TOKENS = 3

    @property
    def hidden_size(self) -> int:
        return 8

    def resize_token_embeddings(self, vocab_size: int):
        pass

    def forward(self, **kwargs):
        raise NotImplementedError("forward not used in generate tests")

    def generate(self, input_ids: torch.LongTensor, **kwargs) -> _FakeGenerateOutput:
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        # Return prompt + 3 new tokens
        new_tokens = torch.full(
            (batch_size, self.NEW_TOKENS), fill_value=2, dtype=torch.long
        )
        sequences = torch.cat([input_ids, new_tokens], dim=1)
        return _FakeGenerateOutput(sequences)


def _make_data_module() -> DataModule:
    return DataModule(
        model_name="stub",
        model_path="stub",
        processor=_FakeProcessor(),
        seg_token_idx=7,
        ignore_idx=-100,
        Dataloader=None,
        Collator=None,
        layout_classes=[],
        family_name="stub",
        tokenizer_vocab_size=None,  # skip embed resize
    )


def _make_vlm_only_model() -> VLMOnly:
    config = ModelConfig(
        sam_args=None,
        vlm_args=VLMArguments(
            family="stub",
            model_name_or_path="stub",
            model_family_name="stub",
        ),
        vlm_family="stub",
        sam_version="none",
    )
    data_module = _make_data_module()
    model = VLMOnly(config=config, data_module=data_module, backbone=_FakeBackbone())
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.composite
@pytest.mark.cpu
def test_vlm_only_generate_returns_preds_and_none_masks():
    """generate() must return (List[str], None) — the shared contract."""
    model = _make_vlm_only_model()
    input_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    attention_mask = torch.ones(1, 3, dtype=torch.long)

    preds, pred_masks = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert isinstance(preds, list), "preds must be a list"
    assert pred_masks is None, "pred_masks must be exactly None for VLM-only"


@pytest.mark.composite
@pytest.mark.cpu
def test_vlm_only_generate_batch_dimension_preserved():
    """Batch size B → len(preds) == B."""
    model = _make_vlm_only_model()
    B = 4
    input_ids = torch.randint(2, 100, (B, 5))
    attention_mask = torch.ones(B, 5, dtype=torch.long)

    preds, pred_masks = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert len(preds) == B, f"Expected {B} predictions, got {len(preds)}"
    assert pred_masks is None


@pytest.mark.composite
@pytest.mark.cpu
def test_vlm_only_generate_strips_sam_keys_silently():
    """SAM-specific keys in kwargs must not cause a TypeError."""
    model = _make_vlm_only_model()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones(1, 3, dtype=torch.long)

    preds, pred_masks = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # SAM-specific keys that should be silently stripped
        images=torch.zeros(1, 3, 64, 64),
        masks=torch.zeros(1, 1, 64, 64),
        mask_counts=torch.tensor([1]),
        resize_list=torch.tensor([[64, 64]]),
        orig_image_size_list=torch.tensor([[64, 64]]),
        debug_original_image_list=[torch.zeros(64, 64, 3)],
    )

    assert isinstance(preds, list)
    assert pred_masks is None


@pytest.mark.composite
@pytest.mark.cpu
def test_vlm_only_generate_handles_tensor_sequences_output():
    """generate() must work when backbone returns a raw tensor (not GenerateOutput)."""

    class _RawTensorBackbone(_FakeBackbone):
        def generate(self, input_ids: torch.LongTensor, **kwargs) -> torch.LongTensor:
            batch_size = input_ids.shape[0]
            new_tokens = torch.full((batch_size, 2), fill_value=3, dtype=torch.long)
            return torch.cat([input_ids, new_tokens], dim=1)

    config = ModelConfig(
        sam_args=None,
        vlm_args=VLMArguments(family="stub", model_name_or_path="stub", model_family_name="stub"),
        vlm_family="stub",
        sam_version="none",
    )
    model = VLMOnly(config=config, data_module=_make_data_module(), backbone=_RawTensorBackbone())
    model.eval()

    input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    preds, pred_masks = model.generate(input_ids=input_ids)

    assert isinstance(preds, list)
    assert len(preds) == 1
    assert pred_masks is None
