import importlib

import torch
import torch.nn as nn

from img_2_svg_pretraining.training.training_core.inference.utils import prepare_generate_batch
from img_2_svg_pretraining.training.training_core.models.sam.base import SAMModelBase
from img_2_svg_pretraining.training.training_core.models.vlm_sam import VLMSam
from img_2_svg_pretraining.training.training_core.models.vlms.base import VLMBase
from img_2_svg_pretraining.training.training_core.vision_encoders.base import VisionEncoderBase
from img_2_svg_pretraining.training.training_core.registry.registry import (
    DataModuleRegistry,
    SAMDataModuleRegistry,
    SAMModelRegistry,
    VisionEncoderRegistry,
    VLMModelRegistry,
)
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, ModelConfig, SamModelArguments, VLMArguments


class TinyBackbone(VLMBase):
    def __init__(self):
        super().__init__()
        self.hidden = 8
        self.resized_to = None

    @property
    def hidden_size(self) -> int:
        return self.hidden

    def resize_token_embeddings(self, vocab_size: int):
        self.resized_to = vocab_size

    def forward(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        hidden = torch.ones(batch, seq_len, self.hidden, dtype=torch.bfloat16)

        class Output:
            loss = torch.tensor(1.0, dtype=torch.float32)
            hidden_states = [hidden]
            logits = torch.zeros(batch, seq_len, 4, dtype=torch.float32)

        return Output()

    def generate(self, **kwargs):
        raise NotImplementedError("TinyBackbone.generate is not used in this contract test")


class TinySam(SAMModelBase):
    @property
    def prompt_embed_dim(self) -> int:
        return 4

    def forward(self, images, masks, vlm_seg_hidden_states, resize_list, label_list, mask_counts=None):
        pred_masks = [torch.zeros(len(vlm_seg_hidden_states[0]), 8, 8)]
        return pred_masks, torch.tensor(0.5), torch.tensor(0.25)


def _stub_data_module(tokenizer_vocab_size: int | None = None) -> DataModule:
    data_module = DataModule(
        model_name="qwen2.5vl",
        model_path="stub",
        processor=type("Processor", (), {"tokenizer": object()})(),
        seg_token_idx=99,
        ignore_idx=-100,
        Dataloader=None,
        Collator=None,
        layout_classes=["title"],
        family_name="qwenvlm",
        tokenizer_vocab_size=tokenizer_vocab_size,
    )
    return data_module


def _stub_config() -> ModelConfig:
    return ModelConfig(
        sam_args=SamModelArguments(False, False, True, "stub", "sam1"),
        vlm_args=VLMArguments(family="qwenvlm", model_name_or_path="stub", model_family_name="qwen2.5vl"),
        vlm_family="qwenvlm",
        sam_version="sam1",
    )


@torch.no_grad()
def test_registry_keys_are_available():
    from img_2_svg_pretraining.training.training_core.data_modules.vlms.molmo import molmo_data  # noqa: F401
    from img_2_svg_pretraining.training.training_core.models.vlms.molmo import molmo_model  # noqa: F401

    assert "qwenvl" in DataModuleRegistry.available()
    assert "qwenvlm" in DataModuleRegistry.available()
    assert "gemmavlm" in DataModuleRegistry.available()
    assert "molmovlm" in DataModuleRegistry.available()
    assert "sam1" in SAMDataModuleRegistry.available()
    assert "qwenvlm" in VLMModelRegistry.available()
    assert "gemmavlm" in VLMModelRegistry.available()
    assert "molmovlm" in VLMModelRegistry.available()
    assert "sam1" in SAMModelRegistry.available()


@torch.no_grad()
def test_builder_modules_importable():
    from img_2_svg_pretraining.training.training_core.builders import extract_vision_encoder, swap_vision_encoder, build_vlm_sam  # noqa: F401
    assert hasattr(extract_vision_encoder, "extract_vision_encoder")
    assert hasattr(swap_vision_encoder, "swap_vision_encoder")
    assert hasattr(build_vlm_sam, "build_vlm_sam")


@torch.no_grad()
def test_swap_no_dim_change_no_adapter():
    """No adapter when new encoder dim matches the projector's input dim."""
    from img_2_svg_pretraining.training.training_core.builders.swap_vision_encoder import swap_vision_encoder

    class StubEncoder(VisionEncoderBase):
        @property
        def embed_dim(self) -> int:
            return 8

        @property
        def preprocessor_config(self) -> dict:
            return {"image_mean": [0.5] * 3, "image_std": [0.5] * 3, "image_size": 224}

        def forward(self, x):
            return x

    backbone = TinyBackbone()
    # TinyBackbone.hidden = 8; inject a mock qwen attribute so the projector helper works.
    import types
    backbone.qwen = types.SimpleNamespace(visual=types.SimpleNamespace(merger=nn.Linear(8, 8)))

    _, adapter = swap_vision_encoder(backbone, StubEncoder(), "qwenvlm")
    assert adapter is None


@torch.no_grad()
def test_swap_dim_mismatch_creates_adapter():
    """An nn.Linear adapter is inserted when encoder dim != projector input dim."""
    import types
    from img_2_svg_pretraining.training.training_core.builders.swap_vision_encoder import swap_vision_encoder

    class StubEncoder1152(VisionEncoderBase):
        @property
        def embed_dim(self) -> int:
            return 1152

        @property
        def preprocessor_config(self) -> dict:
            return {"image_mean": [0.5] * 3, "image_std": [0.5] * 3, "image_size": 224}

        def forward(self, x):
            return x

    backbone = TinyBackbone()
    backbone.qwen = types.SimpleNamespace(visual=types.SimpleNamespace(merger=nn.Linear(8, 16)))

    _, adapter = swap_vision_encoder(backbone, StubEncoder1152(), "qwenvlm")
    assert isinstance(adapter, nn.Linear)
    assert adapter.in_features == 1152
    assert adapter.out_features == 8  # original projector in_features


@torch.no_grad()
def test_vision_encoder_registry_available_keys():
    from img_2_svg_pretraining.training.training_core.vision_encoders.clip import clip_encoder  # noqa: F401
    from img_2_svg_pretraining.training.training_core.vision_encoders.extracted import extracted_encoder  # noqa: F401
    from img_2_svg_pretraining.training.training_core.vision_encoders.siglip import siglip_encoder  # noqa: F401
    from img_2_svg_pretraining.training.training_core.vision_encoders.metaclip import metaclip_encoder  # noqa: F401
    from img_2_svg_pretraining.training.training_core.vision_encoders.openvision import openvision_encoder  # noqa: F401

    expected = ["clip", "extracted", "siglip", "siglip2", "metaclip", "metaclip2", "openvision", "openvision2"]
    available = VisionEncoderRegistry.available()
    for key in expected:
        assert key in available, f"'{key}' not found in VisionEncoderRegistry. Available: {available}"


@torch.no_grad()
def test_vlm_sam_resizes_backbone_embeddings_from_datamodule_vocab():
    backbone = TinyBackbone()
    VLMSam(
        config=_stub_config(),
        data_module=_stub_data_module(tokenizer_vocab_size=123),
        backbone=backbone,
        sam_head=TinySam(),
    )
    assert backbone.resized_to == 123


@torch.no_grad()
def test_vlm_sam_projector_matches_backbone_and_sam_dimensions():
    model = VLMSam(
        config=_stub_config(),
        data_module=_stub_data_module(),
        backbone=TinyBackbone(),
        sam_head=TinySam(),
    )
    projector = model.text_hidden_fcs_layout[0]
    assert projector[0].in_features == 8
    assert projector[0].out_features == 8
    assert projector[2].in_features == 8
    assert projector[2].out_features == 4


def test_vlm_sam_preserves_eval_return_contract():
    model = VLMSam(
        config=_stub_config(),
        data_module=_stub_data_module(),
        backbone=TinyBackbone(),
        sam_head=TinySam(),
    )
    model.eval()

    outputs = model(
        input_ids=torch.tensor([[1, 99, 2]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        labels=torch.tensor([[1, 2, 3]]),
        pixel_values=torch.zeros(1, 3, 4, 4),
        image_grid_thw=torch.ones(1, 3, dtype=torch.long),
        position_ids=torch.ones(3, 1, 3, dtype=torch.long),
        images=torch.zeros(1, 3, 8, 8),
        masks=torch.zeros(1, 1, 8, 8),
        mask_counts=torch.tensor([1]),
        resize_list=torch.tensor([[8, 8]]),
        orig_image_size_list=torch.tensor([[8, 8]]),
        debug_original_image_list=[torch.zeros(8, 8, 3)],
    )

    assert list(outputs.keys()) == ["loss", "ce_loss", "mask_loss", "logits", "pred_masks", "gt_masks", "debug_original_images"]
    assert outputs["gt_masks"][0].shape == (1, 8, 8)


@torch.no_grad()
def test_inference_modules_import_from_checked_in_package():
    fastapi_module = importlib.import_module("img_2_svg_pretraining.training.training_core.inference.fastapi")
    factory_module = importlib.import_module("img_2_svg_pretraining.training.training_core.inference.factory")
    val_generate_module = importlib.import_module("img_2_svg_pretraining.training.training_core.inference.val_set_with_generate")

    assert fastapi_module.__name__ == "img_2_svg_pretraining.training.training_core.inference.fastapi"
    assert factory_module.__name__ == "img_2_svg_pretraining.training.training_core.inference.factory"
    assert val_generate_module.__name__ == "img_2_svg_pretraining.training.training_core.inference.val_set_with_generate"


@torch.no_grad()
def test_prepare_generate_batch_truncates_qwen_style_inputs_from_labels():
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 13, 14]], dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 4, 4),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        "position_ids": torch.arange(15, dtype=torch.long).view(1, 3, 5),
        "images": torch.zeros(1, 3, 8, 8),
        "resize_list": torch.tensor([[8, 8]], dtype=torch.long),
        "orig_image_size_list": torch.tensor([[8, 8]], dtype=torch.long),
    }

    prepared = prepare_generate_batch(batch)

    assert set(prepared.keys()) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "position_ids",
        "images",
        "resize_list",
        "orig_image_size_list",
    }
    assert prepared["input_ids"].shape == (1, 3)
    assert prepared["position_ids"].shape == (1, 3, 3)


@torch.no_grad()
def test_prepare_generate_batch_truncates_gemma_style_inputs_from_labels():
    batch = {
        "input_ids": torch.tensor([[20, 21, 22, 23]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 22, 23]], dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 4, 4),
        "token_type_ids": torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
        "images": torch.zeros(1, 3, 8, 8),
        "resize_list": torch.tensor([[8, 8]], dtype=torch.long),
        "orig_image_size_list": torch.tensor([[8, 8]], dtype=torch.long),
    }

    prepared = prepare_generate_batch(batch)

    assert set(prepared.keys()) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "token_type_ids",
        "images",
        "resize_list",
        "orig_image_size_list",
    }
    assert prepared["input_ids"].shape == (1, 2)
    assert prepared["token_type_ids"].shape == (1, 2)
