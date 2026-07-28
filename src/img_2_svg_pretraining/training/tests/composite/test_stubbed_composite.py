import torch
import pytest

from img_2_svg_pretraining.training.training_core.models.sam.base import SAMModelBase
from img_2_svg_pretraining.training.training_core.models.vlm_sam import VLMSam
from img_2_svg_pretraining.training.training_core.models.vlms.base import VLMBase
from img_2_svg_pretraining.training.training_core.registry.utils import DataModule, ModelConfig, SamModelArguments, VLMArguments


class TinyBackbone(VLMBase):
    @property
    def hidden_size(self) -> int:
        return 6

    def resize_token_embeddings(self, vocab_size: int):
        self.vocab_size = vocab_size

    def forward(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        hidden = torch.ones(batch, seq_len, self.hidden_size, dtype=torch.bfloat16)

        class Output:
            loss = torch.tensor(1.0, dtype=torch.float32)
            hidden_states = [hidden]
            logits = torch.zeros(batch, seq_len, 4, dtype=torch.float32)

        return Output()

    def generate(self, **kwargs):
        raise NotImplementedError("TinyBackbone.generate is not used in this stubbed forward test")


class TinySam(SAMModelBase):
    @property
    def prompt_embed_dim(self) -> int:
        return 3

    def forward(self, images, masks, vlm_seg_hidden_states, resize_list, label_list, mask_counts=None):
        pred_masks = [torch.zeros(1, 8, 8)]
        return pred_masks, torch.tensor(0.5), torch.tensor(0.25)


@pytest.mark.composite
@pytest.mark.cpu
def test_vlm_sam_forward_runs_in_cpu_safe_stubbed_path():
    data_module = DataModule(
        model_name="stub",
        model_path="stub",
        processor=type("Processor", (), {"tokenizer": object()})(),
        seg_token_idx=7,
        ignore_idx=-100,
        Dataloader=None,
        Collator=None,
        layout_classes=["title"],
        family_name="stub",
        tokenizer_vocab_size=32,
    )
    model = VLMSam(
        config=ModelConfig(
            sam_args=SamModelArguments(False, False, True, "stub", "sam1"),
            vlm_args=VLMArguments(family="stub", model_name_or_path="stub", model_family_name="stub"),
            vlm_family="stub",
            sam_version="sam1",
        ),
        data_module=data_module,
        backbone=TinyBackbone(),
        sam_head=TinySam(),
    )
    outputs = model(
        input_ids=torch.tensor([[1, 7, 2]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        labels=torch.tensor([[1, 2, 3]]),
        images=torch.zeros(1, 3, 8, 8),
        masks=torch.zeros(1, 1, 8, 8),
        mask_counts=torch.tensor([1]),
        resize_list=torch.tensor([[8, 8]]),
        orig_image_size_list=torch.tensor([[8, 8]]),
    )

    assert float(outputs["loss"]) > 0
    assert float(outputs["mask_loss"]) > 0
