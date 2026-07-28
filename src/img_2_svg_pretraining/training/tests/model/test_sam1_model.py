import pytest
import torch

from img_2_svg_pretraining.training.training_core.models.sam.sam1.sam1_model import Sam1Model
from img_2_svg_pretraining.training.training_core.registry.utils import SamModelArguments
from tests.support.runtime import runtime_enabled


@pytest.mark.model
@pytest.mark.cpu
def test_sam1_compute_mask_losses_handles_per_sample_mask_tensors():
    model = object.__new__(Sam1Model)
    pred_masks = [
        torch.zeros(1, 8, 8, dtype=torch.float32),
        torch.zeros(2, 6, 6, dtype=torch.float32),
    ]
    gt_masks = [
        torch.zeros(1, 8, 8, dtype=torch.float32),
        torch.zeros(2, 6, 6, dtype=torch.float32),
    ]

    bce, dice = Sam1Model.compute_mask_losses(model, pred_masks, gt_masks)

    assert torch.isfinite(bce)
    assert torch.isfinite(dice)


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.external
@pytest.mark.slow
def test_sam1_model_constructs_with_checkpoint(sam1_checkpoint):
    if not runtime_enabled("gpu"):
        pytest.skip("GPU runtime tests disabled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not sam1_checkpoint:
        pytest.skip("SAM1 checkpoint unavailable")

    model = Sam1Model(
        SamModelArguments(
            tune_image_encoder=False,
            tune_prompt_encoder=False,
            tune_mask_decoder=True,
            checkpoint=sam1_checkpoint,
            version="sam1",
        )
    ).to(torch.device("cuda:0"))

    assert model.prompt_embed_dim > 0
    assert next(model.parameters()).is_cuda
