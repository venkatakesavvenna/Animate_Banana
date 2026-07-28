from typing import List, Tuple

import torch
import torch.nn.functional as F

from img_2_svg_pretraining.training.training_core.data_modules.sam.sam_data import unpack_mask_batch
from img_2_svg_pretraining.training.training_core.models.sam.base import SAMModelBase
from img_2_svg_pretraining.training.training_core.models.sam.build_sam import _build_sam
from img_2_svg_pretraining.training.training_core.models.sam.modeling import Sam
from img_2_svg_pretraining.training.training_core.registry.registry import SAMModelRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import SamModelArguments


class Sam1Model(SAMModelBase):
    def __init__(self, model_args: SamModelArguments):
        super().__init__()
        self.sam: Sam = self.build_sam_vit_h(model_args.checkpoint)
        self.set_model(model_args)

    @property
    def prompt_embed_dim(self) -> int:
        return int(self.sam.prompt_encoder.embed_dim)

    def set_model(self, model_args: SamModelArguments):
        for _name, parameter in self.sam.image_encoder.named_parameters():
            parameter.requires_grad = model_args.tune_image_encoder

        for _name, parameter in self.sam.prompt_encoder.named_parameters():
            parameter.requires_grad = model_args.tune_prompt_encoder

        for _name, parameter in self.sam.mask_decoder.named_parameters():
            parameter.requires_grad = model_args.tune_mask_decoder

    def build_sam_vit_h(self, checkpoint):
        return _build_sam(
            encoder_embed_dim=1280,
            encoder_depth=32,
            encoder_num_heads=16,
            encoder_global_attn_indexes=[7, 15, 23, 31],
            checkpoint=checkpoint,
        )

    def get_visual_embs(self, images):
        image_embeddings_list = []
        with torch.no_grad():
            for index in range(images.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.sam.image_encoder(images[index].unsqueeze(0))
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, dim=0)
        return image_embeddings

    @staticmethod
    def _normalize_size_entry(size_entry) -> Tuple[int, int]:
        if isinstance(size_entry, torch.Tensor):
            values = size_entry.detach().cpu().tolist()
        else:
            values = list(size_entry)
        return int(values[0]), int(values[1])

    def decode_masks(self, vlm_seg_hidden_states, image_embs, resize_list, label_list):
        pred_masks_local = []
        for index in range(len(vlm_seg_hidden_states)):
            cur_hidden_states: torch.FloatTensor = vlm_seg_hidden_states[index]

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=cur_hidden_states.unsqueeze(1),
            )

            sparse_embeddings = sparse_embeddings.to(cur_hidden_states.dtype)
            cur_image_emb: torch.FloatTensor = image_embs[index]
            low_res_masks, _iou_predictions = self.sam.mask_decoder(
                image_embeddings=cur_image_emb.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            resize_h, resize_w = self._normalize_size_entry(resize_list[index])
            orig_w, orig_h = self._normalize_size_entry(label_list[index])
            pred_mask_local = self.sam.postprocess_masks(
                low_res_masks,
                input_size=(resize_h, resize_w),
                original_size=(orig_h, orig_w),
            )
            pred_masks_local.append(pred_mask_local[:, 0])
        return pred_masks_local

    def forward(
        self,
        images: torch.FloatTensor,
        masks,
        vlm_seg_hidden_states: List[torch.FloatTensor],
        resize_list,
        label_list,
        mask_counts=None,
    ):
        image_embs = self.get_visual_embs(images)
        pred_masks_layout = self.decode_masks(vlm_seg_hidden_states, image_embs, resize_list, label_list)
        if masks is None or len(masks) == 0:
            return pred_masks_layout, None, None
        gt_masks = unpack_mask_batch(masks, mask_counts=mask_counts, image_size_list=label_list)
        bce, dice = self.compute_mask_losses(pred_masks_layout, gt_masks)
        return pred_masks_layout, bce, dice

    def compute_mask_losses(self, pred_masks, gt_masks):
        mask_bce_loss, mask_dice_loss, num_masks = 0, 0, 0

        assert len(gt_masks) == len(pred_masks), "The number of output and input masks don't match"

        for batch_idx, pred_mask in enumerate(pred_masks):
            batched_gt_mask = gt_masks[batch_idx].to(
                device=pred_mask.device,
                dtype=pred_mask.dtype,
            )

            if batched_gt_mask.shape[0] == 0:
                assert pred_mask.shape[0] == 0, (
                    f"gt_mask.shape: {batched_gt_mask.shape}, pred_mask.shape: {pred_mask.shape}"
                )
                continue

            assert batched_gt_mask.shape == pred_mask.shape, (
                f"gt_mask.shape: {batched_gt_mask.shape}, pred_mask.shape: {pred_mask.shape}"
            )

            mask_bce_loss += (
                self.sigmoid_ce_loss(pred_mask, batched_gt_mask, num_masks=batched_gt_mask.shape[0])
                * batched_gt_mask.shape[0]
            )
            mask_dice_loss += (
                self.dice_loss(pred_mask, batched_gt_mask, num_masks=batched_gt_mask.shape[0])
                * batched_gt_mask.shape[0]
            )
            num_masks += batched_gt_mask.shape[0]

        mask_bce_loss = mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = mask_dice_loss / (num_masks + 1e-8)
        return mask_bce_loss, mask_dice_loss

    def dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, scale=1000, eps=1e-6):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1, 2)
        targets = targets.flatten(1, 2)
        numerator = 2 * (inputs / scale * targets).sum(-1)
        denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
        loss = 1 - (numerator + eps) / (denominator + eps)
        loss = loss.sum() / (num_masks + 1e-8)
        return loss

    def sigmoid_ce_loss(self, inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
        return loss


@SAMModelRegistry.register_sam("sam1")
def get_sam1_model(model_args: SamModelArguments, **_kwargs):
    return Sam1Model(model_args)
