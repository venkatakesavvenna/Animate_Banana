import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import Trainer, TrainerState

# NVTX range markers for Nsight Systems / torch.profiler.
# No-op when no profiler is attached — essentially free.
_nvtx = getattr(torch.cuda, "nvtx", None)

from img_2_svg_pretraining.training.training_core.validation.compute_metrics import ComputeMetrics

_logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    """Trainer for VLMSam models with mask-prediction metrics and per-layer LR support.

    Per-layer LR is config-driven: pass ``param_groups_cfg`` (a list of
    ``{pattern, lr, weight_decay}`` dicts from the YAML) to activate it.
    """

    def __init__(self, *args, param_groups_cfg: Optional[List[Dict]] = None, **kwargs):
        self._param_groups_cfg = param_groups_cfg or []
        self._efficiency_cb = None  # set by train.py after callback registration
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if _nvtx is not None:
            _nvtx.range_push("training_step")
        try:
            cb = self._efficiency_cb
            if cb is not None:
                stats = cb.step_token_stats
                ids = inputs.get("input_ids")
                labels = inputs.get("labels")
                if isinstance(ids, torch.Tensor):
                    stats["text"] += ids.numel()
                    stats["samples"] += ids.shape[0]
                if isinstance(labels, torch.Tensor):
                    stats["active"] += int((labels != -100).sum())
            return super().training_step(model, inputs, num_items_in_batch, **kwargs)
        finally:
            if _nvtx is not None:
                _nvtx.range_pop()

    def create_optimizer(self):
        if not self._param_groups_cfg:
            return super().create_optimizer()

        from img_2_svg_pretraining.training.training_core.train.train_utils import build_param_groups
        from transformers.trainer_utils import get_last_checkpoint
        from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

        base_lr = self.args.learning_rate
        base_wd = self.args.weight_decay
        param_groups, _ = build_param_groups(
            self.model, self._param_groups_cfg, base_lr, base_wd
        )
        # Strip internal label before passing to optimizer
        clean_groups = [
            {k: v for k, v in g.items() if k != "_group_label"}
            for g in param_groups
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        # optimizer_kwargs already contains 'lr' and 'weight_decay'; those
        # will be overridden per-group, so we only keep non-per-group keys
        # (betas, eps, etc.) from kwargs.
        optimizer_kwargs.pop("lr", None)
        optimizer_kwargs.pop("weight_decay", None)
        self.optimizer = optimizer_cls(clean_groups, **optimizer_kwargs)
        _logger.info("[CustomTrainer] Per-layer optimizer created with %d param groups.", len(clean_groups))
        return self.optimizer
    
    # acts as a wrapper to Trainer._prepare_inputs() so that out custom CPU tesors/data_variables, stay on cpu
    # actual _prepare_inputs recursively calls self._prepare_input on all tensors(even inside dicts, lists, etc.)
    # Inside self._prepare_input(), the actual data.to() call happens.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """
        This self._prepare_inputs is called in both training_step() and prediction_step().
        Collator function returns a dict in which all the tensors/items(whatever we pass) are on cpu itself. `inputs` is this dict. 
        We intercept this dict and make sure the additional debug variables that we are passing, are not passed to super._prepare_inputs(the default).
        This makes sure that only the required tensors(on which we perform gpu operations) go into self._prepare_input and are managed for deepspeed/multigpu training.
        """
        # 1. Extract all debug_* keys
        debug_items = {
            k: inputs.pop(k)
            for k in list(inputs.keys())
            if k.startswith("debug_")
        }

        # 2. Let HF/DeepSpeed move the rest to device
        inputs = super()._prepare_inputs(inputs)

        # 3. Re-attach debug_* items (remain on CPU)
        inputs.update(debug_items)

        return inputs
    
    # acts as a warpper around Trainer.prediction_step function.
    # This function must stictly return only a tuple of size 3 (loss, logits, labels).
    # We save the visualised images + prediction_masks and then discard, so that accumulation across batches does not happen(memory explosion).
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        """
        returned value --> (loss, logits, labels)
        logits and lables are paddded across processes, then sent to preprocess_for_metrics
        """
        # VLM-only path (no compute_metrics / no SAM head): skip mask logic entirely.
        if self.compute_metrics is None:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        original_labels = inputs.get("labels")
        loss,logits,labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        # this logits is a tuple containing all the values of dict returned by forward, except the loss which is returned separately.
        if labels is None:
            labels = original_labels

        original_image_list: List[torch.Tensor] = logits[-1] # on cpu -> List[ #(H_orig x W_orig x 3) ]
        gt_masks: List[List[torch.FloatTensor]] = logits[-2] # on gpu -> List[List[ #(H_orig x W_orig) ]]
        pred_masks: List[torch.FloatTensor] = logits[-3] # on gpu -> List[ #(Ni x H_orig x W_orig)] 
        qwen_logits = logits[-4] # on gpu #(B, seq_len, vocab_size)

        log_ids = torch.argmax(qwen_logits, dim =-1).detach().cpu().numpy() # (B, seq_len)
        cpu_labels = labels.detach().cpu().numpy()

        state: TrainerState = self.state
        debug_images_dir_cur_batch_rank = os.path.join(self.args.logging_dir, str(state.global_step), str(self.args.process_index))
        os.makedirs(debug_images_dir_cur_batch_rank,exist_ok=True)

        # below is generally a function, but we have sent a class, so we can access other elements of the class also.
        compute_metrics : ComputeMetrics = self.compute_metrics

        image_preds: torch.Tensor = None # Accumulation for mAP (B, Max_Np, 7)
        image_gts: torch.Tensor = None # Accumulation for mAP (B, Max_Ng, 6)
        metrics_device = loss.device if loss is not None else qwen_logits.device
        image_preds, image_gts = compute_metrics.evaluate_batch(
            pred_masks,
            gt_masks,
            cpu_labels,
            log_ids,
            original_image_list,
            debug_images_dir_cur_batch_rank,
            metrics_device,
            threshold=0,
        )
        # pred_masks = List of size=Batch_size. -> each item of this batch is a tensor of size (Ni, hi, wi) where Ni is the number of masks for each image.

        # Accelerate's all_gather calls view(-1, *shape[1:]) on gathered tensors.  When
        # Max_Np or Max_Ng is 0 (no predictions / no GT in this batch) the view fails with
        # "cannot reshape tensor of 0 elements".  Pad to at least 1 row with -100 (the
        # PAD_IDX that compute_metrics already skips) so the gather always succeeds.
        _pad = self.compute_metrics.PAD_IDX if hasattr(self.compute_metrics, "PAD_IDX") else -100
        image_preds = self._ensure_nonempty_rows(image_preds, n_cols=7, pad_val=_pad, device=metrics_device)
        image_gts   = self._ensure_nonempty_rows(image_gts,   n_cols=6, pad_val=_pad, device=metrics_device)

        # Return as tuple of tensors (not dict!) so Trainer can properly gather across batches/processes
        # The tuple will be accessible in compute_metrics as eval_preds.predictions
        B = qwen_logits.size(0)
        empty_labels = torch.zeros((B, 1), device=metrics_device)
        return loss, (image_preds, image_gts), empty_labels # we must make sure our compute_metrics does not acces the eval_preds.label_ids.

    @staticmethod
    def _ensure_nonempty_rows(t: torch.Tensor, n_cols: int, pad_val: float, device) -> torch.Tensor:
        """Return t unchanged unless dim-1 is 0, in which case pad with one sentinel row."""
        if t is None or t.size(1) > 0:
            return t
        B = t.size(0)
        return torch.full((B, 1, n_cols), fill_value=pad_val, dtype=t.dtype, device=device)
