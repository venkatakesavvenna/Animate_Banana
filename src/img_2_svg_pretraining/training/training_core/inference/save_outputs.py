"""Save_Outputs: persists per-image inference results to disk.

Supports both VLM+SAM (``pred_masks is not None``) and VLM-only
(``pred_masks is None``) checkpoints.

* **VLM+SAM**: saves ``pred_masks.npy``, ``gt_masks.npy``, ``detections.json``,
  ``original_image.png``, and an annotated ``viz_image.png``.
* **VLM-only**: saves a text-only ``detections.json`` containing
  ``{"pred_str", "label_str", "pred_classes"}`` — no image/mask artifacts.

The branching key is the explicit ``has_masks`` flag (or presence of non-None
pred_masks per image), *not* duck-typing on the mask object, so a broken
caller gets a clear ``TypeError`` rather than a silently wrong branch.
"""

from typing import Callable, List, Optional
import numpy as np
import torch
import json
import os
from PIL import Image
from img_2_svg_pretraining.training.training_core.validation.eval_utils import prepare_masks, visualize_sample_pretrain, mask_to_bbox


class Save_Outputs:
    def __init__(self, extract_layout_fn: Callable):
        self.extract_layout_fn = extract_layout_fn

    def get_layouts(self, decoded_layouts):
        layout_classes = []
        for cur_image_str in decoded_layouts:
            cur_image_labels = self.extract_layout_fn(cur_image_str)
            layout_classes.append(cur_image_labels)
        return layout_classes

    def evaluate_batch(
        self,
        pred_masks,        # List[Tensor | None] or None — None means VLM-only
        gt_masks,          # List[Tensor | None] or None
        labels,            # List[str]  — ground-truth decoded strings
        preds,             # List[str]  — predicted decoded strings
        original_images,
        debug_save_dir,
        threshold: float = 0.0,
        has_masks: Optional[bool] = None,  # explicit flag; auto-detected when None
    ):
        """Evaluate and save results for a batch of images.

        Args:
            pred_masks:    Per-image predicted masks (VLM+SAM) or ``None``/list
                           of ``None`` (VLM-only).
            gt_masks:      Per-image ground-truth masks or ``None``.
            labels:        Ground-truth decoded strings.
            preds:         Predicted decoded strings.
            original_images: List of original images (Tensor or array) or ``None``.
            debug_save_dir: Root directory for per-image output subdirs.
            threshold:     Binarisation threshold for mask arrays.
            has_masks:     Explicit flag indicating whether masks are expected.
                           When ``None``, auto-detected from ``pred_masks``.
        """
        pred_classes = self.get_layouts(preds)
        gt_classes = self.get_layouts(labels)

        os.makedirs(debug_save_dir, exist_ok=True)

        # Auto-detect has_masks when not provided explicitly
        if has_masks is None:
            has_masks = (
                pred_masks is not None
                and len(pred_masks) > 0
                and pred_masks[0] is not None
            )

        B = len(preds)
        for i in range(B):
            orig = None
            if original_images is not None:
                v = original_images[i]
                orig = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)

            pm = pred_masks[i] if (pred_masks is not None and i < len(pred_masks)) else None
            gm = gt_masks[i] if (gt_masks is not None and i < len(gt_masks)) else None

            self.evaluate_single_image(
                pm=pm,
                gm=gm,
                gt_labels=gt_classes[i],
                pred_labels=pred_classes[i],
                orig=orig,
                img_idx=i,
                threshold=threshold,
                debug_save_dir=debug_save_dir,
                preds_str=preds[i],
                labels_str=labels[i],
                has_masks=has_masks,
            )

    def evaluate_single_image(
        self,
        pm,                     # predicted mask Tensor or None
        gm,                     # GT mask Tensor or None
        gt_labels: List[str],
        pred_labels: List[str],
        orig,
        img_idx: int,
        threshold: float,
        debug_save_dir: str,
        preds_str: str,
        labels_str: str,
        has_masks: bool = True,
    ):
        """Persist outputs for a single image.

        When ``has_masks=False`` (VLM-only), only a text-only
        ``detections.json`` is written; mask/image artifacts are skipped.
        When ``has_masks=True`` (VLM+SAM), the full artifact set is written
        (masks, bbox, visualization) but only when at least one mask is
        non-zero.
        """
        image_id = os.path.join(debug_save_dir, str(img_idx))
        os.makedirs(image_id, exist_ok=True)

        if not has_masks:
            # VLM-only path: text-only output
            payload = {
                "pred_str": preds_str,
                "label_str": labels_str,
                "pred_classes": pred_labels,
                "gt_classes": gt_labels,
            }
            with open(os.path.join(image_id, "detections.json"), "w") as f:
                json.dump(payload, f, indent=2)
            return

        # VLM+SAM path — guard every mask operation
        if pm is None or gm is None:
            return

        if pm.sum() == 0 and gm.sum() == 0:
            return

        pm_np, gm_np, _ = prepare_masks(pm, gm, threshold)

        def mask_to_detections(mask_np: np.ndarray, label_list: List[str], is_pred: bool):
            detections = []
            for idx in range(mask_np.shape[0]):
                cls_name = label_list[idx] if idx < len(label_list) else "UNDEFINED"
                if cls_name == "UNDEFINED":
                    print("UNDEFINED CLASS - CHECK LABEL EXTRACTION LOGIC", label_list)
                bbox: List[int] = mask_to_bbox(mask_np[idx])
                if bbox is not None:
                    entry = {"bbox": bbox, "label": cls_name}
                    if is_pred:
                        entry["score"] = 1.0
                    detections.append(entry)
            return detections

        pred_detections = mask_to_detections(pm_np, pred_labels, is_pred=True)
        gt_detections = mask_to_detections(gm_np, gt_labels, is_pred=False)

        # ---- Save original image ----
        if orig is not None:
            orig_hwc = np.transpose(orig, (1, 2, 0)) if orig.ndim == 3 and orig.shape[0] in [1, 3] else orig
            Image.fromarray(orig_hwc.astype(np.uint8)).save(os.path.join(image_id, "original_image.png"))

        # ---- Save masks ----
        np.save(os.path.join(image_id, "pred_masks.npy"), pm_np)
        np.save(os.path.join(image_id, "gt_masks.npy"), gm_np)

        # ---- Save detections JSON ----
        with open(os.path.join(image_id, "detections.json"), "w") as f:
            json.dump({
                "predictions": pred_detections,
                "ground_truth": gt_detections,
                "pred_str": preds_str,
                "label_str": labels_str,
            }, f, indent=2)

        # ---- Save visualization ----
        if orig is not None:
            pred_list = [[*d["bbox"], d["score"], d["label"]] for d in pred_detections]
            gt_list = [[*d["bbox"], d["label"]] for d in gt_detections]
            visualize_sample_pretrain(
                orig, pred_list, gt_list, os.path.join(image_id, "viz_image.png")
            )
