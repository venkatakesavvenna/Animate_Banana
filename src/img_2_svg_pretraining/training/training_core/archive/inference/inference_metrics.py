
from typing import Any, Callable, Dict, List
from tqdm import tqdm
import numpy as np
import hashlib
import torch
import os
from transformers.trainer_utils import EvalPrediction
from img_2_svg_pretraining.training.training_core.validation.eval_utils import prepare_masks, visualize_sample, mask_to_bbox
from img_2_svg_pretraining.training.training_core.validation.layout_map import LayoutMAPAccumulator

class InferenceMetrics():
    """
    Custom ComputeMetrics class tailored for HF Trainer integration.
    
    Flow:
    1. evaluate_batch() is called during the prediction loop.
       -> Calls evaluate_single_image() -> returns Tensor (N, 6)
       -> Calls _stack_and_pad_batch() -> returns Tensor (B, Max_N, 6)
    2. Trainer gathers these tensors across GPUs and Batches.
    3. __call__() is triggered at the end with the massive accumulated tensors.

    below comment imported from Trainer class of transformers
            compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics
    """
    def __init__(self, extract_layout_fn:Callable, layout_classes: List[str]):
        self.extract_layout_fn = extract_layout_fn
        self.layout_classes = layout_classes
        self.class_to_id = {cls: i for i, cls in enumerate(layout_classes)}
        # add an index for undefined class
        self.class_to_id["UNDEFINED"] = len(layout_classes)
        
        # Hugging Face Standard Ignore Index
        # NOTE: Verify this matches the padding index used by the current version of HF Trainer/transformers.
        self.PAD_IDX = -100
        self.layout_map_accumulator: LayoutMAPAccumulator = LayoutMAPAccumulator(layout_classes=layout_classes)

    def __call__(self, image_preds, image_gts):
        """
        
        """

        out_of_range_ids = set()
        # image_preds # (N_img, Np, 7)
        # image_gts   # (N_img, Ng, 6)

        pred_dict = {cls: [] for cls in self.layout_classes}
        gt_dict   = {cls: [] for cls in self.layout_classes}

        # ---------- Predictions ----------
        for img_preds in tqdm(image_preds, desc="Loading Image Preds"):
            for det in img_preds:
                try:
                    if det[0].item() == self.PAD_IDX:
                        continue

                    x1, y1, x2, y2, score, cls_id, img_id = det.tolist()
                    cls_id = int(cls_id)
                    if cls_id >= len(self.layout_classes):
                        out_of_range_ids.add(cls_id)
                        continue   # 🔥 do NOT index layout_classes

                    cls_name = self.layout_classes[cls_id]

                    pred_dict[cls_name].append({
                        "image_id": img_id,
                        "score": score,
                        "bbox": [x1, y1, x2, y2]
                    })
                except Exception as e:
                    print("Error", e)

        # ---------- Ground Truth ----------
        for img_gts in tqdm(image_gts, desc="Loading Image GTs"):
            for det in img_gts:
                try:
                    if det[0].item() == self.PAD_IDX:
                        continue

                    x1, y1, x2, y2, cls_id, img_id = det.tolist()
                    cls_id = int(cls_id)
                    if cls_id >= len(self.layout_classes):
                        out_of_range_ids.add(cls_id)
                        continue   # 🔥 do NOT index layout_classes

                    cls_name = self.layout_classes[cls_id]

                    gt_dict[cls_name].append({
                        "image_id": img_id,
                        "bbox": [x1, y1, x2, y2]
                    })
                except Exception as e:
                    print("Error", e)


        # ---------- mAP accumulation ----------
        self.layout_map_accumulator.preds  = pred_dict
        self.layout_map_accumulator.gts    = gt_dict
        output_metrics = self.layout_map_accumulator.compute_coco_map()
        return output_metrics

    def _hash_string_to_float(self, s: str) -> float:
        """
        Creates a deterministic float hash from a string ID.
        Useful for passing IDs through the Tensor-only barrier of HF Trainer.
        """
        # Truncate to ensure it fits in float32/64 precision without losing too much info
        h = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
        return float(h % 10**9)
    
    def get_layouts(self, decoded_layouts):
        # logits -> (B, seq_len)
        layout_classes = []
        for cur_image_str in decoded_layouts:
            cur_image_labels = self.extract_layout_fn(cur_image_str)
            layout_classes.append(cur_image_labels)
        return layout_classes
    
    def evaluate_batch(
        self,
        pred_masks, gt_masks,
        labels, preds,
        original_images,
        debug_save_dir,
        device,
        threshold=0.0
    ):  
        pred_classes = self.get_layouts(preds) # B, Ni -> Number of Layout Classes predicted per image
        gt_classes = self.get_layouts(labels)  # B, Ngt -> Number of Layout Classes per image (GT)

        os.makedirs(debug_save_dir, exist_ok=True)

        B = len(pred_masks) # TBD
        # Lists to hold results before stacking
        batch_pred_tensors = []
        batch_gt_tensors = []
        for i in range(B):
            orig = None
            if original_images is not None:
                v = original_images[i]
                orig = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)
            cur_img_preds,cur_img_gts = self.evaluate_single_image(
                pred_masks[i], gt_masks[i], gt_classes[i], pred_classes[i], orig,
                i, threshold, debug_save_dir, device
            )
            batch_pred_tensors.append(cur_img_preds)
            batch_gt_tensors.append(cur_img_gts)
        return batch_pred_tensors, batch_gt_tensors

    def evaluate_single_image(self,
        pm, gm, gt_labels: List[str], pred_labels: List[str], orig,
        img_idx, threshold, debug_save_dir, device):
        
        if pm.sum() == 0 and gm.sum() == 0:
            return torch.empty((0, 7), device=device), torch.empty((0, 6), device=device)  # empty contribution

        pm_np, gm_np, _ = prepare_masks(pm, gm, threshold)
 
        image_id = os.path.join(debug_save_dir, str(img_idx))
        image_hash = self._hash_string_to_float(image_id)

        def mask_to_detections(mask_np: np.ndarray, labels: List[str], is_pred: bool):
            detections = []  # Initialize outside the loop
            for idx in range(mask_np.shape[0]):
                cls_name = labels[idx] if idx<len(labels) else "UNDEFINED"
                
                # Filter unwanted classes
                if cls_name not in self.class_to_id:
                    continue

                cls_id = float(self.class_to_id[cls_name])

                bbox: List[int] = mask_to_bbox(mask_np[idx])
                if bbox is not None:
                    # Construct Vector: [x1, y1, x2, y2, score, class_id, image_id_hash] for preds
                    #                   [x1, y1, x2, y2, class_id, image_id_hash] for GT
                    # Score is hardcoded to 1.0
                    if is_pred:
                        detections.append([*bbox, 1.0, cls_id, image_hash])
                    else:
                        detections.append([*bbox, cls_id, image_hash])
            if not detections:
                return torch.empty((0, 7 if is_pred else 6), device=device)
            return torch.tensor(detections, dtype = torch.float32, device=device)
        
        # ---- Predictions ----
        pred_detections = mask_to_detections(pm_np, pred_labels, is_pred=True)
        gt_detections = mask_to_detections(gm_np, gt_labels, is_pred=False)
        
        # ---- visualization ----
        if debug_save_dir is not None and orig is not None:
            # Convert to list for visualization (remove image_hash)
            pred_list = [[*det[:6]] for det in pred_detections.cpu().tolist()] if len(pred_detections) > 0 else []
            gt_list = [[*det[:5]] for det in gt_detections.cpu().tolist()] if len(gt_detections) > 0 else []
            visualize_sample(
                orig, pred_list, gt_list, f"{image_id}.png", self.layout_classes
            )

        return pred_detections, gt_detections