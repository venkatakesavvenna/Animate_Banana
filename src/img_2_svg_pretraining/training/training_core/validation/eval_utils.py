from typing import Union, List
import numpy as np
# from docgrounding.utils.padding import PAD_VALUE
import torch
import os
import cv2

def prepare_masks(pm, gm, threshold):
    # valid_mask = (gm != PAD_VALUE)
    # if isinstance(valid_mask, torch.Tensor):
    #     valid_mask = valid_mask.cpu().numpy()
    valid_mask = None
    pm_np = to_numpy_mask(pm, threshold)
    gm_np = to_numpy_mask(gm, threshold)
    return pm_np, gm_np, valid_mask

def to_numpy_mask(
    x: Union[torch.Tensor, np.ndarray, List],
    threshold: float
) -> np.ndarray:
    """
    Converts tensor / numpy / list of tensors to a binary numpy uint8 mask.
    
    - Tensor → np.ndarray
    - np.ndarray → np.ndarray
    - List[Tensor or np.ndarray] → concatenated np.ndarray along dim=0
    """

    def _convert_single(item) -> np.ndarray:
        if isinstance(item, torch.Tensor):
            arr = item.detach().cpu().float().numpy()
        else:
            arr = np.asarray(item)

        if arr.dtype != np.uint8:
            arr = (arr > threshold).astype(np.uint8)

        return arr

    # -------- list / tuple case --------
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("to_numpy_mask received an empty list")

        arrays = [_convert_single(item) for item in x]

        # sanity check: all shapes except dim-0 must match
        ref_shape = arrays[0].shape[1:]
        for i, arr in enumerate(arrays):
            if arr.shape[1:] != ref_shape:
                raise ValueError(
                    f"Shape mismatch at index {i}: "
                    f"{arr.shape} vs expected (*, {ref_shape})"
                )

        return np.stack(arrays, axis=0)

    # -------- single tensor / array --------
    return _convert_single(x)

def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray, eps: float = 1e-8) -> float:
    inter = ((pred_mask & gt_mask) & valid_mask).sum()
    union = ((pred_mask | gt_mask) & valid_mask).sum()
    return float((inter + eps) / (union + eps))

def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray, eps: float = 1e-8) -> float:
    inter = ((pred_mask & gt_mask) & valid_mask).sum()
    denom = (pred_mask & valid_mask).sum() + (gt_mask & valid_mask).sum()
    return float((2 * inter + eps) / (denom + eps))

def pixel_accuracy(self, pred_mask: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray) -> float:
    return float(((pred_mask == gt_mask) & valid_mask).sum() / valid_mask.sum())

def visualize_sample(
    vis, pred_detections, gt_detections, fname, layout_classes=None
):
    """
    Visualize predictions and ground truth bounding boxes on an image.
    
    Args:
        vis: Image as numpy array or torch.Tensor (H, W, C) or (C, H, W)
        pred_detections: List of predictions, each is [x1, y1, x2, y2, score, class_id]
        gt_detections: List of ground truths, each is [x1, y1, x2, y2, class_id]
        fname: Output filename path
        layout_classes: Optional list of class names for mapping class_id to class name
    """
    if isinstance(vis, torch.Tensor):
        vis = vis.detach().cpu().numpy()
    vis = np.array(vis).astype(np.uint8)
    if vis.ndim == 3 and vis.shape[0] in [1, 3]:
        vis = np.transpose(vis, (1, 2, 0))
    overlay = vis.copy()

    COLOR_GT   = (255, 0, 0)   # Red
    COLOR_PRED = (0, 255, 0)   # Green
    
    # ---- Draw GT ----
    for detection in gt_detections:
        if len(detection) < 5:
            continue
        x1, y1, x2, y2, class_id = detection[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id_int = int(class_id) # Get class name if layout_classes is provided
        if layout_classes and class_id_int < len(layout_classes):
            cls_name = layout_classes[class_id_int]
        else:
            cls_name = f"cls{class_id_int}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_GT, 2)
        cv2.putText(
            overlay, f"GT:{cls_name}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GT, 1
        )
    
    # ---- Draw PRED ----
    for detection in pred_detections:
        if len(detection) < 6:
            continue
        x1, y1, x2, y2, score, class_id = detection[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id_int = int(class_id) # Get class name if layout_classes is provided
        if layout_classes and class_id_int < len(layout_classes):
            cls_name = layout_classes[class_id_int]
        else:
            cls_name = f"cls{class_id_int}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PRED, 2)
        cv2.putText(
            overlay, f"P:{cls_name}",
            (x1, min(vis.shape[0] - 5, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_PRED, 1
        )

    cv2.imwrite(fname, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def visualize_sample_pretrain(
    vis, pred_detections, gt_detections, fname, layout_classes=None
):
    """
    Visualize predictions and ground truth bounding boxes on an image.
    
    Args:
        vis: Image as numpy array or torch.Tensor (H, W, C) or (C, H, W)
        pred_detections: List of predictions, each is [x1, y1, x2, y2, score, predicted_label]
        gt_detections: List of ground truths, each is [x1, y1, x2, y2, label]
        fname: Output filename path
        layout_classes: Optional list of class names for mapping class_id to class name
    """
    if isinstance(vis, torch.Tensor):
        vis = vis.detach().cpu().numpy()
    vis = np.array(vis).astype(np.uint8)
    if vis.ndim == 3 and vis.shape[0] in [1, 3]:
        vis = np.transpose(vis, (1, 2, 0))
    overlay = vis.copy()

    COLOR_GT   = (255, 0, 0)   # Red
    COLOR_PRED = (0, 255, 0)   # Green
    
    # ---- Draw GT ----
    for detection in gt_detections:
        if len(detection) < 5:
            continue
        x1, y1, x2, y2, name = detection[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_GT, 2)
        cv2.putText(
            overlay, f"GT:{name}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GT, 1
        )
    
    # ---- Draw PRED ----
    for detection in pred_detections:
        if len(detection) < 6:
            continue
        x1, y1, x2, y2, score, name = detection[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PRED, 2)
        cv2.putText(
            overlay, f"P:{name}",
            (x1, min(vis.shape[0] - 5, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_PRED, 1
        )

    cv2.imwrite(fname, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def mask_to_bbox(mask: np.ndarray):
    """
    Compute tight axis-aligned bounding box from binary mask
    using contours. Returns [x1, y1, x2, y2] or None.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None
    cnt = max(contours, key=cv2.contourArea) # take largest connected component
    x, y, w, h = cv2.boundingRect(cnt)
    return [int(x), int(y), int(x + w), int(y + h)]
