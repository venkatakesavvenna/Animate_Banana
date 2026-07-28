from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple

# -----------------------------
# Box IoU
# -----------------------------
def box_iou(box1: List[int], box2: List[int]) -> float:
    """
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-8
    return inter / union

# -----------------------------
# COCO-style AP (continuous)
# -----------------------------
def coco_ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    COCO-style AP: continuous area under the PR curve.
    Uses precision envelope and numeric integration.
    """
    # Append sentinel points
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Precision envelope (monotonic decreasing)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Integrate area where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

# -----------------------------
# Accumulator
# -----------------------------
class LayoutMAPAccumulator:
    """
    Accumulates box predictions across images and computes 
    COCO-Style AP50-95 (mean over IoU thresholds 0.50:0.05:0.95)
    """
    def __init__(self, layout_classes: List[str]):
        self.layout_classes = layout_classes
        self.preds: Dict[str, List[Dict]] = {c: [] for c in layout_classes}
        self.gts:   Dict[str, List[Dict]] = {c: [] for c in layout_classes}

    def update(self, image_result):
        image_id = image_result["image_id"]

        for cls in self.layout_classes:
            for p in image_result["predictions"].get(cls, []):
                self.preds[cls].append({
                    "image_id": image_id,
                    "bbox": p["bbox"],
                    "score": p["score"]
                })

            for g in image_result["ground_truth"].get(cls, []):
                self.gts[cls].append({
                    "image_id": image_id,
                    "bbox": g["bbox"]
                })

    def _ap_at_iou(self, cls: str, iou_thr: float) -> float:
        preds = self.preds[cls]
        gts = self.gts[cls]
        if len(gts) == 0:
            return np.nan

        preds = sorted(preds, key=lambda x: float(x.get("score", 1.0)), reverse=True)

        gt_used = np.zeros(len(gts), dtype=bool)

        tp = np.zeros(len(preds), dtype=np.float64)
        fp = np.zeros(len(preds), dtype=np.float64)

        for i, p in enumerate(preds):
            best_iou = 0.0
            best_j = -1

            for j, gt in enumerate(gts):
                if gt_used[j]:
                    continue
                if gt["image_id"] != p["image_id"]:
                    continue

                iou = box_iou(p["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j != -1 and best_iou >= iou_thr:
                tp[i] = 1.0
                gt_used[best_j] = True
            else:
                fp[i] = 1.0

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        rec = tp / (len(gts) + 1e-8)
        prec = tp / (tp + fp + 1e-8)

        return coco_ap_from_pr(rec, prec)
    
    def compute_coco_map(self) -> Dict[str, float]:
        """
        Returns:
        {
          "AP50": float,
          "AP75": float,
          "AP50_95": float,
          "AP_per_class": {
              cls: {
                "AP50": float,
                "AP75": float,
                "AP50_95": float
              }
          }
        }
        """
        iou_thresholds = np.arange(0.50, 0.96, 0.05)

        per_class = {}
        ap50_list = []
        ap75_list = []
        ap5095_list = []

        for cls in tqdm(self.layout_classes, desc="Computing AP per class"):
            aps = []
            for t in iou_thresholds:
                ap_t = self._ap_at_iou(cls, t)
                if not np.isnan(ap_t):
                    aps.append(ap_t)

            if len(aps) == 0:
                continue

            ap50 = self._ap_at_iou(cls, 0.50)
            ap75 = self._ap_at_iou(cls, 0.75)
            ap5095 = float(np.mean(aps))

            per_class[cls] = {
                "AP50": ap50,
                "AP75": ap75,
                "AP50_95": ap5095
            }

            ap50_list.append(ap50)
            ap75_list.append(ap75)
            ap5095_list.append(ap5095)

        return {
            "AP50": float(np.mean(ap50_list)) if ap50_list else 0.0,
            "AP75": float(np.mean(ap75_list)) if ap75_list else 0.0,
            "AP50_95": float(np.mean(ap5095_list)) if ap5095_list else 0.0,
            # "AP_per_class": per_class
        }