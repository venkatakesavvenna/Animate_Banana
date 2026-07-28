"""Generic, task-agnostic streaming metric accumulation for img_2_svg_pretraining training inference.

Design
------
``MetricAccumulator`` is an ABC that defines the interface for accumulating
inference results and computing final metrics.  All implementations must be
**picklable** so they can be used across ``multiprocessing`` workers.

Concrete implementations (auto-selected via ``from_annotation_spec``):

* ``LayoutDetectionMetric`` — bbox mAP (AP50 / AP75 / AP50-95) for
  ``has_localization=True`` datasets.  Requires mask→bbox extraction.
* ``CaptioningMetric``      — BLEU-4 and exact-match for free-form text tasks.
* ``NoOpMetric``            — silent no-op for tasks not yet instrumented.

Adding a new task metric
------------------------
1. Subclass ``MetricAccumulator`` and implement ``update``, ``compute``,
   ``merge``, and register it in ``_AUTO_SELECT`` or override
   ``from_annotation_spec`` logic at the bottom of this file.
2. ``update()`` receives a standardised ``BatchResult`` dict; add any new
   required keys to the ``BatchResult`` TypedDict below.

Usage in multiprocessing
------------------------
The inference loop submits batch results to a ``ProcessPoolExecutor``:

    future = pool.submit(_worker_update, accumulator, batch_result)

Each worker returns a new ``MetricAccumulator`` with the batch's contribution.
After all futures resolve, they are merged into one final accumulator:

    final = NoOpMetric()  # or empty instance of the right class
    for f in futures:
        final.merge(f.result())
    metrics = final.compute()
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from img_2_svg_pretraining.training.training_core.registry.utils import AnnotationSpec

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BatchResult type (standard dict contract between inference loop and workers)
# ---------------------------------------------------------------------------

# Keys guaranteed to be present in every BatchResult passed to update():
#   preds       — List[str]  decoded model predictions (one per sample)
#   labels      — List[str]  decoded ground-truth strings (one per sample)
#   pred_masks  — List[np.ndarray] shape (Ni_pred, H, W) uint8, or None
#   gt_masks    — List[np.ndarray] shape (Ni_gt,  H, W) uint8, or None
#   image_ids   — List[str]  stable IDs for AP computation (one per sample)
#
# BatchResult is intentionally a plain dict (not a dataclass) so it can be
# freely extended without changing existing accumulator code.
BatchResult = Dict[str, Any]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class MetricAccumulator(ABC):
    """Generic interface for streaming, task-agnostic metric accumulation.

    All state must be **picklable** (no GPU tensors, no non-serialisable
    lambdas).  Convert tensors to numpy/lists before calling ``update()``.
    """

    @abstractmethod
    def update(self, batch_result: BatchResult) -> None:
        """Accumulate results from one inference batch."""

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Return final metric dict after all batches have been processed."""

    @abstractmethod
    def merge(self, other: "MetricAccumulator") -> None:
        """Merge another accumulator's partial state into self.

        Used to reduce results from multiple worker processes into one final
        accumulator on the main process.
        """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_annotation_spec(cls, spec: "AnnotationSpec") -> "MetricAccumulator":
        """Auto-select the appropriate accumulator from a dataset's AnnotationSpec.

        Selection priority:
        1. ``has_localization`` with bbox/mask type → ``LayoutDetectionMetric``
        2. ``parse_predictions_fn`` set (text parsing) → ``CaptioningMetric``
        3. Fallback → ``NoOpMetric``
        """
        if spec.has_localization and spec.localization_type in (
            "bbox", "bbox+mask", "point+mask"
        ):
            return LayoutDetectionMetric(label_classes=spec.label_classes or [])

        if spec.parse_predictions_fn is not None:
            return CaptioningMetric(parse_fn=spec.parse_predictions_fn)

        _logger.info(
            "[MetricAccumulator] No metric registered for annotation spec "
            "(has_localization=%s, type=%s). Using NoOpMetric.",
            spec.has_localization,
            spec.localization_type,
        )
        return NoOpMetric()


# ---------------------------------------------------------------------------
# Layout detection metric (mask → bbox → AP)
# ---------------------------------------------------------------------------

class LayoutDetectionMetric(MetricAccumulator):
    """Computes bbox mAP (AP50 / AP75 / AP50-95) for document layout tasks.

    Works with any dataset whose annotation_spec has ``has_localization=True``
    and a bbox/mask localization type.  Mask segmentation maps are converted
    to tight bounding boxes on-the-fly.

    Internally uses ``LayoutMAPAccumulator`` from ``validation/layout_map.py``.
    ``parse_predictions_fn`` from the ``AnnotationSpec`` is used to extract
    class-name labels from decoded prediction strings.
    """

    def __init__(
        self,
        label_classes: List[str],
        parse_fn=None,
    ):
        self.label_classes = list(label_classes)
        self.parse_fn = parse_fn  # optional: override with annotation_spec.parse_predictions_fn
        # Internal storage: list of per-image dicts ready for LayoutMAPAccumulator
        self._image_results: List[Dict] = []

    # -- pickling / copying support --
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __copy__(self):
        """Return a fresh empty accumulator with the same config (label_classes, parse_fn).

        This is used by ``copy.copy(accumulator_template)`` in the inference
        loop to create a per-batch worker accumulator that starts empty but has
        the same configuration as the template.
        """
        return LayoutDetectionMetric(
            label_classes=list(self.label_classes),
            parse_fn=self.parse_fn,
        )

    def update(self, batch_result: BatchResult) -> None:
        """Extract detections from one batch and append to internal state."""
        preds: List[str] = batch_result.get("preds", [])
        labels: List[str] = batch_result.get("labels", [])
        pred_masks = batch_result.get("pred_masks")   # List[np.ndarray] | None
        gt_masks = batch_result.get("gt_masks")       # List[np.ndarray] | None
        image_ids: List[str] = batch_result.get("image_ids", [f"img_{i}" for i in range(len(preds))])

        parse_fn = self.parse_fn
        if parse_fn is None:
            # Fallback: treat entire string as a single-class label
            _parse = lambda s: [s.strip()] if s.strip() else []
        else:
            _parse = parse_fn

        B = len(preds)
        for i in range(B):
            image_id = image_ids[i] if i < len(image_ids) else f"img_{i}"
            pred_labels = _parse(preds[i]) if i < len(preds) else []
            gt_labels   = _parse(labels[i]) if i < len(labels) else []

            pred_dets = {}
            gt_dets   = {}

            if pred_masks is not None and i < len(pred_masks):
                pm = pred_masks[i]  # (Np, H, W) uint8
                for mi in range(pm.shape[0]):
                    cls = pred_labels[mi] if mi < len(pred_labels) else "UNDEFINED"
                    if cls not in pred_dets:
                        pred_dets[cls] = []
                    bbox = _mask_to_bbox(pm[mi])
                    if bbox is not None:
                        pred_dets[cls].append({"bbox": bbox, "score": 1.0})

            if gt_masks is not None and i < len(gt_masks):
                gm = gt_masks[i]    # (Ng, H, W) uint8
                for mi in range(gm.shape[0]):
                    cls = gt_labels[mi] if mi < len(gt_labels) else "UNDEFINED"
                    if cls not in gt_dets:
                        gt_dets[cls] = []
                    bbox = _mask_to_bbox(gm[mi])
                    if bbox is not None:
                        gt_dets[cls].append({"bbox": bbox})

            self._image_results.append({
                "image_id": image_id,
                "predictions": pred_dets,
                "ground_truth": gt_dets,
            })

    def compute(self) -> Dict[str, float]:
        """Compute and return mAP metrics across all accumulated images."""
        from img_2_svg_pretraining.training.training_core.validation.layout_map import LayoutMAPAccumulator

        accumulator = LayoutMAPAccumulator(layout_classes=self.label_classes)
        for image_result in self._image_results:
            accumulator.update(image_result)
        return accumulator.compute_coco_map()

    def merge(self, other: "LayoutDetectionMetric") -> None:
        """Merge another LayoutDetectionMetric's image results into self."""
        if not isinstance(other, LayoutDetectionMetric):
            raise TypeError(f"Cannot merge {type(other)} into LayoutDetectionMetric")
        self._image_results.extend(other._image_results)


# ---------------------------------------------------------------------------
# Captioning metric (BLEU-4 + exact match)
# ---------------------------------------------------------------------------

class CaptioningMetric(MetricAccumulator):
    """Computes BLEU-4 and exact-match accuracy for free-form text prediction.

    Uses ``sacrebleu`` for BLEU-4 (optional dependency — degrades gracefully
    if not installed).  Exact match is always computed.

    ``parse_fn`` is optional; if provided it extracts the canonical string
    from a raw model output (e.g. strips layout tokens, normalises case).
    When ``None`` the raw decoded string is used as-is.
    """

    def __init__(self, parse_fn=None):
        self.parse_fn = parse_fn
        self._hypotheses: List[str] = []
        self._references: List[str] = []

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __copy__(self):
        """Return a fresh empty accumulator preserving parse_fn config."""
        return CaptioningMetric(parse_fn=self.parse_fn)

    def update(self, batch_result: BatchResult) -> None:
        preds: List[str]  = batch_result.get("preds", [])
        labels: List[str] = batch_result.get("labels", [])
        _fn = self.parse_fn

        for pred, label in zip(preds, labels):
            hyp = (_fn(pred) if _fn else pred).strip()
            ref = (_fn(label) if _fn else label).strip()
            self._hypotheses.append(hyp)
            self._references.append(ref)

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Exact match
        if self._hypotheses:
            exact = sum(h == r for h, r in zip(self._hypotheses, self._references))
            metrics["exact_match"] = exact / len(self._hypotheses)

        # BLEU-4 via sacrebleu
        try:
            import sacrebleu as _sacrebleu
            bleu = _sacrebleu.corpus_bleu(self._hypotheses, [self._references])
            metrics["bleu4"] = bleu.score
        except ImportError:
            _logger.warning(
                "[CaptioningMetric] sacrebleu not installed — BLEU-4 skipped. "
                "Install with: pip install sacrebleu"
            )

        return metrics

    def merge(self, other: "CaptioningMetric") -> None:
        if not isinstance(other, CaptioningMetric):
            raise TypeError(f"Cannot merge {type(other)} into CaptioningMetric")
        self._hypotheses.extend(other._hypotheses)
        self._references.extend(other._references)


# ---------------------------------------------------------------------------
# No-op fallback
# ---------------------------------------------------------------------------

class NoOpMetric(MetricAccumulator):
    """Silent no-op for tasks not yet instrumented with metrics."""

    def update(self, batch_result: BatchResult) -> None:
        pass

    def compute(self) -> Dict[str, float]:
        return {}

    def merge(self, other: "MetricAccumulator") -> None:
        pass


# ---------------------------------------------------------------------------
# Multiprocessing helper
# ---------------------------------------------------------------------------

def worker_update(accumulator: MetricAccumulator, batch_result: BatchResult) -> MetricAccumulator:
    """Run in a worker process: update a fresh copy of the accumulator and return it.

    The caller submits this function to a ``ProcessPoolExecutor``:

        future = pool.submit(worker_update, accumulator, batch_result)

    The returned accumulator contains only the contribution of this one batch.
    The main process merges all returned accumulators at the end.

    Note: ``accumulator`` is copied into the worker via pickle.  To keep the
    copy lightweight, pass an *empty* accumulator (no accumulated data) as
    the template; the worker fills it with one batch and returns it.
    """
    accumulator.update(batch_result)
    return accumulator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    """Convert a binary (H, W) uint8 mask to [x1, y1, x2, y2] or None."""
    import cv2
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return [int(x), int(y), int(x + w), int(y + h)]
