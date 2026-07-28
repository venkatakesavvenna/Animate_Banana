from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset
from transformers import PretrainedConfig


@dataclass
class AnnotationSpec:
    """Describes the spatial annotation type and label set for a dataset.

    VLM data modules use this to branch prompt formatting.
    ComputeMetrics uses it to decide which metrics to compute.

    Canonical coordinate representation throughout img_2_svg_pretraining training:
      - bounding boxes: (x1, y1, x2, y2) in [0, 1] float, relative to image dims
      - points: (x, y) in [0, 1] float, relative to image dims
    See img_2_svg_pretraining.training.training_core.datasets.normalization for converters.
    """
    has_localization: bool = False
    # "none" | "bbox" | "point" | "bbox+mask" | "point+mask"
    localization_type: str = "none"
    # Class names for classification-type localization (layout, detection).
    # None means the dataset has no discrete label taxonomy.
    label_classes: Optional[List[str]] = None
    # Parses VLM output text → structured predictions for metric computation.
    # Signature: (decoded_str: str) -> List[Any]
    # None disables structured metrics for this dataset.
    parse_predictions_fn: Optional[Callable] = None


@dataclass
class DataModule:
    model_name: str
    model_path: str
    processor: Any
    seg_token_idx: int
    ignore_idx: int
    Dataloader: type
    Collator: type
    layout_classes: List[str]
    family_name: str = ""
    tokenizer_vocab_size: Optional[int] = None


@dataclass
class DataArguments:
    dataset_path: str
    ds: Dataset
    seed: int
    get_source: Callable
    get_source_kwargs: Dict
    annotation_spec: AnnotationSpec = field(default_factory=AnnotationSpec)
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)

    # ------------------------------------------------------------------
    # Backward-compatibility shims — existing layout datasets pass these
    # positional / keyword args and they still work unchanged.
    # ------------------------------------------------------------------
    extract_from_labels_fn: Optional[Callable] = field(default=None)
    layout_classes: Optional[List[str]] = field(default=None)

    def __post_init__(self):
        # Promote legacy fields into annotation_spec when callers still
        # pass the old positional args (all existing layout datasets).
        if self.extract_from_labels_fn is not None and self.annotation_spec.parse_predictions_fn is None:
            self.annotation_spec.parse_predictions_fn = self.extract_from_labels_fn
        if self.layout_classes is not None and self.annotation_spec.label_classes is None:
            self.annotation_spec.label_classes = self.layout_classes
        if self.layout_classes is not None and not self.annotation_spec.has_localization:
            self.annotation_spec.has_localization = True
            if self.annotation_spec.localization_type == "none":
                self.annotation_spec.localization_type = "bbox"


@dataclass
class SamModelArguments:
    tune_image_encoder: bool
    tune_prompt_encoder: bool
    tune_mask_decoder: bool
    checkpoint: str
    version: str = "sam1"


@dataclass
class VLMArguments:
    family: str = field(default="qwenvlm")
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_family_name: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default=None)
    gradient_checkpointing: bool = field(default=False)
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_lm_head: bool = field(default=True)


class ModelConfig(PretrainedConfig):
    model_type = "vlm_sam"

    def __init__(
        self,
        sam_args: SamModelArguments | dict | None = None,
        vlm_args: VLMArguments | dict | None = None,
        vlm_family: str = "qwenvlm",
        sam_version: str = "sam1",
        out_dim=256,
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        model_max_length=8192,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.attn_implementation = "flash_attention_2"
        self.vlm_family = vlm_family
        self.sam_version = sam_version

        try:
            self.sam_args = self._serialize_dataclass(sam_args)
            self.vlm_args = self._serialize_dataclass(vlm_args)
        except Exception:
            self.sam_args = {}
            self.vlm_args = {}

        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.model_max_length = model_max_length
        self.out_dim = out_dim

    @staticmethod
    def _serialize_dataclass(value: Any) -> Any:
        if value is None:
            return None
        if is_dataclass(value):
            return asdict(value)
        return value
