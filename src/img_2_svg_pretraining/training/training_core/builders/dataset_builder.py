"""Shared dataset construction logic for training and inference.

Both ``train/train.py`` and ``inference/val_set_with_generate.py`` need to
build datasets with identical seed, split_ratio, and localization_mode
settings.  This module provides a single canonical implementation that both
can import.

Functions
---------
build_single_dataset
    Build one dataset bundle from a dataset spec dict and a global config.
    Returns ``{"data_args": ..., "data_module": ..., "train_dataset": ...,
    "val_dataset": ...}``.

add_new_tokens
    Register ``[SEG]`` with the tokenizer and return the updated tokenizer
    + seg token index.  Same signature as the local helper in ``train.py``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from omegaconf import DictConfig

from img_2_svg_pretraining.training.training_core.datasets.layout.utils import split_train_eval
from img_2_svg_pretraining.training.training_core.registry.registry import DataModuleRegistry, DatasetRegistry, SAMDataModuleRegistry
from img_2_svg_pretraining.training.training_core.registry.utils import DataArguments, DataModule


def add_new_tokens(tokenizer):
    """Add the [SEG] token and return (tokenizer, seg_token_idx)."""
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    return tokenizer, seg_token_idx


def build_single_dataset(
    dataset_spec: DictConfig | dict,
    model_name_or_path: str,
    cfg: DictConfig,
    train: bool = True,
    extra_module_kwargs: Optional[Dict[str, Any]] = None,
    change_tokenizer_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Build one dataset bundle from a dataset spec and global config.

    This is the canonical, shared implementation used by both the training
    script and the inference / validation scripts.  It applies the same
    ``localization_mode`` injection (``"sam"`` vs ``"autoregressive"``) and
    ``split_train_eval`` logic that the training path uses, ensuring that
    the validation split seen during inference is **identical** to the one
    held out during training.

    Args:
        dataset_spec:         Dict or DictConfig with ``name`` and optional
                              ``dataset_kwargs`` / ``weight`` keys.
        model_name_or_path:   Base VLM model path.
        cfg:                  Top-level OmegaConf config.  Expected keys:
                              ``vlm_family``, ``model_family_name``,
                              ``sam_version``, ``seed``,
                              ``sam_image_size`` (optional), and any
                              VLM-family-specific overrides.
        train:                If ``False`` the dataset is built for inference
                              (no train split needed; ``train_dataset`` will
                              be ``None``).
        extra_module_kwargs:  Additional kwargs forwarded to
                              ``DataModuleRegistry.get_module()`` (e.g.
                              ``molmo_max_crops``).
        change_tokenizer_fn:  Override the default ``add_new_tokens`` helper.
                              Pass ``None`` to use the module default.

    Returns:
        Dict with keys:
        - ``data_args``      — :class:`DataArguments`
        - ``data_module``    — :class:`DataModule`
        - ``train_dataset``  — dataset split for training (or ``None`` if
                              ``train=False``)
        - ``val_dataset``    — dataset split for validation / inference
    """
    if isinstance(dataset_spec, dict):
        name = dataset_spec["name"]
        dataset_kwargs = dict(dataset_spec.get("dataset_kwargs", {}))
    else:
        name = dataset_spec.name
        dataset_kwargs = dict(getattr(dataset_spec, "dataset_kwargs", {}) or {})

    _change_tokenizer_fn = change_tokenizer_fn or add_new_tokens
    _extra = extra_module_kwargs or {}

    sam_data_module = SAMDataModuleRegistry.get_sam_data_module(cfg.sam_version)

    data_args: DataArguments = DatasetRegistry.get_dataset(
        name,
        get_sam_source_fn=sam_data_module.format_source,
        debug_path="",
        seed=cfg.seed,
        sam_image_size=cfg.get("sam_image_size", 1024),
        **dataset_kwargs,
    )

    # Inject localization_mode so dataset adapters can branch between
    # SAM-supervised ("[SEG]" tokens) and autoregressive (coordinate text).
    localization_mode = "autoregressive" if cfg.get("sam_version", "sam1") == "none" else "sam"
    data_args.get_source_kwargs["localization_mode"] = localization_mode

    data_module: DataModule = DataModuleRegistry.get_module(
        cfg.vlm_family,
        data_args=data_args,
        change_tokenizer_fn=_change_tokenizer_fn,
        sam_collator=sam_data_module.get_collator(),
        model_name=cfg.model_family_name,
        model_path=model_name_or_path,
        **_extra,
    )

    split_ratio = cfg.get("split_ratio", 0.99)
    train_dataset, val_dataset = split_train_eval(data_module.Dataloader, split_ratio, cfg.seed)

    return {
        "data_args": data_args,
        "data_module": data_module,
        "train_dataset": train_dataset if train else None,
        "val_dataset": val_dataset,
    }
