import os
import sys
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _logger.addHandler(handler)
    _logger.propagate = False

if "LOCAL_RANK" in os.environ:
    import torch
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

from img_2_svg_pretraining.training.training_core.data_modules.vlms.common import apply_processor_image_override
from img_2_svg_pretraining.training.training_core.data_modules.qwen.qwen_data import IGNORE_INDEX
from img_2_svg_pretraining.training.training_core.datasets import layout as _layout_datasets  # noqa: F401 — triggers registry
from img_2_svg_pretraining.training.training_core.datasets import pixmo as _pixmo_datasets  # noqa: F401
from img_2_svg_pretraining.training.training_core.datasets import public as _public_datasets  # noqa: F401
from img_2_svg_pretraining.training.training_core.builders.dataset_builder import add_new_tokens, build_single_dataset
from img_2_svg_pretraining.training.training_core.models.vlm_sam import VLMSam
from img_2_svg_pretraining.training.training_core.registry.registry import (
    DataModuleRegistry,
    DatasetRegistry,
    SAMDataModuleRegistry,
)
from img_2_svg_pretraining.training.training_core.registry.utils import DataArguments, DataModule, ModelConfig, SamModelArguments, VLMArguments
from img_2_svg_pretraining.training.training_core.train.custom_trainer import CustomTrainer
from img_2_svg_pretraining.training.training_core.train.train_utils import build_multi_dataset, build_mixed_dataset, build_param_groups
from img_2_svg_pretraining.training.training_core.validation.compute_metrics import ComputeMetrics

# importing for auto-registry side effects
from img_2_svg_pretraining.training.training_core.data_modules.vlms.gemma import gemma_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.data_modules.vlms.molmo import molmo_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.data_modules.vlms.qwen import qwen_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.data_modules.sam.sam1 import sam1_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.data_modules.sam.noop import noop_sam_data  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.vlms.gemma import gemma_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.vlms.molmo import molmo_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.vlms.qwen import qwen_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.models.sam.sam1 import sam1_model  # noqa: F401
from img_2_svg_pretraining.training.training_core.vision_encoders.clip import clip_encoder  # noqa: F401
from img_2_svg_pretraining.training.training_core.vision_encoders.extracted import extracted_encoder  # noqa: F401
from img_2_svg_pretraining.training.training_core.vision_encoders.metaclip import metaclip_encoder  # noqa: F401
from img_2_svg_pretraining.training.training_core.vision_encoders.openvision import openvision_encoder  # noqa: F401
from img_2_svg_pretraining.training.training_core.vision_encoders.siglip import siglip_encoder  # noqa: F401

# promptable_v1 sets HF_DATASETS_OFFLINE=1 at import time as a side effect; undo it so
# streaming datasets (e.g. DocLayNet) can still be fetched from the Hub.
os.environ.pop("HF_DATASETS_OFFLINE", None)

# Patch datasets < 2.3.0 which doesn't support missing 'length' in Sequence
try:
    import datasets.features.features as hf_features
    if hasattr(hf_features, "generate_from_dict"):
        _orig_generate = hf_features.generate_from_dict
        def _patched_generate(obj):
            if isinstance(obj, dict) and obj.get("_type") == "Sequence" and "length" not in obj:
                obj = dict(obj)
                obj["length"] = -1
            return _orig_generate(obj)
        hf_features.generate_from_dict = _patched_generate
except ImportError:
    pass

DEFAULT_CONFIG_PATH = "/code/src/img_2_svg_pretraining/training/configs/archive/run_pramana.yaml"
DEFAULT_WANDB_KEY_FILE = "/code/api_keys/wandb_key"

_PRETRAIN_INIT_MODULES = {
    "molmovlm": "img_2_svg_pretraining.training.training_core.models.vlms.molmo.pretrain_init",
    "qwenvlm":  "img_2_svg_pretraining.training.training_core.models.vlms.qwen.pretrain_init",
    "gemmavlm": "img_2_svg_pretraining.training.training_core.models.vlms.gemma.pretrain_init",
}


def _apply_pretrain_init(vlm_family: str, backbone, pretrain_cfg) -> None:
    module_path = _PRETRAIN_INIT_MODULES.get(vlm_family)
    if module_path is None:
        raise ValueError(
            f"pretrain_init not implemented for vlm_family={vlm_family!r}. "
            f"Supported: {list(_PRETRAIN_INIT_MODULES)}"
        )
    import importlib
    mod = importlib.import_module(module_path)
    _logger.info("[Train] pretrain_init for %s — replacing backbone weights.", vlm_family)
    mod.init_pretrain_weights(backbone, pretrain_cfg)
    _logger.info("[Train] pretrain_init complete.")


def maybe_configure_wandb(cfg: DictConfig):
    report_to = cfg.trainer.get("report_to")
    if not report_to:
        return

    report_targets = [report_to] if isinstance(report_to, str) else list(report_to)
    if "wandb" not in report_targets:
        return

    if not os.environ.get("WANDB_API_KEY"):
        key_path = cfg.get("wandb_key_file", DEFAULT_WANDB_KEY_FILE)
        if os.path.exists(key_path):
            with open(key_path, "r", encoding="utf-8") as handle:
                os.environ["WANDB_API_KEY"] = handle.read().strip()

    if cfg.get("wandb_project") and not os.environ.get("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = cfg.wandb_project

    if cfg.get("wandb_entity") and not os.environ.get("WANDB_ENTITY"):
        os.environ["WANDB_ENTITY"] = cfg.wandb_entity


# Note: build_single_dataset is now in img_2_svg_pretraining.training.training_core.builders.dataset_builder
# and imported at the top of this file.  The local wrapper below passes
# the Molmo max_crops override that is specific to training runs.

def _build_single_dataset_with_molmo(
    dataset_spec: DictConfig | dict, model_name_or_path: str, cfg: DictConfig
):
    """Thin training-time wrapper that injects molmo_max_crops if configured."""
    extra_module_kwargs = {}
    if cfg.vlm_family == "molmovlm" and cfg.get("molmo_max_crops") is not None:
        extra_module_kwargs["molmo_max_crops"] = cfg.molmo_max_crops
    return build_single_dataset(
        dataset_spec=dataset_spec,
        model_name_or_path=model_name_or_path,
        cfg=cfg,
        train=True,
        extra_module_kwargs=extra_module_kwargs,
    )


def _assert_checkpoint_weights_exist(path: str):
    has_bin = os.path.exists(os.path.join(path, "pytorch_model.bin"))
    has_bin_sharded = os.path.exists(os.path.join(path, "pytorch_model.bin.index.json"))
    has_safe = os.path.exists(os.path.join(path, "model.safetensors"))
    has_safe_sharded = os.path.exists(os.path.join(path, "model.safetensors.index.json"))
    if not (has_bin or has_bin_sharded or has_safe or has_safe_sharded):
        msg = f"No standard model weights found at {path}."
        if os.path.exists(os.path.join(path, "zero_to_fp32.py")):
            msg += "\n\n[Detected DeepSpeed Checkpoint] internal files present but no model.safetensors/pytorch_model.bin."
            msg += f"\nPlease run consolidation:\n  cd {path} && python zero_to_fp32.py . pytorch_model.bin"
        raise FileNotFoundError(msg)


def run_training(cfg: DictConfig, debug=False):
    model_name_or_path = cfg.base_model
    train_specs = cfg.datasets.train
    vlm_only = cfg.get("sam_version", "sam1") == "none"

    def get_dataset(dataset_spec: DictConfig | dict):
        return _build_single_dataset_with_molmo(
            dataset_spec=dataset_spec,
            model_name_or_path=model_name_or_path,
            cfg=cfg,
        )

    train_datasets_dict_list, train_weights = build_multi_dataset(train_specs, get_dataset)
    qwen_data_module = train_datasets_dict_list[0]["data_module"]

    vlm_args = VLMArguments(
        family=cfg.vlm_family,
        model_name_or_path=model_name_or_path,
        model_family_name=cfg.model_family_name,
        attn_implementation=cfg.get("attn_implementation"),
        gradient_checkpointing=bool(cfg.trainer.get("gradient_checkpointing", False)),
        tune_mm_llm=bool(cfg.get("tune_mm_llm", not debug)),
        tune_mm_vision=bool(cfg.get("tune_mm_vision", not debug)),
        tune_mm_mlp=bool(cfg.get("tune_mm_mlp", True)),
        tune_mm_lm_head=bool(cfg.get("tune_mm_lm_head", True)),
    )

    if vlm_only:
        config = ModelConfig(
            sam_args=None,
            vlm_args=vlm_args,
            vlm_family=cfg.vlm_family,
            sam_version="none",
            out_dim=cfg.get("out_dim", 256),
            ce_loss_weight=cfg.loss.ce_loss_weight,
            bce_loss_weight=0.0,
            dice_loss_weight=0.0,
        )
    else:
        config = ModelConfig(
            sam_args=SamModelArguments(
                tune_image_encoder=False,
                tune_prompt_encoder=False,
                tune_mask_decoder=True,
                checkpoint=cfg.sam_checkpoint,
                version=cfg.sam_version,
            ),
            vlm_args=vlm_args,
            vlm_family=cfg.vlm_family,
            sam_version=cfg.sam_version,
            out_dim=cfg.get("out_dim", 256),
            bce_loss_weight=cfg.loss.get("bce_loss_weight", 2.0),
            dice_loss_weight=cfg.loss.get("dice_loss_weight", 0.5),
            ce_loss_weight=cfg.loss.ce_loss_weight,
        )

    ckpt_cfg = cfg.get("checkpoint", None)
    load_weights_only = bool(ckpt_cfg and ckpt_cfg.resume and ckpt_cfg.mode == "weights_only")

    if vlm_only:
        from img_2_svg_pretraining.training.training_core.models.vlm_only import VLMOnly

        if load_weights_only:
            assert ckpt_cfg.path, "checkpoint.path must be set when checkpoint.resume is true"
            _logger.info(f"[Checkpoint] Loading weights using from_pretrained from {ckpt_cfg.path}")
            _assert_checkpoint_weights_exist(ckpt_cfg.path)
            model = VLMOnly.from_pretrained(
                ckpt_cfg.path,
                config=config,
                data_module=qwen_data_module,
                ignore_mismatched_sizes=True,
            )
        else:
            model = VLMOnly(config, qwen_data_module)

        _logger.info("[Train] VLM-only mode (sam_version=none) — SAM head not loaded.")

    pretrain_init_cfg = cfg.get("pretrain_init")
    if vlm_only:
        # `model` (VLMOnly) is already built above; pretrain_init is optional
        # here — a plain fine-tune config omits it entirely and keeps the
        # base model's existing connector and LLM weights as-is.
        if pretrain_init_cfg:
            _apply_pretrain_init(cfg.vlm_family, model.backbone, pretrain_init_cfg)
    else:
        if load_weights_only:
            assert ckpt_cfg.path, "checkpoint.path must be set when checkpoint.resume is true"
            _logger.info(f"[Checkpoint] Loading weights using from_pretrained from {ckpt_cfg.path}")
            _assert_checkpoint_weights_exist(ckpt_cfg.path)
            model = VLMSam.from_pretrained(
                ckpt_cfg.path,
                config=config,
                data_module=qwen_data_module,
                ignore_mismatched_sizes=True,
            )
        else:
            model = VLMSam(config, qwen_data_module)

    # Optional vision encoder swap (cfg.vision_encoder + cfg.vision_encoder_checkpoint)
    if cfg.get("vision_encoder"):
        from img_2_svg_pretraining.training.training_core.registry.registry import VisionEncoderRegistry
        from img_2_svg_pretraining.training.training_core.builders.swap_vision_encoder import swap_vision_encoder

        ve_key = cfg.vision_encoder
        ve_ckpt = cfg.get("vision_encoder_checkpoint")
        ve_arch = cfg.get("vision_encoder_arch")

        if ve_key == "extracted":
            if ve_ckpt:
                enc = VisionEncoderRegistry.get_encoder("extracted", checkpoint=ve_ckpt)
            else:
                from img_2_svg_pretraining.training.training_core.builders.extract_vision_encoder import extract_vision_encoder

                ve_src_family = cfg.get("vision_encoder_source_family", cfg.vlm_family)
                ve_src_ckpt = cfg.get("vision_encoder_source_checkpoint", model_name_or_path)
                ve_src_model_family_name = cfg.get(
                    "vision_encoder_source_model_family_name",
                    cfg.model_family_name,
                )
                ve_extract_dir = cfg.get("vision_encoder_extract_dir", "/tmp/ve_extract")
                ve_name = cfg.get("vision_encoder_name", "extracted_ve")
                enc = extract_vision_encoder(
                    vlm_family=ve_src_family,
                    vlm_checkpoint=ve_src_ckpt,
                    encoder_name=ve_name,
                    output_dir=ve_extract_dir,
                    model_family_name=ve_src_model_family_name,
                )
        else:
            enc_kwargs = {"checkpoint": ve_ckpt}
            if ve_arch:
                enc_kwargs["arch"] = ve_arch
            enc = VisionEncoderRegistry.get_encoder(ve_key, **enc_kwargs)

        apply_processor_image_override(qwen_data_module.processor, getattr(enc, "preprocessor_config", None))
        model.backbone, adapter = swap_vision_encoder(model.backbone, enc, cfg.vlm_family)
        if adapter is not None:
            _logger.info(f"[VE Swap] Adapter inserted: {adapter}")
        else:
            _logger.info(f"[VE Swap] No adapter needed (dim match). Encoder: {ve_key}")

    collator = qwen_data_module.Collator
    mixing_cfg = dict(cfg.datasets.get("mixing", {})) or None
    train_dataset = build_mixed_dataset(
        train_datasets_dict_list, train_weights, split="train",
        mixing_cfg=mixing_cfg, seed=cfg.seed,
    )
    val_dataset = build_mixed_dataset(
        train_datasets_dict_list, train_weights, split="val",
        mixing_cfg=mixing_cfg, seed=cfg.seed,
    )

    debug_images_dir = os.path.join(cfg.logging_dir, cfg.run_name)
    os.makedirs(debug_images_dir, exist_ok=True)

    training_args = TrainingArguments(
        **cfg.trainer,
        output_dir=cfg.output_dir,
        logging_dir=debug_images_dir,
        remove_unused_columns=False,
    )

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            _logger.info(f"Trainable: {name} {tuple(parameter.shape)}")

    # Per-layer LR: optional list of {pattern, lr, weight_decay} from config.
    param_groups_cfg = list(cfg.get("param_groups", []))

    if vlm_only:
        # VLM-only uses CustomTrainer too so we get per-layer LR support;
        # mask-prediction metrics are not passed.
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            processing_class=qwen_data_module.processor,
            param_groups_cfg=param_groups_cfg,
        )
    else:
        layout_class_lists = [ds["data_args"].layout_classes for ds in train_datasets_dict_list]
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            processing_class=qwen_data_module.processor,
            param_groups_cfg=param_groups_cfg,
            compute_metrics=ComputeMetrics(
                qwen_data_module.processor.tokenizer,
                IGNORE_INDEX,
                train_datasets_dict_list[0]["data_args"].extract_from_labels_fn,
                sorted(set().union(*layout_class_lists)),
            ),
        )

    # Training efficiency: step time, token throughput, peak GPU memory.
    from img_2_svg_pretraining.training.training_core.train.training_efficiency_callback import TrainingEfficiencyCallback
    eff_cb = TrainingEfficiencyCallback()
    trainer.add_callback(eff_cb)
    trainer._efficiency_cb = eff_cb
    _logger.info("[Train] TrainingEfficiencyCallback registered.")

    # Optional generation-based captioning eval (BLEU-4 + W&B table).
    captioning_cfg = cfg.get("captioning_eval")
    if captioning_cfg:
        from img_2_svg_pretraining.training.training_core.train.captioning_eval_callback import GenerativeEvalCallback
        trainer.add_callback(
            GenerativeEvalCallback(
                val_dataset=val_dataset,
                processor=qwen_data_module.processor,
                cfg=captioning_cfg,
                collate_fn=collator,
            )
        )
        _logger.info("[Train] GenerativeEvalCallback registered.")

    # Optional profiling: torch.profiler trace + NVTX range markers.
    profiling_cfg = cfg.get("profiling")
    if profiling_cfg and profiling_cfg.get("enabled", False):
        from img_2_svg_pretraining.training.training_core.train.profiling_callback import ProfilingCallback
        prof_output = profiling_cfg.get("output_dir") or os.path.join(debug_images_dir, "profiler_traces")
        trainer.add_callback(ProfilingCallback(
            wait_steps=int(profiling_cfg.get("wait_steps", 1)),
            warmup_steps=int(profiling_cfg.get("warmup_steps", 1)),
            active_steps=int(profiling_cfg.get("active_steps", 3)),
            output_dir=prof_output,
            with_stack=bool(profiling_cfg.get("with_stack", False)),
            nvtx=bool(profiling_cfg.get("nvtx", True)),
        ))
        _logger.info("[Train] ProfilingCallback registered (output: %s).", prof_output)

    # Log git commit/branch + upload launch files to W&B at training start.
    report_to = cfg.trainer.get("report_to", "")
    report_targets = [report_to] if isinstance(report_to, str) else list(report_to or [])
    if "wandb" in report_targets:
        from img_2_svg_pretraining.training.training_core.train.captioning_eval_callback import RunMetadataCallback
        config_path = os.environ.get("TRAINING_CONFIG_PATH", DEFAULT_CONFIG_PATH)
        # Derive scripts dir: config is at <repo>/src/configs/, scripts at <repo>/scripts/
        src_dir = os.path.dirname(os.path.dirname(config_path))
        scripts_dir = os.path.join(os.path.dirname(src_dir), "scripts")
        config_stem = os.path.splitext(os.path.basename(config_path))[0]
        launch_script = os.path.join(scripts_dir, f"{config_stem}.sh")
        artifact_paths = [p for p in [config_path, launch_script] if os.path.isfile(p)]
        trainer.add_callback(RunMetadataCallback(artifact_paths=artifact_paths))
        _logger.info("[Train] RunMetadataCallback registered (artifacts: %s).", artifact_paths)

    resume_from = ckpt_cfg.path if (ckpt_cfg and ckpt_cfg.resume and ckpt_cfg.mode == "full") else None
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_state()
    if cfg.get("skip_final_model_save", False):
        _logger.info("[Train] Skipping final model save because skip_final_model_save=true")
    else:
        trainer.save_model(os.path.join(cfg.output_dir, "final"))


if __name__ == "__main__":
    config_path = os.environ.get("TRAINING_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    cfg: DictConfig = OmegaConf.load(config_path)

    debug = False
    if debug:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)
        cfg.trainer.pop("deepspeed")

    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # Support dotlist CLI overrides: trainer.max_steps=200 run_name=foo etc.
    if sys.argv[1:]:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))

    maybe_configure_wandb(cfg)
    _logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    run_training(cfg, debug)
