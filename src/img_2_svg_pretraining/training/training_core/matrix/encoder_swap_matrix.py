from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderSpec:
    slug: str
    vlm_family: str
    model_family_name: str
    checkpoint_env_var: str
    attn_implementation: str | None
    gradient_checkpointing: bool


@dataclass(frozen=True)
class EncoderSpec:
    slug: str
    kind: str
    registry_key: str | None
    checkpoint: str | None = None
    arch: str | None = None
    source_decoder_slug: str | None = None


@dataclass(frozen=True)
class MatrixRunSpec:
    decoder_slug: str
    vlm_family: str
    model_family_name: str
    checkpoint_env_var: str
    attn_implementation: str | None
    gradient_checkpointing: bool
    encoder_slug: str
    encoder_kind: str
    vision_encoder: str | None
    vision_encoder_checkpoint: str | None
    vision_encoder_arch: str | None
    encoder_source_decoder_slug: str | None
    mode: str
    sam_version: str = "sam1"
    sam_checkpoint_env_var: str = "TRAINING_TEST_SAM1_CHECKPOINT"

    @property
    def run_name(self) -> str:
        return f"{self.decoder_slug}__{self.encoder_slug}__{self.mode}"


DECODER_SPECS: tuple[DecoderSpec, ...] = (
    DecoderSpec(
        slug="qwen25",
        vlm_family="qwenvlm",
        model_family_name="qwen2.5vl",
        checkpoint_env_var="TRAINING_TEST_QWEN25_MODEL",
        attn_implementation="flash_attention_2",
        gradient_checkpointing=False,
    ),
    DecoderSpec(
        slug="gemma3",
        vlm_family="gemmavlm",
        model_family_name="gemma3",
        checkpoint_env_var="TRAINING_TEST_GEMMA3_MODEL",
        attn_implementation="eager",
        gradient_checkpointing=True,
    ),
    DecoderSpec(
        slug="molmo7b-o",
        vlm_family="molmovlm",
        model_family_name="molmo",
        checkpoint_env_var="TRAINING_TEST_MOLMO_O_MODEL",
        attn_implementation=None,
        gradient_checkpointing=False,
    ),
    DecoderSpec(
        slug="molmo7b-d",
        vlm_family="molmovlm",
        model_family_name="molmo",
        checkpoint_env_var="TRAINING_TEST_MOLMO_D_MODEL",
        attn_implementation=None,
        gradient_checkpointing=False,
    ),
)

ENCODER_SPECS: tuple[EncoderSpec, ...] = (
    EncoderSpec(slug="native", kind="native", registry_key=None),
    EncoderSpec(
        slug="extracted-qwen25",
        kind="extracted",
        registry_key="extracted",
        source_decoder_slug="qwen25",
    ),
    EncoderSpec(
        slug="extracted-gemma3",
        kind="extracted",
        registry_key="extracted",
        source_decoder_slug="gemma3",
    ),
    EncoderSpec(
        slug="extracted-molmo7bo",
        kind="extracted",
        registry_key="extracted",
        source_decoder_slug="molmo7b-o",
    ),
    EncoderSpec(
        slug="extracted-molmo7bd",
        kind="extracted",
        registry_key="extracted",
        source_decoder_slug="molmo7b-d",
    ),
    EncoderSpec(
        slug="clip",
        kind="standalone",
        registry_key="clip",
        checkpoint="openai/clip-vit-large-patch14-336",
    ),
    EncoderSpec(
        slug="siglip",
        kind="standalone",
        registry_key="siglip",
        checkpoint="google/siglip-so400m-patch14-384",
    ),
    EncoderSpec(
        slug="siglip2",
        kind="standalone",
        registry_key="siglip2",
        checkpoint="google/siglip2-so400m-patch14-384",
    ),
    EncoderSpec(
        slug="metaclip",
        kind="standalone",
        registry_key="metaclip",
        checkpoint="facebook/metaclip-l14-fullcc2.5b",
    ),
    EncoderSpec(
        slug="metaclip2",
        kind="standalone",
        registry_key="metaclip2",
        checkpoint="facebook/metaclip-2-worldwide-huge-quickgelu",
    ),
    EncoderSpec(
        slug="openvision",
        kind="standalone",
        registry_key="openvision",
        checkpoint="UCSC-VLAA/openvision-vit-large-patch14-224",
        arch="ViT-L-14",
    ),
)

MATRIX_MODES: tuple[str, ...] = ("vlm_only", "sam1")

MATRIX_TRAINING_DEFAULTS = {
    "dataset_name": "doclaynet",
    "dataset_kwargs": {
        "data_path": "ds4sd/DocLayNet-v1.2",
        "streaming": True,
        "sample_limit": 8,
    },
    "trainer": {
        "max_steps": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "eval_strategy": "steps",
        "eval_steps": 3,
        "save_strategy": "no",
        "learning_rate": 1e-6,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "constant",
        "logging_steps": 1,
        "bf16": True,
        "dataloader_num_workers": 0,
        "report_to": "none",
    },
    "loss_by_mode": {
        "vlm_only": {
            "ce_loss_weight": 1.0,
            "bce_loss_weight": 0.0,
            "dice_loss_weight": 0.0,
        },
        "sam1": {
            "ce_loss_weight": 1.0,
            "bce_loss_weight": 1.0,
            "dice_loss_weight": 0.5,
        },
    },
}


def expand_encoder_swap_matrix() -> tuple[MatrixRunSpec, ...]:
    runs: list[MatrixRunSpec] = []
    for decoder in DECODER_SPECS:
        for encoder in ENCODER_SPECS:
            if encoder.kind == "extracted" and encoder.source_decoder_slug == decoder.slug:
                continue
            for mode in MATRIX_MODES:
                runs.append(
                    MatrixRunSpec(
                        decoder_slug=decoder.slug,
                        vlm_family=decoder.vlm_family,
                        model_family_name=decoder.model_family_name,
                        checkpoint_env_var=decoder.checkpoint_env_var,
                        attn_implementation=decoder.attn_implementation,
                        gradient_checkpointing=decoder.gradient_checkpointing,
                        encoder_slug=encoder.slug,
                        encoder_kind=encoder.kind,
                        vision_encoder=encoder.registry_key,
                        vision_encoder_checkpoint=encoder.checkpoint,
                        vision_encoder_arch=encoder.arch,
                        encoder_source_decoder_slug=encoder.source_decoder_slug,
                        mode=mode,
                    )
                )
    return tuple(runs)


MATRIX_RUN_SPECS: tuple[MatrixRunSpec, ...] = expand_encoder_swap_matrix()
