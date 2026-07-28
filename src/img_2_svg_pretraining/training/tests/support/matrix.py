from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VLMCheckpointSpec:
    slug: str
    family: str
    model_family_name: str
    env_var: str
    default_ref: str | None = None
    require_existing_path: bool = False
    representative: bool = False


DEFAULT_QWEN25_CACHE = (
    "/fsxvision_new/venkat.kesav/backup/hf_cache/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
DEFAULT_SAM1_CHECKPOINT = (
    "/fsxvision_new/srihari.bandarupalli/DocGrounding/checkpoints/"
    "sam_vit_h_4b8939.pth"
)
DEFAULT_GEMMA3_MODEL = "hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration"


QWEN2_SPEC = VLMCheckpointSpec(
    slug="qwen2vl",
    family="qwenvlm",
    model_family_name="qwen2vl",
    env_var="TRAINING_TEST_QWEN2_MODEL",
)
QWEN25_SPEC = VLMCheckpointSpec(
    slug="qwen2.5vl",
    family="qwenvlm",
    model_family_name="qwen2.5vl",
    env_var="TRAINING_TEST_QWEN25_MODEL",
    default_ref=DEFAULT_QWEN25_CACHE,
    require_existing_path=True,
    representative=True,
)
QWEN3_SPEC = VLMCheckpointSpec(
    slug="qwen3vl",
    family="qwenvlm",
    model_family_name="qwen3vl",
    env_var="TRAINING_TEST_QWEN3_MODEL",
)
QWEN3_MOE_SPEC = VLMCheckpointSpec(
    slug="qwen3vl-moe",
    family="qwenvlm",
    model_family_name="qwen3vl",
    env_var="TRAINING_TEST_QWEN3_MOE_MODEL",
)
GEMMA3_SPEC = VLMCheckpointSpec(
    slug="gemma3",
    family="gemmavlm",
    model_family_name="gemma3",
    env_var="TRAINING_TEST_GEMMA3_MODEL",
    default_ref=DEFAULT_GEMMA3_MODEL,
    representative=True,
)

MOLMO_D_SPEC = VLMCheckpointSpec(
    slug="molmo7b-d",
    family="molmovlm",
    model_family_name="molmo",
    env_var="TRAINING_TEST_MOLMO_D_MODEL",
    require_existing_path=True,
    representative=True,
)
MOLMO_O_SPEC = VLMCheckpointSpec(
    slug="molmo7b-o",
    family="molmovlm",
    model_family_name="molmo",
    env_var="TRAINING_TEST_MOLMO_O_MODEL",
    require_existing_path=True,
)
MOLMO_1B_SPEC = VLMCheckpointSpec(
    slug="molmoe-1b",
    family="molmovlm",
    model_family_name="molmo",
    env_var="TRAINING_TEST_MOLMO_1B_MODEL",
    require_existing_path=True,
)

SUPPORTED_MOLMO_SPECS = (MOLMO_D_SPEC, MOLMO_O_SPEC, MOLMO_1B_SPEC)
SUPPORTED_QWEN_SPECS = (QWEN2_SPEC, QWEN25_SPEC, QWEN3_SPEC, QWEN3_MOE_SPEC)
SUPPORTED_VLM_SPECS = SUPPORTED_QWEN_SPECS + (GEMMA3_SPEC,) + SUPPORTED_MOLMO_SPECS
REPRESENTATIVE_VLM_SPECS = tuple(spec for spec in SUPPORTED_VLM_SPECS if spec.representative)
REPRESENTATIVE_TRAINER_SPECS = (QWEN25_SPEC, GEMMA3_SPEC, MOLMO_D_SPEC)
SPEC_BY_SLUG = {spec.slug: spec for spec in SUPPORTED_VLM_SPECS}


def selected_molmo_specs() -> tuple[VLMCheckpointSpec, ...]:
    return selected_specs(SUPPORTED_MOLMO_SPECS)


def _selected_slugs() -> set[str]:
    raw = os.environ.get("TRAINING_TEST_SELECTED_VLM_SPECS", "")
    return {value.strip() for value in raw.split(",") if value.strip()}


def selected_specs(default_specs: tuple[VLMCheckpointSpec, ...]) -> tuple[VLMCheckpointSpec, ...]:
    selected_slugs = _selected_slugs()
    if not selected_slugs:
        return default_specs
    filtered = tuple(spec for spec in default_specs if spec.slug in selected_slugs)
    return filtered or default_specs


def selected_qwen_specs() -> tuple[VLMCheckpointSpec, ...]:
    return selected_specs(SUPPORTED_QWEN_SPECS)


def selected_representative_trainer_specs() -> tuple[VLMCheckpointSpec, ...]:
    return selected_specs(REPRESENTATIVE_TRAINER_SPECS)


def resolve_optional_ref(raw_value: str | None, require_existing_path: bool = False) -> str | None:
    if not raw_value:
        return None

    path = Path(raw_value)
    if require_existing_path or raw_value.startswith("/") or raw_value.startswith("."):
        return str(path) if path.exists() else None
    return raw_value


def resolve_model_ref(spec: VLMCheckpointSpec) -> str | None:
    return resolve_optional_ref(
        os.environ.get(spec.env_var, spec.default_ref),
        require_existing_path=spec.require_existing_path,
    )


def resolve_sam1_checkpoint() -> str | None:
    return resolve_optional_ref(
        os.environ.get("TRAINING_TEST_SAM1_CHECKPOINT", DEFAULT_SAM1_CHECKPOINT),
        require_existing_path=True,
    )


def get_spec(slug: str) -> VLMCheckpointSpec:
    return SPEC_BY_SLUG[slug]
