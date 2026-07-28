import gc
import os
from pathlib import Path
import tempfile

import pytest
from PIL import Image
import torch

from tests.support.builders import make_synthetic_image
from tests.support.matrix import (
    DEFAULT_GEMMA3_MODEL,
    MOLMO_1B_SPEC,
    MOLMO_D_SPEC,
    MOLMO_O_SPEC,
    QWEN2_SPEC,
    QWEN25_SPEC,
    QWEN3_MOE_SPEC,
    QWEN3_SPEC,
    resolve_model_ref,
    resolve_sam1_checkpoint,
)
from tests.support.runtime import runtime_enabled


HF_TEST_HOME = "/tmp/training_hf_tests"
HF_TEST_HUB = os.path.join(HF_TEST_HOME, "hub")
HF_TEST_MODULES = os.path.join(HF_TEST_HOME, "modules")
os.makedirs(HF_TEST_HUB, exist_ok=True)
os.makedirs(HF_TEST_MODULES, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_TEST_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_TEST_HUB)
os.environ.setdefault("HF_MODULES_CACHE", HF_TEST_MODULES)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_TEST_HUB)


@pytest.fixture(scope="session")
def qwen2_model_id() -> str | None:
    return resolve_model_ref(QWEN2_SPEC)


@pytest.fixture(scope="session")
def qwen25_model_id() -> str | None:
    return resolve_model_ref(QWEN25_SPEC)


@pytest.fixture(scope="session")
def qwen3_model_id() -> str | None:
    return resolve_model_ref(QWEN3_SPEC)


@pytest.fixture(scope="session")
def qwen3_moe_model_id() -> str | None:
    return resolve_model_ref(QWEN3_MOE_SPEC)


@pytest.fixture(scope="session")
def qwen_snapshot() -> str | None:
    return resolve_model_ref(QWEN25_SPEC)


@pytest.fixture(scope="session")
def sam1_checkpoint() -> str | None:
    return resolve_sam1_checkpoint()


@pytest.fixture(scope="session")
def sam_checkpoint() -> str | None:
    return resolve_sam1_checkpoint()


@pytest.fixture(scope="session")
def molmo_d_model_id() -> str | None:
    return resolve_model_ref(MOLMO_D_SPEC)


@pytest.fixture(scope="session")
def molmo_o_model_id() -> str | None:
    return resolve_model_ref(MOLMO_O_SPEC)


@pytest.fixture(scope="session")
def molmo_1b_model_id() -> str | None:
    return resolve_model_ref(MOLMO_1B_SPEC)


@pytest.fixture(scope="session")
def gemma3_model_id() -> str:
    return os.environ.get("TRAINING_TEST_GEMMA3_MODEL", DEFAULT_GEMMA3_MODEL)


@pytest.fixture(scope="session")
def gemma3_tiny_model_id() -> str:
    return os.environ.get("TRAINING_TEST_GEMMA3_MODEL", DEFAULT_GEMMA3_MODEL)


@pytest.fixture(scope="session", autouse=True)
def huggingface_test_home():
    return HF_TEST_HOME


@pytest.fixture()
def synthetic_image() -> Image.Image:
    return make_synthetic_image()


@pytest.fixture()
def large_artifact_dir() -> Path:
    root = Path(
        os.environ.get(
            "TRAINING_TEST_LARGE_ARTIFACT_ROOT",
            "/fsxvision_new/venkat.kesav/img_2_svg_pretraining/outputs/test_artifacts",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="training_extract_", dir=str(root)))


def _marker_requested(markexpr: str, marker_name: str) -> bool:
    return marker_name in {token.strip("() ") for token in markexpr.replace("and", " ").replace("or", " ").replace("not", " ").split()}


def _external_tests_enabled() -> bool:
    if os.environ.get("TRAINING_TEST_INCLUDE_EXTERNAL", "0") == "1":
        return True
    if os.environ.get("TRAINING_RUN_LARGE_MODEL_TESTS", "0") == "1":
        return True
    if os.environ.get("TRAINING_TEST_SELECTED_VLM_SPECS", ""):
        return True
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    markexpr = config.option.markexpr or ""
    explicit_external = _marker_requested(markexpr, "external")
    explicit_gpu = _marker_requested(markexpr, "gpu")
    explicit_ddp = _marker_requested(markexpr, "ddp")
    explicit_dataparallel = _marker_requested(markexpr, "dataparallel")
    external_enabled = _external_tests_enabled()

    deselected: list[pytest.Item] = []
    kept: list[pytest.Item] = []
    for item in items:
        if item.get_closest_marker("external") and not external_enabled and not explicit_external:
            deselected.append(item)
            continue
        if item.get_closest_marker("gpu") and not runtime_enabled("gpu") and not explicit_gpu:
            deselected.append(item)
            continue
        if item.get_closest_marker("ddp") and not runtime_enabled("ddp") and not explicit_ddp:
            deselected.append(item)
            continue
        if item.get_closest_marker("dataparallel") and not runtime_enabled("dataparallel") and not explicit_dataparallel:
            deselected.append(item)
            continue
        kept.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


@pytest.fixture(autouse=True)
def release_cuda_memory_between_tests():
    yield
    if not torch.cuda.is_available():
        return
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass
