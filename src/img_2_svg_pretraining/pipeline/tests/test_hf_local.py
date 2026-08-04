"""hf_local backend wiring, without loading a model.

These cover the seams that broke in practice: the attribute the loaded module
is stored under, and the attention settings Gemma 4 needs in this container.
"""
from __future__ import annotations

import inspect

from img_2_svg_pretraining.pipeline.backends.hf_local import HFLocalBackend


def _params():
    return inspect.signature(HFLocalBackend.__init__).parameters


def test_loaded_module_is_not_stored_as_self_model():
    """Regression: `ChatBackend.__init__` sets `self.model` to the model *name*,
    which `_fingerprint` hashes for the response cache. Storing the nn.Module
    there too shadowed the string, and every request died with
    `'Gemma4UnifiedForConditionalGeneration' object has no attribute 'encode'`.
    """
    src = inspect.getsource(HFLocalBackend)
    assert "self.model = " not in src, "self.model would shadow the base class's model name"
    assert "self.hf_model = " in src


def test_generation_uses_the_renamed_attribute():
    src = inspect.getsource(HFLocalBackend)
    assert "self.hf_model.generate(" in src
    assert "self.model.generate(" not in src


def test_attention_defaults_avoid_flash_attn_and_cudnn():
    """Neither flash_attention_2 nor sdpa works here: the container's
    flash_attn is built against a different torch, and sdpa routes into cuDNN,
    which has no execution plan for Gemma 4's head shapes."""
    p = _params()
    assert p["attn_implementation"].default == "eager"
    assert p["cudnn"].default is False


def test_weights_stream_to_device_rather_than_transfer_after_load():
    """`.to(device)` after loading OOM'd an idle 80GB H100 on gemma-4-31B."""
    src = inspect.getsource(HFLocalBackend)
    assert "device_map=device" in src
    assert ").to(device).eval()" not in src


def test_local_generation_is_serialised_by_default():
    """One GPU: concurrent calls would only contend for the same device."""
    src = inspect.getsource(HFLocalBackend.__init__)
    assert 'kwargs.setdefault("max_concurrency", 1)' in src
