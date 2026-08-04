"""Backend registry and factory.

Adding a provider is two steps: write a module implementing `ChatBackend`
(or the vision interfaces), then add one line to `CHAT_BACKENDS`. Backends
are imported lazily inside `make_backend` so an API-only run never imports
torch, and a run without the anthropic package still works with Gemini.

Credentials come from the environment. Each backend type has a default env
var; a config may override with `api_key_env`.
"""
from __future__ import annotations

from pathlib import Path

from .base import BackendError, ChatBackend, PointBackend, RetriableError, SegmentBackend
from .keys import KeyRing, find_key_file, resolve_keys
from .types import GenerationResult, ImagePart, Message, Part, TextPart

__all__ = [
    "BackendError", "ChatBackend", "PointBackend", "SegmentBackend", "RetriableError",
    "GenerationResult", "ImagePart", "Message", "Part", "TextPart",
    "KeyRing", "find_key_file", "resolve_api_key", "resolve_api_keys",
    "make_backend", "CHAT_BACKENDS", "DEFAULT_KEY_ENV",
]

CHAT_BACKENDS = ("openai_compat", "gemini", "anthropic", "hf_local")

DEFAULT_KEY_ENV = {
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai_compat": "OPENAI_API_KEY",
}

# Keys consumed by the factory itself; everything else is passed to the
# backend constructor.
_FACTORY_KEYS = {"type", "api_key_env", "api_key", "api_key_file", "cache_dir"}


def resolve_api_keys(backend_type: str, cfg: dict) -> list[str]:
    """Every usable key for this backend: inline > env var > api_keys.csv."""
    return resolve_keys(backend_type, cfg, DEFAULT_KEY_ENV.get(backend_type))


def resolve_api_key(backend_type: str, cfg: dict) -> str | None:
    keys = resolve_api_keys(backend_type, cfg)
    return keys[0] if keys else None


def make_backend(name: str, cfg: dict, cache_root: Path | None = None) -> ChatBackend:
    """Build a chat backend from its config block.

    `cfg` is the YAML mapping under `backends: <name>:`. Unknown keys raise
    rather than being ignored, so a typo'd option fails at startup instead of
    silently doing nothing.
    """
    backend_type = cfg.get("type")
    if backend_type is None:
        raise ValueError(f"backend '{name}': missing required field 'type'")
    if backend_type not in CHAT_BACKENDS:
        raise ValueError(
            f"backend '{name}': unknown type '{backend_type}'. Known: {sorted(CHAT_BACKENDS)}"
        )

    kwargs = {k: v for k, v in cfg.items() if k not in _FACTORY_KEYS}

    cache_dir = cfg.get("cache_dir")
    if cache_dir is not None:
        kwargs["cache_dir"] = Path(cache_dir)
    elif cache_root is not None:
        kwargs["cache_dir"] = Path(cache_root) / "responses" / name

    if backend_type == "openai_compat":
        from .openai_compat import OpenAICompatBackend
        kwargs["api_key"] = resolve_api_key(backend_type, cfg)
        return OpenAICompatBackend(name=name, **kwargs)

    if backend_type == "gemini":
        from .gemini import GeminiBackend
        # Pass every key: Gemini quotas are per key, so a pool lets a long
        # batch rotate past a rate limit instead of stalling on backoff.
        kwargs["api_keys"] = resolve_api_keys(backend_type, cfg)
        return GeminiBackend(name=name, **kwargs)

    if backend_type == "anthropic":
        from .anthropic import AnthropicBackend
        kwargs["api_key"] = resolve_api_key(backend_type, cfg)
        return AnthropicBackend(name=name, **kwargs)

    if backend_type == "hf_local":
        from .hf_local import HFLocalBackend
        kwargs.setdefault("model", kwargs.get("hf_repo", ""))
        return HFLocalBackend(name=name, **kwargs)

    raise AssertionError(f"unhandled backend type {backend_type}")  # pragma: no cover
