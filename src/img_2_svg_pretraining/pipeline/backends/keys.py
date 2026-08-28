"""API key discovery, from a CSV key file or the environment.

The project keeps keys in a gitignored CSV at the repo root with a
`Key, Service` header, several rows per service. Holding more than one key
matters in practice: Gemini's free tier rate-limits per key, and a batch of
samples will exhaust one long before the batch finishes.

`KeyRing` therefore hands out keys round-robin and can rotate to the next one
when a call is rejected as rate-limited, which is why keys are resolved to a
list rather than a single string.
"""
from __future__ import annotations

import csv
import os
import threading
from pathlib import Path

# Service label in the CSV -> backend type in the config.
SERVICE_ALIASES = {
    "google": "gemini",
    "gemini": "gemini",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai_compat",
}

DEFAULT_KEY_FILES = ("api_keys.csv", "api_keys/api_keys.csv")


def _repo_root() -> Path:
    # backends/keys.py -> backends -> pipeline -> img_2_svg_pretraining -> src -> repo
    return Path(__file__).resolve().parents[4]


def find_key_file(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    env_path = os.environ.get("PIPELINE_KEY_FILE")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    root = _repo_root()
    for candidate in DEFAULT_KEY_FILES:
        path = root / candidate
        if path.exists():
            return path
    return None


def load_keys_from_csv(path: Path, backend_type: str) -> list[str]:
    """Read every key registered for this backend type.

    Tolerates the header's spacing (`Key, Service`) and rows whose columns
    carry stray whitespace, since the file is hand-maintained.
    """
    keys: list[str] = []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                normalized = {(k or "").strip().lower(): (v or "").strip()
                              for k, v in row.items()}
                key, service = normalized.get("key"), normalized.get("service", "")
                if not key:
                    continue
                if SERVICE_ALIASES.get(service.lower()) == backend_type:
                    keys.append(key)
    except (OSError, csv.Error):
        return []
    # Preserve order but drop duplicates.
    return list(dict.fromkeys(keys))


def _split_keys(value: str) -> list[str]:
    """One key, or several comma-separated ones.

    A pool given this way rotates through `KeyRing` exactly like a CSV pool, so
    a key that hits its spend cap is marked exhausted and the next one takes
    over mid-run instead of the run dying. That is the whole reason this accepts
    a list: OpenRouter keys carry a hard credit limit, and a sweep that outlives
    one key should not have to be restarted to reach the second.

    No provider this repo talks to issues keys containing a comma (OpenRouter
    `sk-or-v1-<hex>`, Google `AIza<base64url>`), so splitting is unambiguous.
    Blanks are dropped, which makes a trailing comma harmless.
    """
    return [part.strip() for part in str(value).split(",") if part.strip()]


def resolve_keys(backend_type: str, cfg: dict, default_env: str | None = None) -> list[str]:
    """All usable keys for a backend, most explicit source first.

    Order: an inline `api_key` in the config, then the env var, then the CSV.
    An explicit setting should always win over a file the user may have
    forgotten about.
    """
    inline = cfg.get("api_key")
    if inline:
        return _split_keys(inline)

    env_var = cfg.get("api_key_env") or default_env
    from_env = os.environ.get(env_var) if env_var else None
    if from_env:
        return _split_keys(from_env)

    key_file = find_key_file(cfg.get("api_key_file"))
    if key_file:
        return load_keys_from_csv(key_file, backend_type)
    return []


class KeyRing:
    """Thread-safe rotation over a pool of interchangeable API keys.

    Keys that return a rate-limit error are marked exhausted and skipped for
    the rest of the run. Without this the ring keeps handing back dead keys,
    and since each call burns its retry budget before rotating, a pool with a
    few live keys behaves almost as badly as a pool with none -- observed
    with 3 live keys out of 11, where quota errors still dominated because
    the first key tried was always a dead one.
    """

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("KeyRing needs at least one key")
        self._keys = list(keys)
        self._lock = threading.Lock()
        self._index = 0
        self._exhausted: set[int] = set()

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def live(self) -> int:
        """Keys not yet known to be rate-limited."""
        return len(self._keys) - len(self._exhausted)

    @property
    def current(self) -> str:
        with self._lock:
            return self._keys[self._index]

    def mark_exhausted(self) -> None:
        """Record that the active key is rate-limited, so it is skipped."""
        with self._lock:
            self._exhausted.add(self._index)
            # All dead: clear and start over rather than deadlock. Quotas do
            # reset, and a stale mark should not permanently disable a key.
            if len(self._exhausted) >= len(self._keys):
                self._exhausted.clear()

    def rotate(self) -> str:
        """Advance to the next key that is not known to be exhausted."""
        with self._lock:
            for step in range(1, len(self._keys) + 1):
                candidate = (self._index + step) % len(self._keys)
                if candidate not in self._exhausted:
                    self._index = candidate
                    return self._keys[candidate]
            # Every key is marked; fall back to simple advance.
            self._index = (self._index + 1) % len(self._keys)
            return self._keys[self._index]

    def masked(self) -> str:
        """Identify the active key in logs without exposing it."""
        key = self.current
        return f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "***"
