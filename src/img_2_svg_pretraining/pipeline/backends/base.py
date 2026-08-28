"""Backend interfaces and the cross-cutting behavior shared by all of them.

Two interfaces, deliberately separate:

- `ChatBackend`  -- messages in, text out. Used by every LLM/VLM agent.
- `PointBackend` / `SegmentBackend` -- image in, structure out. Used only by
  the raster integrator (Molmo2 / SAM3). These are not chat models; Molmo2
  emits `<points coords=.../>` and SAM3 consumes point prompts and returns
  masks, so forcing them behind ChatBackend would misrepresent them.

Retry, concurrency limiting and response caching live here rather than in
each provider so adding a backend stays a small job: implement `_generate`
and the shared behavior comes with it.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .types import GenerationResult, ImagePart, Message, TextPart, VideoPart


class BackendError(Exception):
    pass


class RetriableError(BackendError):
    """Transient failure (429, 5xx, timeout) -- worth retrying."""


def _fingerprint(model: str, messages: list[Message], params: dict) -> str:
    """Stable hash of a request, for the response cache.

    Images and videos contribute their bytes' hash, not their path, so moving
    the data root doesn't invalidate the cache and two samples with identical
    media share an entry.

    EVERY part must contribute something. A part type that falls through this
    loop is invisible to the cache, so two requests differing only in that part
    collide and the second is served the first one's answer -- silently, with a
    plausible result. That is not hypothetical for video: the TikZ and SVG bench
    configs share a dataset root, so the same (sample, style) cell sends an
    identical prompt and an identical source image in both, differing only in
    the mp4. Hence the explicit `raise` rather than an `else: pass`.
    """
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    for m in messages:
        h.update(m.role.encode())
        for part in m.content:
            if isinstance(part, TextPart):
                h.update(b"t")
                h.update(part.text.encode())
            elif isinstance(part, ImagePart):
                h.update(b"i")
                h.update(hashlib.sha256(part.read_bytes()).digest())
            elif isinstance(part, VideoPart):
                h.update(b"v")
                h.update(hashlib.sha256(part.read_bytes()).digest())
            else:
                raise TypeError(
                    f"_fingerprint cannot hash {type(part).__name__}; add a "
                    "branch for it or the response cache will collide")
    return h.hexdigest()[:32]


class ChatBackend(ABC):
    """Messages in, text out.

    Subclasses implement `_generate`; everything public routes through
    `generate`, which adds caching, retries and concurrency control.
    """

    def __init__(
        self,
        name: str,
        model: str,
        max_concurrency: int = 4,
        max_retries: int = 4,
        timeout: float = 600.0,
        cache_dir: Path | None = None,
        default_params: dict | None = None,
    ):
        self.name = name
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.default_params = default_params or {}
        self._sem = threading.Semaphore(max_concurrency)
        self._max_concurrency = max_concurrency

    # -- to implement ----------------------------------------------------

    @abstractmethod
    def _generate(self, messages: list[Message], **params) -> GenerationResult:
        """One real call to the provider. Raise RetriableError for transient
        failures; any other exception is treated as terminal."""

    def unload(self) -> None:
        """Release resources. No-op for API backends; local backends free GPU."""

    # -- public ----------------------------------------------------------

    def generate(self, messages: list[Message], **params) -> GenerationResult:
        merged = {**self.default_params, **params}

        cached = self._cache_get(messages, merged)
        if cached is not None:
            return cached

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with self._sem:
                    started = time.time()
                    result = self._generate(messages, **merged)
                result.latency_s = result.latency_s or (time.time() - started)
                result.model = result.model or self.model
                self._cache_put(messages, merged, result)
                return result
            except RetriableError as e:
                last_err = e
                if attempt == self.max_retries - 1:
                    break
                # A key-exhausted failure is not transient for *this* key, so
                # backing off achieves nothing -- the backend has already
                # rotated to a different key and the retry should go out
                # immediately. Sleeping here is what made a pool with live
                # keys still fail: the retry budget drained on waits.
                if getattr(e, "rotated", False):
                    continue
                # Exponential backoff with jitter; jitter matters because
                # generate_batch fans out concurrently and un-jittered retries
                # would re-collide in lockstep.
                delay = min(2 ** attempt, 30) * (0.5 + random.random())
                time.sleep(delay)
            except Exception as e:  # terminal
                return GenerationResult(text="", model=self.model, error=f"{type(e).__name__}: {e}")

        return GenerationResult(
            text="", model=self.model,
            error=f"exhausted {self.max_retries} retries: {last_err}",
        )

    def generate_batch(self, batch: list[list[Message]], **params) -> list[GenerationResult]:
        """Default: thread-pool over `generate`, which suits API backends.

        `hf_local` overrides this with a true batched `.generate()` call.
        Order is preserved; failures come back as error results, not raises.
        """
        if not batch:
            return []
        if self._max_concurrency <= 1:
            return [self.generate(m, **params) for m in batch]
        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            return list(pool.map(lambda m: self.generate(m, **params), batch))

    # -- cache -----------------------------------------------------------

    def _cache_path(self, messages: list[Message], params: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{_fingerprint(self.model, messages, params)}.json"

    def _cache_get(self, messages: list[Message], params: dict) -> GenerationResult | None:
        path = self._cache_path(messages, params)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return GenerationResult(
            text=payload["text"], model=payload.get("model", self.model),
            latency_s=payload.get("latency_s", 0.0), usage=payload.get("usage", {}),
            from_cache=True,
        )

    def _cache_put(self, messages: list[Message], params: dict, result: GenerationResult) -> None:
        path = self._cache_path(messages, params)
        if path is None or not result.ok:
            return
        # AN EMPTY RESPONSE MUST NOT BECOME A PERMANENT ANSWER.
        #
        # A provider can return HTTP 200 with no content, or with content cut
        # off mid-token, and report nonsense usage alongside it -- observed on
        # OpenRouter as `completion_tokens: 1` beside 11KB of half-written SVG,
        # and again as `completion_tokens: 1` beside an empty string. Caching
        # that turns one transient provider failure into a permanent one: every
        # retry is served the same broken text, the stage fails identically, and
        # it looks exactly like the model being incapable of the task.
        #
        # Cost of being wrong in each direction is asymmetric. Refusing to cache
        # a genuinely-empty answer costs one repeated call; caching a truncated
        # one costs the artifact, silently, for as long as the entry lives.
        if not (result.text or "").strip():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "text": result.text, "model": result.model,
            "latency_s": result.latency_s, "usage": result.usage,
        }))
        tmp.replace(path)  # atomic, so a killed run can't leave a half-written entry


# -- vision backends -----------------------------------------------------


class PointBackend(ABC):
    """Image + query -> points. Molmo2 / MolmoPoint."""

    @abstractmethod
    def point(self, image_path: Path, query: str) -> list[tuple[float, float]]:
        """Return points in pixel coordinates of the source image."""

    def unload(self) -> None:
        pass


class SegmentBackend(ABC):
    """Image + point prompts -> masks. SAM3."""

    @abstractmethod
    def segment(self, image_path: Path, points: list[tuple[float, float]]) -> list[dict]:
        """Return one dict per mask with at least `bbox` (x0, y0, x1, y1)."""

    def unload(self) -> None:
        pass
