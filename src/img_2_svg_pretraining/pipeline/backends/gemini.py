"""Google Gemini chat backend (google-genai SDK).

Gemini differs from the OpenAI shape in two ways this adapter absorbs:
system instructions are a top-level config field rather than a message with
role="system", and images are inline `Part` blobs rather than data URLs.
"""
from __future__ import annotations

import time

from .base import ChatBackend, RetriableError
from .keys import KeyRing
from .types import GenerationResult, ImagePart, Message, TextPart, VideoPart

_RETRIABLE_MARKERS = ("429", "500", "502", "503", "504",
                      "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED",
                      "401", "UNAUTHENTICATED", "403", "PERMISSION_DENIED")

# Errors that condemn the *key*, not the request. A disabled service account
# or a project denied access answers identically however many times it is
# asked, so the only useful response is to move to another key -- exactly what
# a quota error already does. These were missing, and the cost was not
# theoretical: with roughly a third of the pool dead, a run would reach a bad
# key, raise a non-retriable error without ever rotating, and lose every
# remaining call. An animation eval died at step 10 of 17 that way.
_DEAD_KEY_MARKERS = ("401", "UNAUTHENTICATED", "403", "PERMISSION_DENIED")
_QUOTA_MARKERS = ("429", "RESOURCE_EXHAUSTED")


class GeminiBackend(ChatBackend):
    def __init__(self, name: str, model: str, api_key: str | None = None,
                 api_keys: list[str] | None = None, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai package required for gemini backend: pip install google-genai"
            ) from e

        keys = list(api_keys) if api_keys else ([api_key] if api_key else [])
        if not keys:
            raise ValueError(
                f"backend '{name}': no API key. Set GOOGLE_API_KEY, api_key_env in the "
                f"config, or add a Google row to api_keys.csv."
            )

        self._genai = genai
        self._ring = KeyRing(keys)
        self._clients = {}
        self._client = self._client_for(self._ring.current)

        # With a key pool, the retry budget must be able to walk the whole
        # pool: quotas are per key, so the first N-1 attempts may each be a
        # different exhausted key rather than a transient fault.
        if len(self._ring) > 1:
            self.max_retries = max(self.max_retries, len(self._ring) + 1)

    def _client_for(self, key: str):
        """One client per key, built once and reused."""
        if key not in self._clients:
            self._clients[key] = self._genai.Client(api_key=key)
        return self._clients[key]

    def _rotate_key(self) -> None:
        """Move to the next key after a rate-limit rejection.

        With one key this is a no-op beyond the backoff the base class already
        applies; with several it turns a per-key quota into a pooled one.
        """
        if len(self._ring) > 1:
            self._client = self._client_for(self._ring.rotate())

    def _to_gemini(self, messages: list[Message]) -> tuple[list, str | None]:
        """Returns (contents, system_instruction)."""
        from google.genai import types as gtypes

        system_text: list[str] = []
        contents = []
        for m in messages:
            if m.role == "system":
                system_text.append(m.text())
                continue
            parts = []
            for part in m.content:
                if isinstance(part, TextPart):
                    parts.append(gtypes.Part.from_text(text=part.text))
                elif isinstance(part, ImagePart):
                    parts.append(gtypes.Part.from_bytes(
                        data=part.read_bytes(), mime_type=part.media_type()))
                elif isinstance(part, VideoPart):
                    # Inline bytes, not the Files API: every mp4 in this bench
                    # is well under the ~20MB inline request cap, and inline
                    # needs no upload/poll/delete lifecycle.
                    parts.append(gtypes.Part.from_bytes(
                        data=part.read_bytes(), mime_type=part.media_type()))
            # Gemini names the assistant role "model".
            role = "model" if m.role == "assistant" else "user"
            contents.append(gtypes.Content(role=role, parts=parts))
        return contents, ("\n".join(system_text) if system_text else None)

    def _generate(self, messages: list[Message], **params) -> GenerationResult:
        from google.genai import types as gtypes

        started = time.time()
        contents, system_instruction = self._to_gemini(messages)

        cfg_kwargs = {}
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        if "temperature" in params:
            cfg_kwargs["temperature"] = params["temperature"]
        if "top_p" in params:
            cfg_kwargs["top_p"] = params["top_p"]
        if "max_tokens" in params:
            cfg_kwargs["max_output_tokens"] = params["max_tokens"]
        if params.get("media_resolution"):
            # Video frames are tokenized far more coarsely than still images:
            # measured at 60 tokens/frame by default against ~700 for the same
            # diagram sent as an image. For a judge asked about text legibility
            # and geometric alignment that is not a cost setting, it is whether
            # the model can see the labels at all. Opt-in, since it multiplies
            # the token cost of every frame.
            cfg_kwargs["media_resolution"] = params["media_resolution"]

        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=gtypes.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None,
            )
        except Exception as e:
            message = str(e)
            if any(marker in message for marker in _RETRIABLE_MARKERS):
                # Quotas are per key, so move to the next one before the base
                # class retries -- otherwise every retry hits the same limit.
                error = RetriableError(message)
                if any(m in message for m in _QUOTA_MARKERS + _DEAD_KEY_MARKERS):
                    # Quotas are per key, and so is a disabled account, so move
                    # to the next one before the base class retries -- otherwise
                    # every retry hits the same limit or the same dead key.
                    self._ring.mark_exhausted()
                    self._rotate_key()
                    # Tell the retry loop not to back off: a different key is
                    # now active, so the next attempt should go straight out.
                    error.rotated = len(self._ring) > 1
                raise error from e
            raise

        usage = {}
        if getattr(resp, "usage_metadata", None):
            usage = {
                "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0),
            }
        return GenerationResult(
            text=resp.text or "",
            raw=resp,
            model=self.model,
            latency_s=time.time() - started,
            usage=usage,
        )
