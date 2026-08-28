"""OpenAI-compatible chat backend.

Covers any OpenAI-compatible provider -- OpenAI itself, and the hosted APIs
several open-weight vendors expose (Moonshot, MiniMax, DeepSeek), which is how
to reach models too large to run locally.

For local open weights use `hf_local` instead; this container cannot run an
inference server (see that module's docstring).

The API key falls back to a dummy rather than failing, since some endpoints
ignore it entirely.
"""
from __future__ import annotations

import time

from .base import ChatBackend, RetriableError, BackendError
from .keys import KeyRing
from .types import GenerationResult, ImagePart, Message, TextPart, VideoPart

# A key that is out of credit, disabled, or revoked. Retrying the SAME key
# after these is pointless -- unlike a 429, they do not clear with time -- so
# they trigger rotation rather than backoff.
#
# Matched on the message because providers disagree on the status code:
# OpenRouter returns 403 "Key limit exceeded" for an exhausted credit cap,
# where a 402 or 429 would be the more obvious choice. Classifying it as
# non-retriable (the default for a 4xx) meant a pool with a spent first key
# failed every call while a funded second key sat unused.
_DEAD_KEY_MARKERS = (
    "key limit exceeded", "insufficient credit", "insufficient_quota",
    "exceeded your current quota", "billing", "payment required",
    "invalid api key", "incorrect api key", "account is disabled",
)


class OpenAICompatBackend(ChatBackend):
    def __init__(self, name: str, model: str, base_url: str | None = None,
                 api_key: str | None = None, api_keys: list[str] | None = None,
                 **kwargs):
        super().__init__(name=name, model=model, **kwargs)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required for openai_compat backend: pip install openai"
            ) from e

        self._openai = OpenAI
        self._base_url = base_url
        # vLLM ignores the key unless launched with --api-key; a dummy keeps
        # the client from refusing to construct.
        pool = [k for k in (api_keys or ([api_key] if api_key else [])) if k]
        self._ring = KeyRing(pool or ["EMPTY"])
        self._client = self._client_for(self._ring.current)

    def _client_for(self, key: str):
        return self._openai(
            base_url=self._base_url,
            api_key=key or "EMPTY",
            timeout=self.timeout,
            max_retries=0,  # base.generate owns retries so backoff is uniform
        )

    def _rotate_key(self) -> None:
        """Move to the next key after a terminal per-key rejection.

        With one key this is a no-op; with several it turns a per-key credit
        cap into a pooled budget.
        """
        if len(self._ring) > 1:
            self._client = self._client_for(self._ring.rotate())

    def _to_openai(self, messages: list[Message]) -> list[dict]:
        out = []
        for m in messages:
            content = []
            for part in m.content:
                if isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": part.to_data_url()},
                    })
                elif isinstance(part, VideoPart):
                    # OpenRouter's documented video format: a `video_url` part
                    # carrying a base64 data URL, alongside `image_url` for
                    # stills. VERIFIED against the live API on a real judge
                    # deck -- the usage breakdown came back with
                    # `video_tokens: 1680`, so the video reaches the model
                    # rather than being dropped on the floor.
                    #
                    # This previously raised, on the belief that the OpenAI-
                    # compatible route had no video part type. That was wrong:
                    # https://openrouter.ai/docs/features/multimodal/videos
                    #
                    # The instinct behind the old guard is still right, though,
                    # and is why this must stay a real part rather than a
                    # best-effort attachment: a backend that accepted the call
                    # but silently discarded the video would return a
                    # confident score computed from the source image alone,
                    # indistinguishable from a genuine one. If a provider on
                    # this route cannot take video it must fail the request,
                    # not answer without it.
                    #
                    # Provider support VARIES (the docs note Gemini on AI
                    # Studio takes only YouTube links); google/gemini-3.7-flash
                    # via OpenRouter accepts inline base64, which is what the
                    # animation judges send.
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": part.to_data_url()},
                    })
            out.append({"role": m.role, "content": content})
        return out

    def _generate(self, messages: list[Message], **params) -> GenerationResult:
        import openai

        started = time.time()
        kwargs = {
            "model": self.model,
            "messages": self._to_openai(messages),
        }
        if "max_tokens" in params:
            kwargs["max_tokens"] = params["max_tokens"]
        if "temperature" in params:
            kwargs["temperature"] = params["temperature"]
        if "top_p" in params:
            kwargs["top_p"] = params["top_p"]

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            raise RetriableError(str(e)) from e
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                raise RetriableError(str(e)) from e
            message = str(e)
            if any(m in message.lower() for m in _DEAD_KEY_MARKERS):
                self._ring.mark_exhausted()
                self._rotate_key()
                error = RetriableError(message)
                # No backoff: a different key is active now, so the next
                # attempt should go straight out rather than sleeping on a
                # limit that will never clear for the key that just failed.
                error.rotated = len(self._ring) > 1
                raise error from e
            raise

        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return GenerationResult(
            text=choice.message.content or "",
            raw=resp,
            model=resp.model or self.model,
            latency_s=time.time() - started,
            usage=usage,
        )
