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

from .base import ChatBackend, RetriableError
from .types import GenerationResult, ImagePart, Message, TextPart


class OpenAICompatBackend(ChatBackend):
    def __init__(self, name: str, model: str, base_url: str | None = None,
                 api_key: str | None = None, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required for openai_compat backend: pip install openai"
            ) from e

        self._client = OpenAI(
            base_url=base_url,
            # vLLM ignores the key unless launched with --api-key; a dummy keeps
            # the client from refusing to construct.
            api_key=api_key or "EMPTY",
            timeout=self.timeout,
            max_retries=0,  # base.generate owns retries so backoff is uniform
        )

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
