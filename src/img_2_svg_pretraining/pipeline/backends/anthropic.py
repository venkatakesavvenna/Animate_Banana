"""Anthropic Claude chat backend.

Like Gemini, Anthropic takes the system prompt as a top-level parameter
rather than a message. Images are base64 source blocks. `max_tokens` is
required by the API, so a default is supplied when the config omits it.
"""
from __future__ import annotations

import time

from .base import ChatBackend, RetriableError, BackendError
from .types import GenerationResult, ImagePart, Message, TextPart, VideoPart

DEFAULT_MAX_TOKENS = 8192


class AnthropicBackend(ChatBackend):
    def __init__(self, name: str, model: str, api_key: str | None = None, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required for anthropic backend: pip install anthropic"
            ) from e
        if not api_key:
            raise ValueError(
                f"backend '{name}': no API key. Set ANTHROPIC_API_KEY or api_key_env in the config."
            )
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=self.timeout,
            max_retries=0,  # base.generate owns retries
        )

    def _to_anthropic(self, messages: list[Message]) -> tuple[list[dict], str | None]:
        """Returns (messages, system)."""
        system_text: list[str] = []
        out = []
        for m in messages:
            if m.role == "system":
                system_text.append(m.text())
                continue
            content = []
            for part in m.content:
                if isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type(),
                            "data": part.to_base64(),
                        },
                    })
                elif isinstance(part, VideoPart):
                    # Loudly, not silently. A backend that ignored the video
                    # would still return a confident score computed from the
                    # source image alone -- indistinguishable from a real one.
                    raise BackendError(
                        f"{self.name}: this backend cannot send video "
                        f"({part.path}); use a 'gemini' backend for video judging")
            out.append({"role": m.role, "content": content})
        return out, ("\n".join(system_text) if system_text else None)

    def _generate(self, messages: list[Message], **params) -> GenerationResult:
        started = time.time()
        msgs, system = self._to_anthropic(messages)

        kwargs = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": params.get("max_tokens", DEFAULT_MAX_TOKENS),
        }
        if system:
            kwargs["system"] = system
        if "temperature" in params:
            kwargs["temperature"] = params["temperature"]
        if "top_p" in params:
            kwargs["top_p"] = params["top_p"]

        try:
            resp = self._client.messages.create(**kwargs)
        except (self._anthropic.RateLimitError,
                self._anthropic.APITimeoutError,
                self._anthropic.APIConnectionError) as e:
            raise RetriableError(str(e)) from e
        except self._anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise RetriableError(str(e)) from e
            raise

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return GenerationResult(
            text=text,
            raw=resp,
            model=resp.model or self.model,
            latency_s=time.time() - started,
            usage={
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            },
        )
