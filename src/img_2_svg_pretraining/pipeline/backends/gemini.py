"""Google Gemini chat backend (google-genai SDK).

Gemini differs from the OpenAI shape in two ways this adapter absorbs:
system instructions are a top-level config field rather than a message with
role="system", and images are inline `Part` blobs rather than data URLs.
"""
from __future__ import annotations

import time

from .base import ChatBackend, RetriableError
from .types import GenerationResult, ImagePart, Message, TextPart

_RETRIABLE_MARKERS = ("429", "500", "502", "503", "504",
                      "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED")


class GeminiBackend(ChatBackend):
    def __init__(self, name: str, model: str, api_key: str | None = None, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai package required for gemini backend: pip install google-genai"
            ) from e
        if not api_key:
            raise ValueError(
                f"backend '{name}': no API key. Set GOOGLE_API_KEY or api_key_env in the config."
            )
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

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

        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=gtypes.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None,
            )
        except Exception as e:
            if any(marker in str(e) for marker in _RETRIABLE_MARKERS):
                raise RetriableError(str(e)) from e
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
