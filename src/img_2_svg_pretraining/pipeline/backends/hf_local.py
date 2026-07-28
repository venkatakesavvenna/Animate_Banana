"""In-process HuggingFace transformers chat backend.

Ported from `benchmark/infer.py::ModelRunner`, keeping what worked there
(chat-template application, prompt-length slicing, left-padded batching) and
fixing one thing: the original branched on literal model names

    if spec.name != "gemma-4-12b" and spec.name != "gemma-4-e4b":

to pick between AutoModelForImageTextToText and AutoModelForMultimodalLM,
which silently sent `gemma-4-31b` down the wrong arm. Here the class is a
config field (`model_class`) instead.

Prefer the openai_compat backend against a vLLM server for anything at
scale; this exists for models that are easier to run in-process, and for
running without a server.
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import ChatBackend
from .types import GenerationResult, ImagePart, Message, TextPart

MODEL_CLASSES = ("image_text_to_text", "multimodal_lm")


class HFLocalBackend(ChatBackend):
    def __init__(self, name: str, model: str, hf_repo: str,
                 model_class: str = "image_text_to_text",
                 trust_remote_code: bool = False,
                 attn_implementation: str = "sdpa",
                 dtype: str = "bfloat16",
                 device: str = "cuda",
                 max_new_tokens: int = 16384,
                 **kwargs):
        # Local generation is inherently serial on one GPU; concurrency > 1
        # would only contend for the same device.
        kwargs.setdefault("max_concurrency", 1)
        kwargs.setdefault("max_retries", 1)
        super().__init__(name=name, model=model or hf_repo, **kwargs)

        if model_class not in MODEL_CLASSES:
            raise ValueError(
                f"backend '{name}': model_class must be one of {MODEL_CLASSES}, got '{model_class}'"
            )

        import torch
        from transformers import AutoProcessor

        self._torch = torch
        self.device = device
        self.hf_repo = hf_repo
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(
            hf_repo, trust_remote_code=trust_remote_code,
        )

        if model_class == "multimodal_lm":
            from transformers import AutoModelForMultimodalLM as _Model
        else:
            from transformers import AutoModelForImageTextToText as _Model

        self.model = _Model.from_pretrained(
            hf_repo,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            dtype=getattr(torch, dtype),
        ).to(device).eval()

    def unload(self) -> None:
        import gc
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        self._torch.cuda.empty_cache()

    def _to_hf(self, messages: list[Message]) -> list[dict]:
        from PIL import Image

        out = []
        for m in messages:
            content = []
            for part in m.content:
                if isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    content.append({
                        "type": "image",
                        "image": Image.open(part.path).convert("RGB"),
                    })
            out.append({"role": m.role, "content": content})
        return out

    def _generate(self, messages: list[Message], **params) -> GenerationResult:
        torch = self._torch
        started = time.time()
        max_new_tokens = params.get("max_tokens", self.max_new_tokens)

        with torch.no_grad():
            inputs = self.processor.apply_chat_template(
                self._to_hf(messages),
                add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.device)

            input_len = inputs["input_ids"].shape[-1]
            gen_kwargs = {"max_new_tokens": max_new_tokens}
            temperature = params.get("temperature")
            if temperature:
                gen_kwargs.update(do_sample=True, temperature=temperature)
                if "top_p" in params:
                    gen_kwargs["top_p"] = params["top_p"]
            else:
                gen_kwargs["do_sample"] = False

            output_ids = self.model.generate(**inputs, **gen_kwargs)
            text = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

        return GenerationResult(
            text=text, model=self.model_name, latency_s=time.time() - started,
        )

    @property
    def model_name(self) -> str:
        return self.hf_repo

    def generate_batch(self, batch: list[list[Message]], **params) -> list[GenerationResult]:
        """True batched generation via left padding.

        Left-padding is required for decoder-only `.generate()`: right-padding
        would put pad tokens between the prompt and the first generated token.
        Caching is checked per-item first so a partially-cached batch only
        pays for its misses.
        """
        if not batch:
            return []

        merged = {**self.default_params, **params}
        results: list[GenerationResult | None] = [None] * len(batch)
        todo: list[int] = []
        for i, messages in enumerate(batch):
            cached = self._cache_get(messages, merged)
            if cached is not None:
                results[i] = cached
            else:
                todo.append(i)

        if todo:
            try:
                fresh = self._generate_batch_uncached([batch[i] for i in todo], **merged)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                fresh = [GenerationResult(text="", model=self.model_name, error=err)
                         for _ in todo]
            for i, result in zip(todo, fresh):
                results[i] = result
                self._cache_put(batch[i], merged, result)

        return [r for r in results if r is not None]

    def _generate_batch_uncached(self, batch: list[list[Message]], **params
                                 ) -> list[GenerationResult]:
        torch = self._torch
        started = time.time()
        max_new_tokens = params.get("max_tokens", self.max_new_tokens)
        tokenizer = self.processor.tokenizer

        prev_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        prev_pad_token = tokenizer.pad_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            with torch.no_grad():
                inputs = self.processor.apply_chat_template(
                    [self._to_hf(m) for m in batch],
                    add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt", padding=True,
                ).to(self.device)

                input_len = inputs["input_ids"].shape[-1]
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                )
                texts = [
                    self.processor.decode(row, skip_special_tokens=True)
                    for row in output_ids[:, input_len:]
                ]
        finally:
            tokenizer.padding_side = prev_padding_side
            # Restore pad_token too -- benchmark/infer.py left this mutated.
            tokenizer.pad_token = prev_pad_token

        elapsed = time.time() - started
        return [
            GenerationResult(text=t, model=self.model_name, latency_s=elapsed / len(batch))
            for t in texts
        ]
