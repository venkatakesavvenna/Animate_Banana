"""The evaluation judge: one model, strict JSON, cached, evidence kept.

Wraps a pipeline ChatBackend (Gemini 3.6 Flash by default) behind a single
`ask_json` call. Three behaviours every judge metric relies on:

- **Strict JSON or nothing.** The response must parse to a JSON object; one
  retry with an explicit nudge, then the metric records a judge failure
  rather than a fabricated score. A truncated response is refused outright
  (`looks_truncated`) -- the raster integrator taught us that a thinking
  model's half-emitted JSON looks like a format error, not a length error.
- **Caching for free.** The pipeline backend already caches responses by
  content hash, so re-running an eval never re-pays for unchanged inputs.
- **Evidence kept.** Raw responses are written under evals/raw/, because a
  judge score without its transcript cannot be audited.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from img_2_svg_pretraining.pipeline.backends import Message, make_backend
from img_2_svg_pretraining.pipeline.extract import extract_json, looks_truncated

PROMPTS_ROOT = Path(__file__).parent / "prompts"

# Thinking models spend output budget on reasoning before any JSON appears.
DEFAULT_PARAMS = {"temperature": 0.0, "max_tokens": 16384}


class JudgeError(Exception):
    pass


class Judge:
    def __init__(self, backend_name: str, backend_cfg: dict,
                 cache_root: Path | None = None, raw_dir: Path | None = None):
        self.backend_name = backend_name
        self.model = str(backend_cfg.get("model") or backend_name)
        self._backend = make_backend(backend_name, backend_cfg, cache_root=cache_root)
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self._raw_count = 0

    def provenance(self) -> dict:
        return {"judge_backend": self.backend_name, "judge_model": self.model,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def _save_raw(self, tag: str, text: str) -> str | None:
        if self.raw_dir is None:
            return None
        self._raw_count += 1
        path = self.raw_dir / f"{self._raw_count:03d}_{tag}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def ask_json(self, prompt: str, images: list[Path] | None = None,
                 tag: str = "judge", **params) -> dict:
        """One judged question -> parsed JSON object, or JudgeError."""
        merged = {**DEFAULT_PARAMS, **params}
        attempt_prompt = prompt
        last_error = "no attempts made"

        for attempt in range(2):
            result = self._backend.generate(
                [Message.user(attempt_prompt, images=list(images or []))], **merged)
            if not result.ok:
                last_error = result.error or "call failed"
                continue
            raw_path = self._save_raw(tag, result.text or "")
            if looks_truncated(result.text or ""):
                last_error = (f"response truncated mid-JSON (raise max_tokens); "
                              f"raw at {raw_path}")
                continue
            data = extract_json(result.text or "")
            if isinstance(data, dict):
                return data
            last_error = f"no JSON object in response; raw at {raw_path}"
            # A model that chatted instead of answering usually complies when
            # reminded; a second failure is recorded, not retried forever.
            attempt_prompt = (prompt + "\n\nYour previous reply was not a single "
                              "valid JSON object. Respond with ONLY the JSON object.")

        raise JudgeError(f"judge failed after 2 attempts: {last_error}")

    def ask_text(self, prompt: str, images: list[Path] | None = None,
                 tag: str = "judge", accept=None,
                 retry_note: str = "Your previous reply did not follow the "
                                   "required output format. Respond again using "
                                   "exactly the section headers specified.",
                 **params) -> str:
        """One judged question -> raw text, or JudgeError.

        The selection-sensibility and granularity-pacing judges answer in a
        `### HEADER: value` block rather than JSON, so `ask_json` cannot serve
        them -- its retry fires on "no JSON object" and would burn a second
        call on every single timestep, and there are hundreds of them.

        `accept` is an optional predicate over the response text; when it
        rejects, one retry is spent with `retry_note` appended, which is the
        same one-nudge-then-record policy `ask_json` uses. Parsing stays with
        the caller, where the design document's own regexes live.
        """
        merged = {**DEFAULT_PARAMS, **params}
        attempt_prompt = prompt
        last_error = "no attempts made"

        for _ in range(2):
            result = self._backend.generate(
                [Message.user(attempt_prompt, images=list(images or []))], **merged)
            if not result.ok:
                last_error = result.error or "call failed"
                continue
            text = result.text or ""
            raw_path = self._save_raw(tag, text)
            if not text.strip():
                last_error = f"empty response; raw at {raw_path}"
                continue
            if accept is None or accept(text):
                return text
            last_error = f"response did not match the expected format; raw at {raw_path}"
            attempt_prompt = f"{prompt}\n\n{retry_note}"

        raise JudgeError(f"judge failed after 2 attempts: {last_error}")

    def unload(self) -> None:
        self._backend.unload()


# Used when the candidate config defines no gemini backend of its own (the
# local-model configs replaced it). Keys still come from api_keys.csv via the
# backend factory's normal resolution.
FALLBACK_JUDGE = {"type": "gemini", "model": "gemini-3.6-flash", "max_concurrency": 2}


def make_judge(cfg, cache_root: Path | None = None,
               raw_dir: Path | None = None, backend_name: str = "gemini_flash") -> Judge:
    """Judge from a pipeline config's backend definition.

    The named backend is used **as declared**, whatever its type. That is what
    lets the animation suite point at a served Qwen judge through an
    `openai_compat` block without any code change here.

    Falls back to a standalone gemini-3.6-flash definition only when the config
    carries no such backend at all, so judging a local-model run still needs no
    config edits. This used to fall back whenever the type was not `gemini`,
    which silently rewrote every non-Gemini judge -- `--judge-backend qwen` ran
    Gemini and recorded Gemini's name, so the substitution was invisible in the
    record rather than wrong in it. A fallback is now announced on stderr,
    because a score attributed to a model that did not produce it cannot be
    audited.
    """
    try:
        backend_cfg = cfg.backend_cfg(backend_name)
    except Exception as e:
        print(f"judge backend '{backend_name}' unavailable ({e}); "
              f"falling back to {FALLBACK_JUDGE['model']}", file=sys.stderr)
        backend_cfg, backend_name = FALLBACK_JUDGE, "judge_gemini"
    return Judge(backend_name, backend_cfg, cache_root=cache_root, raw_dir=raw_dir)
