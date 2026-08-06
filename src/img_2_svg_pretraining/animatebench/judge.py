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

    def unload(self) -> None:
        self._backend.unload()


# Used when the candidate config defines no gemini backend of its own (the
# local-model configs replaced it). Keys still come from api_keys.csv via the
# backend factory's normal resolution.
FALLBACK_JUDGE = {"type": "gemini", "model": "gemini-3.6-flash", "max_concurrency": 2}


def make_judge(cfg, cache_root: Path | None = None,
               raw_dir: Path | None = None, backend_name: str = "gemini_flash") -> Judge:
    """Judge from a pipeline config's backend definition.

    Prefers the candidate config's own gemini entry (and thereby the
    api_keys.csv pool); falls back to a standalone gemini-3.6-flash definition
    when the config doesn't carry one, so judging a local-model run needs no
    config edits.
    """
    try:
        backend_cfg = cfg.backend_cfg(backend_name)
        if backend_cfg.get("type") != "gemini":
            backend_cfg = FALLBACK_JUDGE
    except Exception:
        backend_cfg = FALLBACK_JUDGE
    return Judge(backend_name if backend_cfg is not FALLBACK_JUDGE else "judge_gemini",
                 backend_cfg, cache_root=cache_root, raw_dir=raw_dir)
