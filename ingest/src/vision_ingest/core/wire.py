"""
The wire contract between the stage (`prediction.py`) and whichever `WorkerSpec`
is serving requests — vLLM online, vLLM async, or an ordinary Python function.

Why this module exists (v1.8)
-----------------------------
Up to v1.7 there were *two* wire shapes and *no* response type:

  - The REQUEST payload was decided on the stage side and differed per backend
    (raw text + paths for online; a chat-templated string with PIL images
    embedded for async/batch). Two processes therefore had to agree about
    `args.backend` through two independent call chains, and the whole
    `_payload_kind` / `assert_payload_kind()` / `BackendPayloadMismatch`
    apparatus existed only to make disagreement loud instead of silent.
  - The RESPONSE was an untyped union: `None` meant "infrastructure failure",
    an `_OnlineFailure` instance (detected by `type(pred).__name__`, a string
    match) meant "the server refused this input", and anything else was assumed
    to be an answer, with `.usage` duck-typed to tell the online shim apart from
    a real offline `RequestOutput`. The stage had to reverse-engineer, after the
    fact, information the worker already knew firsthand.

v1.8 collapses both into one shape each:

  REQUEST   {"prompt": <raw user text>, "image_paths": [...],
             "enable_thinking": bool}
            Identical for every backend. The worker owns turning that into
            whatever its engine needs — chat templating and image decoding both
            happen worker-side, using vLLM's own code, so online and offline run
            the same templating path instead of two implementations that happen
            to agree today.

  RESPONSE  VLLMResponse(req_id, kind, text, stats, error)
            `kind` is one of OK / REJECT / INFRA, decided by the worker, which
            is the only place that actually knows. `stats` is built by the
            worker too, from whatever its engine reports, so the stage never
            duck-types a response object again.

The practical payoff is that the taxonomy stops being backend-dependent. A
prompt over `max_model_len` is a REJECT on every backend now; before, it was a
REJECT online (via the server's 400) and a bare `None` offline — which burned
all five infra retries it could never pass and was finally logged as a *backend*
failure. Same input, opposite classification, purely because of which backend
happened to run it.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional

# --- request --------------------------------------------------------------


class VLLMRequest(NamedTuple):
    """
    One request on the shared request queue — the other half of the contract this
    module owns.

    Moved here from the deleted `vlm_service.py` in v1.9.2: it was never a
    property of `VLLMService`, and every backend (including a pure-CPU function
    pool, which uses the identical shape) already had to know it.

    `sampling_params` is typed `Any` rather than `SamplingParams` on purpose —
    importing vLLM's type here would drag vLLM into the import path of
    `core/service.py`, which must stay importable on a machine with no GPU and no
    vLLM installed. A function backend leaves it `None`.
    """
    req_id: str          # uuid4 string — set by stage() in retry_validate
    stage_name: str      # routes the response to the right response queue
    # The unified v1.8 payload, identical for every backend:
    #   {"prompt": <raw user text>, "image_paths": [...], "enable_thinking": bool}
    # NOT chat-templated and carrying no decoded pixels — the worker does both,
    # through vLLM's own renderer. For a function backend it is whatever
    # `FunctionTask.build_request()` returned.
    prompt: Dict
    sampling_params: Any = None


class CallResult(NamedTuple):
    """
    What a `call_fn` returns when it has diagnostics worth recording, not just an
    answer (v1.9.2).

    A `call_fn` may return a bare value — that is the whole output and the
    framework adds only its own timing. A backend that counted something the
    record should carry (vLLM's prompt/completion tokens, a finish_reason) returns
    this instead, and `core/service.py` merges `stats` into the response's stats
    dict under the keys `writer.py` already writes as `full_vlm_object`.

    Explicit rather than "return a 2-tuple and we'll unpack it": plenty of honest
    `call_fn`s return a pair, and guessing which pairs are secretly annotated is
    how a task's real output silently becomes its metadata.
    """
    output: Any
    stats: Optional[Dict[str, Any]] = None


# --- response kinds -------------------------------------------------------
# Deliberately the same four-way taxonomy prediction.py already classifies on
# (`ok` / `fail` / `reject` / `infra`), minus `fail`: whether an answer is
# *valid* is the task's decision (validate_output), so only the stage can make
# it. The worker decides the other three.
KIND_OK = "ok"          # the model answered; `text` is its output
KIND_REJECT = "reject"  # the backend refused THIS input — deterministic, so
                        # re-sending the identical request cannot help. Terminal
                        # on first occurrence.
KIND_INFRA = "infra"    # the backend could not answer at all (engine dead,
                        # connection refused, worker crash). Says nothing about
                        # the input, so it must NEVER mark a path FAILED.


class VLLMResponse(NamedTuple):
    """
    One completed request, as the worker saw it.

    A plain NamedTuple so it pickles across the response queue with no custom
    __reduce__ and no vLLM types leaking into the stage process — `text` is a
    str and `stats` is a plain dict, both of which the stage can consume without
    importing anything from vLLM.
    """
    req_id: str
    kind: str
    text: Optional[Any] = None          # KIND_OK only — see `output` below
    stats: Optional[Dict[str, Any]] = None   # KIND_OK only
    error: Optional[str] = None         # KIND_REJECT / KIND_INFRA only

    @property
    def output(self) -> Any:
        """
        The worker's result payload, whatever it is (v1.9 §2).

        For a vLLM backend that is the generated text, exactly as `text` always
        was. For a function backend (`core/service.py`) it is whatever that
        task's `call_fn` returned — typically a JSON-able dict such as
        `{"bytes_written": n, "w": ..., "h": ...}`.

        The field keeps the name `text` so every existing worker construction
        site (`VLLMResponse.ok(req_id, text, stats)` in online_worker /
        async_worker) is untouched and the vLLM JSONL stays
        byte-identical. `prediction.py` reads it through this name because the
        one thing it does with the value — hand it to
        `wrap_and_validate_output()` — is now type-agnostic.
        """
        return self.text

    @classmethod
    def ok(cls, req_id: str, text: Any, stats: Optional[dict] = None) -> "VLLMResponse":
        return cls(req_id, KIND_OK, text=text, stats=stats or {})

    @classmethod
    def reject(cls, req_id: str, error: str) -> "VLLMResponse":
        return cls(req_id, KIND_REJECT, error=error)

    @classmethod
    def infra(cls, req_id: str, error: str) -> "VLLMResponse":
        return cls(req_id, KIND_INFRA, error=error)


# --- per-request stats ----------------------------------------------------
# Both builders produce the SAME keys, so writer.py's `full_vlm_object` and
# PipelineMetrics' token averages read identically whichever backend ran.
#
#   prompt_tokens / completion_tokens — token counts
#   mm_tokens                         — per-modality multimodal breakdown, or
#                                       None where the backend doesn't report it
#   finish_reason / stop_reason       — check finish_reason == "length" to catch
#                                       silently truncated output

_EMPTY_STATS = {
    "prompt_tokens": None,
    "completion_tokens": None,
    "mm_tokens": None,
    "finish_reason": None,
    "stop_reason": None,
}


def stats_from_request_output(output) -> dict:
    """
    Build the stats dict from a vLLM `RequestOutput` (async / batch backends).

    Never raises: this only feeds diagnostics, and a failure here must not turn
    an already-successful generation into a retry.
    """
    try:
        completion = output.outputs[0]
        prompt_token_ids = getattr(output, "prompt_token_ids", None)
        completion_token_ids = getattr(completion, "token_ids", None)
        return {
            "prompt_tokens": len(prompt_token_ids) if prompt_token_ids is not None else None,
            "completion_tokens": (
                len(completion_token_ids) if completion_token_ids is not None else None
            ),
            # vLLM's offline RequestOutput exposes no per-modality token
            # breakdown — only the HTTP server counts that (see below).
            "mm_tokens": None,
            "finish_reason": getattr(completion, "finish_reason", None),
            "stop_reason": getattr(completion, "stop_reason", None),
        }
    except Exception:
        return dict(_EMPTY_STATS)


def stats_from_usage(usage, finish_reason=None, stop_reason=None) -> dict:
    """
    Build the stats dict from a `vllm serve` /v1/chat/completions `usage` object
    (online backend). The server already counted all of this, so nothing is
    re-derived here.
    """
    try:
        usage = usage or {}
        mm_tokens = (usage.get("prompt_tokens_details") or {}).get("multimodal_tokens")
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "mm_tokens": mm_tokens,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
        }
    except Exception:
        return dict(_EMPTY_STATS)


# --- request payload ------------------------------------------------------

def build_chat_messages(prompt: dict) -> List[dict]:
    """
    Turn one unified request payload into OpenAI-style chat messages — the input
    format every backend now shares.

    `prompt` is what `VLLMConfig.get_prompt_with_image()` built:
        {"prompt": <raw user text>, "image_paths": [<abs path>, ...],
         "enable_thinking": <bool>}

    Images are referenced as `file://` URLs rather than embedded: `vllm serve`
    reads them itself, and offline vLLM's own `MediaConnector` does the same via
    `--allowed-local-media-path` / `allowed_local_media_path`. That is what lets
    one payload shape serve both, and it is why the queue now only ever carries
    paths + text instead of 3-12 MB of PIL image per prompt.

    Text-only prompts send `content` as a plain string rather than a
    single-element list: that is the shape a text-only model's chat template
    expects, and it is the shape the online backend has been sending in
    production since v1.6.
    """
    from vision_ingest.vllm_module.media import to_file_url

    text = prompt["prompt"]
    image_paths = prompt.get("image_paths") or []
    if not image_paths:
        return [{"role": "user", "content": text}]

    content = [
        {"type": "image_url", "image_url": {"url": to_file_url(p)}}
        for p in image_paths
    ]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def chat_template_kwargs(prompt: dict) -> dict:
    """
    Chat-template kwargs carried by the payload. `enable_thinking` is the only
    one today; it rides on the request (rather than being baked in at prompt
    build time as it was offline pre-v1.8) so both backends hand it to the
    template the same way.
    """
    enable_thinking = prompt.get("enable_thinking")
    return {} if enable_thinking is None else {"enable_thinking": enable_thinking}
