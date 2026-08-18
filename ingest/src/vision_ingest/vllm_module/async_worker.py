"""
The in-process vLLM backend as three functions: build an AsyncLLM engine, drive
one request through it, shut it down (v1.9.2 §4).

`vllm_module/specs.py` wraps these into a `WorkerSpec`. What used to be here — an
accept loop, two bridge threads, a lazy-load branch, an abort-and-drain shutdown
sequence and a `_SENTINEL` — is `core/service.py`'s async worker loop now, shared
with the online backend and with anything else that ever wants concurrency.

Continuous batching is unchanged, and is still the reason this backend exists:
one request is added per submission, one async generator per request, finished
items stream back the instant they complete, and fresh/retry work joins the
running batch immediately, so a straggler delays only itself. The framework's
`asyncio.Semaphore(max_inflight_per_worker)` is what admits many at once.

v1.8's other half also stands: the worker owns prompt construction, via
`engine.renderer.render_chat_async()` — the exact call `vllm serve` makes for a
/v1/chat/completions request. Two things follow from using vLLM's own renderer
rather than hand-rolling apply_chat_template + PIL decode on the stage side:

  - online and offline run the SAME templating code, so they cannot drift. They
    previously agreed only by coincidence (chandra ships a standalone
    chat_template.jinja that differs from its tokenizer_config.json template for
    enable_thinking=True).
  - the CPU work is already off the event loop, with no thread pool of our own:
    render_messages_async offloads chat templating through
    `make_async(..., executor=self._executor)`, image decoding through
    MediaConnector's `run_in_executor(global_thread_pool, ...)`, and the
    multimodal processor through `make_async(..., executor=self._mm_executor)`.

`lazy_loading` is retired (v1.9.2 §4). It let the engine be built on the first
request instead of at startup, which meant `ServiceStatus.ready` could latch
before anything was loaded and a model-load failure surfaced as a mid-run request
failure rather than a startup failure. Ready now means ready.
"""

from __future__ import annotations

import os
import time
import traceback

from vision_ingest.core.task import BackendGone, InputRejected
from vision_ingest.utils.utils import get_logger

from vision_ingest.core.wire import (
    CallResult,
    build_chat_messages,
    chat_template_kwargs,
    stats_from_request_output,
)


def _build_async_engine(engine_args: dict, gpus: str, logger):
    """
    Build the AsyncLLM engine. Must be called from inside a running event loop —
    `from_engine_args` schedules the engine's background output loop on it, which
    is why `async_init` is a coroutine and the framework awaits it in the worker's
    own loop.
    """
    logger.info("Model Loading started")
    cur_time = time.time()
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ['NCCL_NET_PLUGIN'] = ""

    engine_args = dict(engine_args)
    n_gpus = len(gpus.split(",")) if gpus else 0
    if n_gpus > 0:
        engine_args['tensor_parallel_size'] = n_gpus

    # Local imports: keep vLLM's async surface out of the module import path so
    # the rest of the package imports cleanly without a GPU/vLLM install.
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext

    # async_scheduling is pinned by VLLMConfig.get_engine_args() (default False —
    # vLLM 0.25+ would otherwise enable it implicitly). Log the resolved value: it
    # changes how the scheduler overlaps steps and has been implicated in
    # node-level hangs, so it must never be a mystery which mode a run used.
    logger.info(f"async_scheduling={engine_args.get('async_scheduling')!r}")
    logger.info(f"AsyncLLM engine_args: {engine_args}")
    # AsyncLLM.from_engine_args defaults usage_context to ENGINE_CONTEXT, which is
    # not a key in vLLM's scheduler-defaults lookup table and silently falls
    # through to max_num_batched_tokens=2048 / max_num_seqs=128 — far below what
    # `vllm serve` picks for the same hardware (OPENAI_API_SERVER -> 8192 / 1024 on
    # this class of GPU). Passing OPENAI_API_SERVER here makes the offline engine
    # schedule batches exactly like `vllm serve` does, which matters for output
    # parity since vLLM's kernels are not batch-invariant. See
    # docs/vllm_serve_offline_parity.md, Divergence 1.
    temp = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**engine_args),
        usage_context=UsageContext.OPENAI_API_SERVER,
    )
    logger.info(f"Model loaded in {time.time()-cur_time}")
    return temp


def _is_engine_dead_error(exc: BaseException) -> bool:
    """
    Tell "this request failed" apart from "the engine is gone".

    vLLM raises EngineDeadError / EngineGenerateError once the engine core has
    died; every subsequent generate() then fails the same way. Without this
    distinction the stage retried each request three times and marked the path
    FAILED — turning one engine crash into an entire dataset in state=3.
    """
    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
        return True
    name = type(exc).__name__
    if name in ("EngineDeadError", "EngineGenerateError", "EngineCoreDead"):
        return True
    # Some paths surface it only as a plain RuntimeError/AsyncEngineDeadError
    # message, so fall back to matching the text vLLM uses. "enginecore" (one word)
    # is what EngineDeadError itself says — "EngineCore encountered an issue. See
    # stack trace (above) for the root cause." — which the "engine core"/"died"
    # pairing below does not match on its own.
    text = str(exc).lower()
    if "enginecore" in text:
        return True
    return "engine core" in text and "died" in text or "engine is dead" in text


async def _render(engine, prompt: dict):
    """
    Turn one unified payload into an engine input, using vLLM's own renderer.

    This is `vllm serve`'s own preprocessing path (chat template + `file://` image
    loading + multimodal processing), so an offline request and an online request
    built from the same payload reach the model identically. All three of its
    expensive steps are internally offloaded to executors, so awaiting this does
    not block the event loop — see the module header.
    """
    from vllm.renderers.params import ChatParams
    from vllm.utils.mistral import is_mistral_tokenizer

    renderer = engine.renderer
    chat_params = ChatParams(
        chat_template_kwargs={
            "add_generation_prompt": True,
            "continue_final_message": False,
            # NOT optional, and not a detail: safe_apply_chat_template defaults to
            # tokenize=True, which would return token ids instead of the templated
            # string — so the multimodal processor would run against pre-tokenized
            # input and the engine input would carry no prompt text. Both of vLLM's
            # own call sites (LLM._preprocess_chat and the server's preprocess_chat)
            # pass exactly this expression; mirror it rather than relying on a
            # default neither of them relies on.
            "tokenize": is_mistral_tokenizer(renderer.tokenizer),
            **chat_template_kwargs(prompt),
        },
    )
    _, (engine_input,) = await renderer.render_chat_async(
        [build_chat_messages(prompt)],
        chat_params,
    )
    return engine_input


# ---------------------------------------------------------------------------
# The WorkerSpec triple
# ---------------------------------------------------------------------------


async def async_init(args, rank: int, device):
    """
    `init_fn`: one in-process AsyncLLM engine, fully loaded.

    A coroutine because AsyncLLM must be constructed on the loop that will drive
    it. `AsyncLLM.from_engine_args()` is synchronous and only returns once the
    engine (weights included) is built, so a rank that reaches the end of this
    function really can serve requests.
    """
    logger = get_logger(args.logs_dir, f"vllm_async_{rank}")
    gpus = device if device is not None else getattr(args, "gpus", "") or ""
    engine = _build_async_engine(args.engine_args, gpus, logger)
    return {"engine": engine, "logger": logger, "req_seq": [0]}


async def async_call(ctx, prompt: dict, params):
    """
    `call_fn`: render one request, drive its generator to completion.

    The three-way split is v1.8 §2's point — the worker is the only place that
    knows which of these happened, so it says so rather than emitting a bare None
    the stage has to guess about:

      return          the model answered.
      InputRejected   this request is bad and will be bad every time (unreadable
                      or oversized image, prompt over max_model_len, a chat
                      template that refuses the message shape). Deterministic, so
                      the stage fails it once with the reason instead of burning
                      five infra retries on it.
      BackendGone     the engine core is dead. Every following request would fail
                      identically, so the worker stops rather than reporting the
                      same infra failure a thousand times.
    """
    engine, logger = ctx["engine"], ctx["logger"]

    try:
        engine_input = await _render(engine, prompt)
    except Exception as e:
        # Rendering failed: a bad image, an over-long prompt, a message shape this
        # model's template rejects. All specific to THIS request.
        logger.error(f"rendering failed: {type(e).__name__}: {e}\n"
                     f"{traceback.format_exc()}")
        raise InputRejected(
            f"could not build the request ({type(e).__name__}: {e})"
        ) from e

    # vLLM wants its own request id. The framework's req_id is a uuid4 the stage
    # owns for routing; keeping them separate means a retry of the same item is a
    # genuinely new engine request rather than a collision with an aborted one.
    ctx["req_seq"][0] += 1
    engine_req_id = f"r{ctx['req_seq'][0]}"

    try:
        final = None
        async for output in engine.generate(engine_input, params, engine_req_id):
            final = output
            if getattr(output, "finished", False):
                break
    except Exception as e:
        logger.error(f"generate failed: {e}\n{traceback.format_exc()}")
        if _is_engine_dead_error(e):
            raise BackendGone(f"engine is dead ({type(e).__name__}: {e})") from e
        # A generate() failure that is NOT engine death is this request's own
        # problem — most often the prompt exceeding max_model_len. Before v1.8 this
        # came back as a bare None, burned all five infra retries it could never
        # pass, and was finally reported as a *backend* failure, while the
        # identical input online was correctly reported as a rejection with the
        # server's explanation.
        raise InputRejected(f"{type(e).__name__}: {e}") from e

    if final is None:
        # Not a rejection and not engine death — infra, so it retries.
        raise RuntimeError("engine.generate() ended without producing an output")
    return CallResult(final.outputs[0].text, stats_from_request_output(final))


def async_term(ctx) -> None:
    """`term_fn`: best-effort engine shutdown, which is what frees the GPU."""
    engine = ctx.get("engine")
    if engine is None:
        return
    try:
        engine.shutdown()
    except Exception as e:
        ctx["logger"].error(f"engine.shutdown() failed (ignoring): {e}")
