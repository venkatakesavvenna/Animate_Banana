import threading
from multiprocessing import Queue
from typing import List
from uuid import uuid4
from queue import Empty

from vision_ingest.utils import GracefulShutdown
from .vllm import SamplingParams
from vision_ingest.core.wire import VLLMRequest, VLLMResponse

# ---------------------------------------------------------------------------
# IPC helpers — thin stage()/submit_token()/drain_available() around the queues.
#
# As of v1.5 these are the only survivors of this file: retry and batching
# logic moved into VLMPrediction (prediction.py), which owns the retry/ready
# pools. Phase A2 split the old one-shot submit() into two steps:
#
#   stage()        — build the VLLMRequest (and, back when payloads were huge,
#                    pre-pickle it into a shared-memory slot).
#   submit_token() — the ONLY stage-side work left between generate(i) and
#                    generate(i+1): a bare request_queue.put() of what stage()
#                    produced.
#
# Per-item sampling params are supported: each stage() call takes one item's
# params, so fresh prompts and retries can ride the same batch at different
# settings.
#
# v1.9.1 note: the two-step shape is now purely vestigial. It existed to hide an
# expensive shm acquire+pickle under GPU time, and both the expense and the
# shared-memory pool are gone — a v1.8 request is paths + text, so building one
# costs nothing worth scheduling around. The split is kept because it costs
# nothing either and `_stage_and_submit` reads clearly as "prepare, then hand
# off"; the shm parameters that used to thread through every function here are
# gone. See docs/v1.9.1_changes.md §4.4.
# ---------------------------------------------------------------------------

def stage(
    stage_name: str,
    prompt: dict,
    sampling_params: SamplingParams,
) -> tuple:
    """
    Build one VLLMRequest and return `(req_id, token)`. The token IS the request —
    submit_token() puts it straight on the request queue.
    """
    req_id = str(uuid4())
    return req_id, VLLMRequest(
        req_id=req_id,
        stage_name=stage_name,
        prompt=prompt,
        sampling_params=sampling_params,
    )


def submit_token(request_queue: Queue, token) -> None:
    """
    Put one prepared token on the request queue.

    This is intentionally the whole of stage-side submission: feeding the
    backend is one cheap queue.put(). Keep it trivial — anything heavier belongs
    in stage().
    """
    request_queue.put(token)


def _decode_response(got) -> VLLMResponse:
    """
    Turn one response-queue token into the VLLMResponse it carries.

    Since v1.9.2 that is the identity: `debug_big_payload` — the x10 payload
    inflation used to benchmark IPC cost, which every worker and both directions of
    this file had to honour — retired with `timed_worker.py`. See
    docs/archive/timed_worker.py.removed.
    """
    return got


def drain_available(
    response_queue: Queue,
    stop_event: threading.Event = None,
    first_timeout: float = 0.25,
) -> List[VLLMResponse]:
    """
    Streaming response drain (Phase C). Block at most first_timeout for the
    FIRST response, then greedily take every response already queued.

    Returns a list of VLLMResponse — possibly EMPTY (timed out with nothing
    available); the caller (the prediction pump) simply loops. The 0.25 s poll
    lets a shutdown be noticed promptly; raises GracefulShutdown if stop_event
    is set on entry.

    Order-independent: the pump keys each response back to its record by
    `req_id` via the in-flight dict, so arrival order does not matter.
    """
    if stop_event and stop_event.is_set():
        raise GracefulShutdown("Graceful shutdown requested during drain_available()")

    results: List[VLLMResponse] = []
    try:
        got = response_queue.get(timeout=first_timeout)
    except Empty:
        return results
    results.append(_decode_response(got))

    # Greedily drain everything else already sitting in the queue (no blocking).
    while True:
        try:
            got = response_queue.get_nowait()
        except Empty:
            break
        results.append(_decode_response(got))
    return results


# NOTE (v1.8): `collect()` and `post_process_full_vllm_object()` used to live
# here and are gone.
#
# `collect()` — the blocking "wait for this exact req_id set" helper — was
# already dead code: the streaming pump replaced it in v1.5 and nothing has
# called it since.
#
# `post_process_full_vllm_object()` reconstructed a stats dict from whatever
# object the worker happened to send, duck-typing on `.usage` to tell an online
# shim from a real offline RequestOutput. The worker builds that dict directly
# now, from information it already had firsthand — see wire.stats_from_usage /
# wire.stats_from_request_output.
