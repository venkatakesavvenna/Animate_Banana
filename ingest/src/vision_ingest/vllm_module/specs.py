"""
vLLM as a `WorkerSpec` — the whole of "starting a VLM" (v1.9.2 §4).

    pool = WorkerPool(vllm_online_spec(args), channel, logs_dir=logs_dir)
    pool.start()

The only genuine difference between vLLM and a headless-chrome renderer is that
one vLLM worker must hold many requests at once. That is expressed by
`online_call` / `async_call` being `async def`, which makes `core/service.py` run
an event loop with an `asyncio.Semaphore(max_inflight_per_worker)`. Everything
else `VLLMService` did — being a `Process`, dispatching on a `backend` string,
running its own accept loop and bridge threads, reporting its own liveness — was
a second backend manager, and is gone.

What that buys, beyond the deletion:

  - **Two servers are `n_workers=2`.** N processes competing on one request queue
    is work-stealing by construction, so load balancing across two `vllm serve`
    instances on disjoint GPU groups needs no scheduler and no new code. Two
    *different models* remain two pools on two channels, which is the same
    topology as before.
  - **A vLLM worker is supervised.** Its death is noticed in ≤0.5s by the pool
    that spawned it, its in-flight requests come back as `infra`, and
    `on_worker_death` decides whether a replacement is worth launching.

`backend="batch"` is retired. The batch-boundary `LLM.generate()` worker existed
as an A/B timing baseline and cannot be a call-per-request `call_fn` — it holds
finished results hostage until the last straggler in its batch finishes, which is
the exact problem continuous batching solved. `timed_worker.py` and
`debug_big_payload` are in `docs/archive/`; no escape hatch was added to keep the
benchmark alive.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from vision_ingest.core.service import ON_DEATH_ABORT, WorkerSpec

from .async_worker import async_call, async_init, async_term
from .online_worker import online_call, online_init, online_term

VALID_BACKENDS = ("online", "async")


def _spec_args(engine_args: Dict, gpus: str, logs_dir: str, vllm_bin: Optional[str] = None) -> dict:
    """
    The JSON-able bundle every vLLM worker needs. `gpus` is the fallback for a
    single-worker pool that pins no devices; a multi-worker pool passes one device
    string per rank through `WorkerSpec.devices`, and the rank's own device wins.
    """
    return {"engine_args": dict(engine_args), "gpus": gpus or "", "logs_dir": logs_dir,
            "vllm_bin": vllm_bin}


def vllm_online_spec(
    engine_args: Dict,
    gpus: str,
    max_inflight: int,
    logs_dir: str = "./logs",
    devices: Optional[List[str]] = None,
    on_worker_death: str = ON_DEATH_ABORT,
    max_respawns: int = 3,
    vllm_bin: Optional[str] = None,
) -> WorkerSpec:
    """
    `vllm serve` over HTTP, one subprocess per worker.

    `devices` is a list of CUDA_VISIBLE_DEVICES strings — `["0,1,2,3", "4,5,6,7"]`
    launches two 4-GPU servers that share the request queue. Each picks its own
    ephemeral port, so there is no port arithmetic and no chance of two ranks
    fighting over 8000 (or of either one silently adopting a stale server from a
    previous run — see `_verify_served_model`).

    `max_inflight` is how many requests one server holds concurrently; the stage's
    `batch_size` is the natural value, since that is what prediction.py's admission
    control keeps in flight per stage.

    `vllm_bin`: which `vllm` executable to launch — None (default) resolves
    "vllm" off PATH, i.e. whatever env this process is already running in.
    Pass an absolute path to serve this model from a DIFFERENT venv than the
    one driving the pipeline (e.g. a model needing a newer/older vllm than
    the other stages) — see `online_worker._launch_server`.
    """
    return WorkerSpec(
        init_fn=online_init,
        call_fn=online_call,
        term_fn=online_term,
        args=_spec_args(engine_args, gpus, logs_dir, vllm_bin=vllm_bin),
        devices=devices,
        n_workers=1 if not devices else len(devices),
        max_inflight_per_worker=max_inflight,
        on_worker_death=on_worker_death,
        max_respawns=max_respawns,
    )


def vllm_async_spec(
    engine_args: Dict,
    gpus: str,
    max_inflight: int,
    logs_dir: str = "./logs",
    devices: Optional[List[str]] = None,
    on_worker_death: str = ON_DEATH_ABORT,
    max_respawns: int = 3,
) -> WorkerSpec:
    """
    An in-process AsyncLLM engine per worker — continuous batching with no HTTP
    hop and no subprocess.

    Same shape as the online spec, which is the point: `args.backend` picks a
    builder and nothing downstream of that knows which one ran.
    """
    return WorkerSpec(
        init_fn=async_init,
        call_fn=async_call,
        term_fn=async_term,
        args=_spec_args(engine_args, gpus, logs_dir),
        devices=devices,
        n_workers=1 if not devices else len(devices),
        max_inflight_per_worker=max_inflight,
        on_worker_death=on_worker_death,
        max_respawns=max_respawns,
    )


def vllm_spec(backend: str, *args, **kwargs) -> WorkerSpec:
    """
    Pick a spec builder from `args.backend`, which is the one place that string is
    still interpreted. It no longer reaches any class, any process, or any worker
    loop — a `main.py` resolves it here and passes a `WorkerSpec` onward.

    On `on_worker_death` (v1.9.2 §9 Q1): both vLLM specs default to `"abort"`. A
    vLLM worker dying is usually an OOM or a driver fault that respawning
    reproduces, and a relaunch costs minutes of GPU time while the operator sees
    nothing but a gap in throughput. Aborting is cheap and honest — in-flight rows
    reset to their fetch state, nothing marked FAILED, and the run is restartable
    the moment the cause is fixed. Pools of many cheap independent workers (chrome)
    keep the `"respawn"` default, where one death really is just throughput. Pass
    `on_worker_death="respawn", max_respawns=1` to ride out a genuinely transient
    crash.
    """
    if backend == "online":
        return vllm_online_spec(*args, **kwargs)
    if backend == "async":
        return vllm_async_spec(*args, **kwargs)
    if backend == "batch":
        raise ValueError(
            "backend='batch' was retired in v1.9.2. It was a batch-boundary "
            "LLM.generate() A/B timing baseline and cannot be expressed as a "
            "call-per-request worker; see docs/archive/timed_worker.py. Use "
            "'async' for an in-process engine or 'online' for `vllm serve`."
        )
    raise ValueError(
        f"unknown backend {backend!r}; expected one of {VALID_BACKENDS}"
    )


__all__ = ["vllm_spec", "vllm_online_spec", "vllm_async_spec", "VALID_BACKENDS"]
