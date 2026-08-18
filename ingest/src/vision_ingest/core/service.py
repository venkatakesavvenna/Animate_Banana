"""
`WorkerPool` — the one backend manager (v1.9.2 §2).

It spawns N worker processes on one shared request queue, watches them, restarts
or buries them, recovers a dead worker's in-flight requests, and publishes a
single `ServiceStatus` so `cli()` can tell "the backend is gone" from "this image
failed". There is no second one of these. vLLM is a `WorkerSpec` (see
`vllm_module/specs.py`), exactly like a headless-chrome renderer is.

Why this replaced two classes with one (v1.9.2 §0)
--------------------------------------------------
v1.9 added `FunctionService` *beside* `VLLMService` rather than under it, and the
codebase then paid for two parallel supervision stories:

    concern              VLLMService (v1.4)          FunctionService (v1.9)
    process management   was itself a Process,       pool + supervisor thread
                         nobody watched it
    worker death         invisible -> a per-rank     proc.is_alive() @0.5s
                         heartbeat hack (v1.9.1)
    in-flight recovery   none (one implicit slot)    ledger -> infra responses
    restart              never                       bounded respawn budget
    concurrency          hand-rolled asyncio queue   strict get-one/answer-one
                         loops in two modules

Almost every concept the code was criticised for — two service classes, two
`from_channel`s, a health object named "VLM" that chrome workers published, a
per-`k` ctypes factory with a module `__getattr__` so it could be pickled,
30-second SIGKILL detection through `STALE_S` — existed to keep those two columns
in sync. Deleting the left column deleted all of it. The per-worker heartbeat in
particular was never a real mechanism: it existed because nothing held the
`Process` handles in a place where the answer mattered. The supervisor does, so
`proc.is_alive()` every 0.5s replaces it outright.

`(rank, device, ctx)` is the whole abstraction
---------------------------------------------
One shape covers every workload with no branching:

    workload          devices              n_workers   init_fn builds
    HTML->PNG / SoM   None                 32          one headless chrome per process
    SAM3              ["cuda:0".."cuda:7"] 8           one model replica per GPU
    `vllm serve`      ["0,1,2,3", "4,..."] 2           one server per GPU group
    AsyncLLM engine   ["0,1,2,3"]          1           one in-process engine

Hosting two models means two pools on two channels; hosting two replicas of one
model means `n_workers=2` on one channel, and nothing else changes.

Load balancing is a dispatcher in the pool that hands each request to the
least-loaded worker with free capacity. Through v1.9.1 all N workers instead
competed on the one shared queue — no scheduler, no per-worker queues — which
reads better and cannot survive a SIGKILL: a consumer killed while holding the
queue's shared reader lock never releases it, and every survivor then blocks on it
forever. The full argument, the measured rate, and what the inversion bought back
are on `WorkerPool`.

Sync or async — the loop's only degree of freedom
-------------------------------------------------
`inspect.iscoroutinefunction(spec.call_fn)` picks the loop at spawn, and that is
the only branch in this file:

  - **sync**: take one assignment, call, respond, acknowledge. A worker holds one
    request at a time, so `max_inflight_per_worker` is 1.
  - **async**: one event loop, one task per assignment, up to
    `max_inflight_per_worker` at a time. This is what vLLM's continuous batching
    needs, and it is the *only* thing vLLM ever genuinely needed that a chrome
    renderer did not. Written once here instead of twice in `online_worker.py` and
    `async_worker.py`.

Neither loop enforces its own ceiling: the pool's dispatcher never assigns a worker
more than `max_inflight_per_worker`, so admission is bounded by the one component
that owns the ledger. v1.9.1 needed a semaphore in the worker as well, and two
places that can disagree about a limit eventually do.

Failure classification — no new taxonomy
----------------------------------------
    call_fn returns              -> ok      wrap_and_validate_output() still
                                             decides whether the output is GOOD
    call_fn raises InputRejected -> reject   terminal on first occurrence
    call_fn raises BackendGone   -> infra    for this request, and the worker then
                                             stops: its own resource is dead
    call_fn raises anything else -> infra    retried on the infra budget; NEVER
                                             marks a path FAILED

`infra` as the default for an unexpected exception is the correct bias: a chrome
crash or a CUDA OOM is infrastructure, not data. `prediction.py`'s
`_classify_infra` fuse turns a sustained streak of them into `VLMServiceDown` —
aborting with in-flight paths left at state=1 — rather than marching the dataset
into state=3. That taxonomy is unchanged since v1.9 and knows nothing about any
of this.
"""

from __future__ import annotations

import asyncio
import inspect
import multiprocessing as mp
import os
import threading
import time
import traceback
from queue import Empty
from types import SimpleNamespace
from typing import Callable, Dict, List, NamedTuple, Optional, Set

from vision_ingest.utils.utils import get_logger
from .wire import CallResult, VLLMResponse

from .status import (
    HEARTBEAT_INTERVAL_S,
    MAX_INFLIGHT_HARD_CEILING,
    REASON_SERVER_EXITED,
    REASON_STARTUP_FAILED,
    REASON_WORKER_DIED,
    Ledger,
    ServiceStatus,
)
from .task import BackendGone, InputRejected

# What to do when a worker dies unexpectedly.
ON_DEATH_RESPAWN = "respawn"
ON_DEATH_ABORT = "abort"
_VALID_ON_DEATH = (ON_DEATH_RESPAWN, ON_DEATH_ABORT)

# How often the supervisor checks whether its children are alive. This is the
# whole hard-kill detection latency now (v1.9.1 needed STALE_S = 30s for the same
# fact, because the watcher did not live where the handles lived).
SUPERVISE_INTERVAL_S = 0.5

# Backoff before a respawn, so a fast-crashing worker cannot spin the CPU.
RESPAWN_BACKOFF_S = 2.0

# Exit status a worker uses when it stops because its own resource died
# (`BackendGone`), so the supervisor can report *why* rather than just "exitcode
# 1". Chosen outside the ranges the interpreter and signals use.
EXIT_BACKEND_GONE = 17


class WorkerSpec(NamedTuple):
    """
    Everything one worker process needs to become itself.

    `init_fn` / `call_fn` / `term_fn` MUST be module-level functions: they are
    pickled into a spawned child, and a lambda, a closure or a bound method is
    not picklable. `args` must be JSON-able — the same constraint StageWeaver
    already imposes on `init_vars`, and for the same reason.

        init_fn(args, rank, device) -> ctx
            Build this worker's resource: launch a headless chrome, load a SAM3
            replica onto `device`, launch a `vllm serve` subprocess, open an
            AsyncLLM engine. Returns whatever `call_fn` needs; the framework
            treats it as opaque. May be `async def` when `call_fn` is — it then
            runs inside the worker's event loop, which is a hard requirement for
            AsyncLLM (its background output loop is scheduled on the running
            loop).

        call_fn(ctx, payload, params) -> output
            Do the work for one item. `payload` is the dict `build_request()`
            returned (or the unified vLLM prompt payload); `params` is that
            request's inference parameters — a `SamplingParams` for a vLLM spec,
            `None` for everything else, since only a model has any use for it.
            Raise `InputRejected` for "this input is unusable", `BackendGone` for
            "my resource is dead"; anything else raised is treated as
            infrastructure.

            The two arguments are the request's *content*, deliberately not the
            request itself: `req_id` and `stage_name` are routing, which is the
            framework's business and none of the task's.

            **If this is `async def`, the worker runs an event loop and holds up
            to `max_inflight_per_worker` calls concurrently.** That single fact is
            what lets vLLM be an ordinary spec.

        term_fn(ctx) -> None
            Optional teardown: close the browser, terminate the subprocess, free
            the model. May be `async def` alongside an async `call_fn`.
    """
    init_fn: Callable                    # (args, rank, device) -> ctx
    call_fn: Callable                    # (ctx, payload) -> output; may be async
    term_fn: Optional[Callable] = None   # (ctx) -> None
    # None rather than {} — a NamedTuple default is created once and shared by
    # every instance that omits it, so a mutable one is a footgun waiting to be
    # mutated. The worker normalises with `spec.args or {}`.
    args: Optional[dict] = None          # JSON-able
    devices: Optional[List[str]] = None  # ["cuda:0", ...] — one process per entry
    n_workers: int = 1                   # == len(devices) when devices is given

    # How many requests ONE worker process may hold at once. For a sync `call_fn`
    # this can only be 1 (the loop is strictly get-one/answer-one); for an async
    # one it is the concurrency ceiling the event loop enforces with a semaphore,
    # and it should be the real number that worker's own resource limits imply.
    #
    # Deliberately NOT derived from `batch_size`: that is prediction.py's DB-fetch
    # admission control, a different subsystem tuned for a different reason — and
    # not even a single value, since several stages with different batch_sizes can
    # share one pool under one StageWeaver init_source group. Also deliberately
    # not one large shared constant: a ceiling generous enough to "never matter"
    # is generous enough that a real bug (a leaked semaphore permit inflating
    # concurrency 5-10x) sails through undetected. Hitting it is a logged signal.
    max_inflight_per_worker: int = 1

    # What the supervisor does when a worker dies unexpectedly (v1.9.2 §2).
    #
    #   "respawn" — recover its in-flight requests as `infra`, restart it up to
    #               `max_respawns`, and only fail the service when every rank is
    #               gone. Right for a pool of many cheap, independent workers: one
    #               chrome crash costs throughput, not the run.
    #   "abort"   — the first unexpected death fails the service. Right for an
    #               expensive singleton whose death usually reproduces (a vLLM OOM
    #               or driver fault), where a silent relaunch burns GPU-minutes to
    #               arrive at the same crash. The run ends cleanly instead: rows
    #               reset to their fetch state, nothing marked FAILED.
    on_worker_death: str = ON_DEATH_RESPAWN

    # How many times one rank may be respawned before the pool gives up on it.
    # Bounded on purpose: a worker whose init_fn is broken dies instantly and
    # forever, and an unbounded supervisor would respawn it in a tight loop for
    # the whole run instead of letting the pool go all-dead and abort cleanly.
    max_respawns: int = 3

    def resolved_n_workers(self) -> int:
        """
        One process per device when `devices` is given, else `n_workers`.

        Disagreeing about the count is a configuration bug worth failing on
        rather than silently reconciling — "8 GPUs, 32 workers" means someone
        expects something this pool will not do. `n_workers` left at its default
        of 1 alongside a device list is just an omission, not a disagreement.
        """
        if self.devices:
            if self.n_workers != 1 and self.n_workers != len(self.devices):
                raise ValueError(
                    f"WorkerSpec has {len(self.devices)} devices but n_workers="
                    f"{self.n_workers}; one process is spawned per device, so these "
                    "must agree (or leave n_workers unset)."
                )
            return len(self.devices)
        return max(1, int(self.n_workers))

    def device_for(self, rank: int) -> Optional[str]:
        return self.devices[rank] if self.devices else None

    def resolved_max_inflight(self) -> int:
        """
        Validated `max_inflight_per_worker`.

        Refused at construction time the way `resolved_n_workers()` already
        refuses a devices/n_workers mismatch: an absurd value here is a units
        mistake (someone reaching for batch_size) rather than a real ceiling, and
        it would silently disable the very overflow signal the parameter exists to
        produce.

        A sync `call_fn` above 1 is refused outright rather than quietly ignored:
        the sync loop physically cannot hold two requests, so declaring otherwise
        means whoever wrote the spec expects concurrency they will not get.
        """
        k = int(self.max_inflight_per_worker)
        if k < 1:
            raise ValueError(
                f"WorkerSpec.max_inflight_per_worker must be >= 1, got {k}."
            )
        if k > MAX_INFLIGHT_HARD_CEILING:
            raise ValueError(
                f"WorkerSpec.max_inflight_per_worker={k} exceeds the sanity ceiling "
                f"of {MAX_INFLIGHT_HARD_CEILING}. This is how many requests ONE "
                "worker process holds at once, not the pipeline's batch size — "
                "prediction.py's admission control is a separate knob."
            )
        if k > 1 and not self.is_async():
            raise ValueError(
                f"WorkerSpec.max_inflight_per_worker={k} but call_fn "
                f"{getattr(self.call_fn, '__name__', '?')!r} is not `async def`. The "
                "synchronous worker loop answers one request at a time, so it can "
                "never hold more than one; make call_fn async to get concurrency, or "
                "leave max_inflight_per_worker at 1."
            )
        return k

    def resolved_on_worker_death(self) -> str:
        policy = str(self.on_worker_death)
        if policy not in _VALID_ON_DEATH:
            raise ValueError(
                f"WorkerSpec.on_worker_death={policy!r} is not one of "
                f"{_VALID_ON_DEATH}."
            )
        return policy

    def is_async(self) -> bool:
        """Whether this spec's worker runs an event loop — the single branch."""
        return inspect.iscoroutinefunction(self.call_fn)


# ---------------------------------------------------------------------------
# Response construction — one taxonomy, one place
# ---------------------------------------------------------------------------


def _ok_response(req, rank: int, device: Optional[str], result, t0: float) -> VLLMResponse:
    """
    One successful call, plus the stats every record carries.

    A `call_fn` returning a bare value gets only the framework's own timing; one
    returning a `CallResult` also contributes its own counters (vLLM's token
    counts, finish_reason). The worker's identity is written last so it can never
    be shadowed by a backend that happens to use the same key — with n_workers > 1
    "which replica answered this" is the first thing anyone asks.
    """
    if isinstance(result, CallResult):
        output, stats = result.output, dict(result.stats or {})
    else:
        output, stats = result, {}
    stats.update({
        "worker_rank": rank,
        "device": device,
        "duration_ms": round((time.time() - t0) * 1000, 2),
    })
    return VLLMResponse.ok(req.req_id, output, stats)


def _failure_response(req, rank: int, exc: BaseException, logger) -> VLLMResponse:
    """
    Classify one raised exception. Shared verbatim by the sync and async loops so
    the taxonomy cannot drift between them.
    """
    if isinstance(exc, InputRejected):
        # Deterministic per-input refusal: this HTML is unrenderable, this file is
        # not what it claims, this prompt exceeds max_model_len. Re-running is
        # pointless, so prediction.py fails it terminally on the first occurrence.
        logger.warning(f"rank={rank} rejected input for {req.req_id}: {exc}")
        return VLLMResponse.reject(req.req_id, str(exc))
    # Everything else is infrastructure until proven otherwise — a chrome crash,
    # a CUDA OOM, a full disk, a refused connection. Retried on the infra budget;
    # never marks a path FAILED.
    logger.error(
        f"rank={rank} call_fn failed for {req.req_id}: "
        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    )
    return VLLMResponse.infra(
        req.req_id, f"worker {rank}: {type(exc).__name__}: {exc}"
    )


def _worker_main(
    spec: WorkerSpec,
    rank: int,
    device: Optional[str],
    inbox,
    response_queues: Dict[str, object],
    stop_event,
    ledger: Ledger,
    logs_dir: str,
) -> None:
    """
    One worker process. Module-level so it is picklable under the `spawn` start
    method, which this repo forces everywhere.

    `inbox` is THIS worker's private queue, fed by the pool's dispatcher — not the
    shared request queue. See `WorkerPool._dispatch_loop` for why that matters; the
    short version is that a hard-killed consumer leaves an `mp.Queue`'s internal
    read lock held forever, so a queue with more than one consumer is destroyed by
    the death this pool exists to survive.

    No heartbeat, and no cooperative death flag: this process's liveness is the
    supervisor's business, and the supervisor holds its `Process` handle. What this
    process reports is its readiness (once) and its completions (one per request) —
    the only two facts nothing outside it can observe.
    """
    logger = get_logger(logs_dir, f"worker_{rank}")
    logger.info(
        f"worker rank={rank} pid={os.getpid()} device={device} "
        f"mode={'async' if spec.is_async() else 'sync'} "
        f"max_inflight={spec.max_inflight_per_worker}"
    )
    try:
        if spec.is_async():
            asyncio.run(_async_worker(
                spec, rank, device, inbox, response_queues, stop_event, ledger,
                logger,
            ))
        else:
            _sync_worker(
                spec, rank, device, inbox, response_queues, stop_event, ledger,
                logger,
            )
    except SystemExit as e:
        # A deliberate exit — today only EXIT_BACKEND_GONE, already logged with its
        # actual reason by the loop that raised it. No traceback: this is a
        # decision, not a crash.
        logger.info(f"worker rank={rank} exiting with status {e.code}")
        raise
    except BaseException:
        # The only record of why this process died — the supervisor sees an exit
        # code and nothing else.
        logger.error(f"worker rank={rank} died:\n{traceback.format_exc()}")
        raise
    finally:
        # Opt out of the exit-time flush-and-join for every queue this process
        # put() to. A response_queue can hold unread items if the collecting stage
        # died first, which would otherwise leave this process's feeder thread
        # blocked on a full pipe with no reader and hang exit — taking the parent's
        # join() with it. See docs/archive/exit_hang_postmortem.md.
        for q in response_queues.values():
            try:
                q.cancel_join_thread()
            except Exception:
                pass


def _init_args(spec: WorkerSpec, stop_event) -> SimpleNamespace:
    """
    The namespace `init_fn` receives: the spec's own JSON-able `args`, plus the
    group's `stop_event`.

    The stop event is injected rather than left out because an expensive `init_fn`
    must be interruptible. Launching `vllm serve` and waiting for `/health` can
    take half an hour; without a shutdown signal to poll, Ctrl-C during startup
    would either hang for that whole window or leave the pool to SIGTERM a worker
    mid-launch and orphan its GPU subprocess. `spec.args` stays JSON-able — the
    framework supplies the primitive, which the worker process already holds.
    """
    fields = dict(spec.args or {})
    fields["stop_event"] = stop_event
    return SimpleNamespace(**fields)


def _run_term(spec: WorkerSpec, ctx, rank: int, logger) -> None:
    if ctx is None or spec.term_fn is None:
        return
    try:
        spec.term_fn(ctx)
    except Exception as e:
        logger.error(f"rank={rank} term_fn failed: {type(e).__name__}: {e}")


def _sync_worker(
    spec, rank, device, inbox, response_queues, stop_event, ledger, logger
) -> None:
    """Take one assignment, answer it, acknowledge it. The pool never sends more
    than one at a time to a sync worker (`max_inflight_per_worker` is 1)."""
    ctx = None
    gone: Optional[BackendGone] = None
    try:
        t0 = time.time()
        logger.info(f"worker rank={rank} running init_fn")
        ctx = spec.init_fn(_init_args(spec, stop_event), rank, device)
        logger.info(f"worker rank={rank} ready in {time.time() - t0:.1f}s")
        ledger.mark_worker_ready(rank)

        while not stop_event.is_set():
            try:
                slot, req = inbox.get(timeout=0.25)
            except Empty:
                continue

            call_start = time.time()
            try:
                response = _ok_response(
                    req, rank, device,
                    spec.call_fn(ctx, req.prompt, req.sampling_params), call_start
                )
            except BackendGone as e:
                # This worker's own resource is dead. Answer this request as infra
                # so its item is retried rather than failed, then stop: the pool
                # decides whether a replacement is worth spawning.
                logger.error(f"rank={rank} backend gone: {e}")
                response = VLLMResponse.infra(
                    req.req_id, f"worker {rank}: backend gone: {e}"
                )
                gone = e
            except Exception as e:
                response = _failure_response(req, rank, e, logger)

            response_queues[req.stage_name].put(response)
            # Release AFTER the response is on its way, so the pool cannot reuse
            # the slot while the answer is still in flight.
            ledger.release(rank, slot)
            if gone is not None:
                break

        if gone is None:
            logger.info(f"worker rank={rank} stopping (stop_event set)")
    finally:
        _run_term(spec, ctx, rank, logger)

    if gone is not None:
        # After term_fn, so the resource is cleaned up before the process goes. The
        # distinct status is what lets the supervisor say why.
        raise SystemExit(EXIT_BACKEND_GONE)


def _blocking_get(inbox, timeout: float):
    """`inbox.get()` for an executor thread; None means "nothing yet"."""
    try:
        return inbox.get(timeout=timeout)
    except Empty:
        return None


async def _async_worker(
    spec, rank, device, inbox, response_queues, stop_event, ledger, logger
) -> None:
    """
    One event loop, one task per assignment.

    This is `online_worker._amain` / `async_worker._amain` promoted to framework
    code and written once. Three mechanisms those loops needed are gone:

      - the two bridge threads: the blocking queue read is one `run_in_executor`
        await, and `mp.Queue.put()` on an unbounded queue never blocks (it appends
        and wakes the feeder), so responses go straight out from the task.
      - the concurrency semaphore: the pool hands out at most
        `max_inflight_per_worker` slots, so admission is already bounded by the
        thing that owns the ledger. One enforcement point, not two.
      - the per-request abort/drain bookkeeping: a task that is cancelled emits its
        own `infra` response, and the pool recovers anything that never got that
        far from the slots it still holds.
    """
    loop = asyncio.get_running_loop()
    active: Dict[str, asyncio.Task] = {}
    ctx = None
    # One-element list rather than a `nonlocal`: `_serve` is a closure and this
    # keeps the assignment visible without making the whole flag a class.
    gone: List[BaseException] = []

    async def _serve(slot: int, req) -> None:
        """One request: call, classify, respond, acknowledge. Always emits exactly
        one response, including on shutdown cancellation — a missing response would
        wedge that req_id in prediction.py's in-flight dict forever."""
        t0 = time.time()
        try:
            try:
                response = _ok_response(
                    req, rank, device,
                    await spec.call_fn(ctx, req.prompt, req.sampling_params), t0
                )
            except asyncio.CancelledError:
                response_queues[req.stage_name].put(VLLMResponse.infra(
                    req.req_id, f"worker {rank}: cancelled during shutdown"
                ))
                raise
            except BackendGone as e:
                # This worker's resource is dead — every other in-flight request is
                # about to fail the same way. Answer this one as infra and tell the
                # accept loop to stop pulling; the pool applies the policy.
                logger.error(f"rank={rank} backend gone: {e}")
                response = VLLMResponse.infra(
                    req.req_id, f"worker {rank}: backend gone: {e}"
                )
                gone.append(e)
            except Exception as e:
                response = _failure_response(req, rank, e, logger)
            response_queues[req.stage_name].put(response)
        finally:
            ledger.release(rank, slot)
            active.pop(req.req_id, None)

    try:
        ctx = spec.init_fn(_init_args(spec, stop_event), rank, device)
        if inspect.isawaitable(ctx):
            # An async init_fn is not a convenience: AsyncLLM schedules its
            # background output loop on the *running* loop, so its engine must be
            # constructed in here rather than handed in from a sync caller.
            ctx = await ctx
        logger.info(
            f"worker rank={rank} ready (async, up to "
            f"{spec.max_inflight_per_worker} concurrent)"
        )
        ledger.mark_worker_ready(rank)

        while not stop_event.is_set() and not gone:
            got = await loop.run_in_executor(None, _blocking_get, inbox, 0.25)
            if got is None:
                continue
            slot, req = got
            active[req.req_id] = asyncio.create_task(_serve(slot, req))

        # Shutdown: stop accepting, let in-flight tasks finish (each emits its own
        # response), then cancel stragglers — which also emit one.
        logger.info(
            f"worker rank={rank} stopping "
            f"({'backend gone' if gone else 'stop_event set'}) — draining "
            f"{len(active)} in-flight request(s)"
        )
        pending = list(active.values())
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=30
                )
            except asyncio.TimeoutError:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
    finally:
        if ctx is not None and spec.term_fn is not None:
            try:
                result = spec.term_fn(ctx)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                logger.error(f"rank={rank} term_fn failed: {type(e).__name__}: {e}")

    if gone:
        raise SystemExit(EXIT_BACKEND_GONE)



# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


class WorkerPool:
    """
    Owns `n_workers` processes, the dispatcher that feeds them, the supervisor that
    keeps them honest, and the `ServiceStatus` it publishes on the channel.

    Not a `Process` itself (unlike the deleted `VLLMService`): there is nothing for
    a parent process to do except manage children, and a supervisor must live in
    the same process as the `Process` objects it watches. That single fact is what
    makes hard-kill detection cost 0.5 seconds instead of 30.

        pool = WorkerPool(spec, channel, logs_dir=logs_dir)
        pool.start()      # publishes the status, spawns workers, starts the
                          # dispatcher, the supervisor and the status heartbeat
        ...
        pool.stop()       # stop_event, join workers (term_fn runs in each), join
                          # threads

    That is the whole `init_fn` contract now. v1.9.1 needed five steps — build a
    `VLMHealth` with a matching `n_workers`/`max_inflight`, construct the service
    `from_channel`, `start()`, `attach_process()`, `set_health()` — and each was a
    chance to get the ordering or the sizes wrong. The pool owns all of it because
    it is the only thing that knows any of it.

    Why the pool dispatches instead of letting workers share one queue
    -----------------------------------------------------------------
    Through v1.9.1 all N workers called `get()` on the one shared `request_queue`,
    which reads beautifully — "N processes competing on one queue is work-stealing
    by construction" — and does not survive the failure this class exists to
    handle.

    `mp.Queue.get(timeout=...)` acquires the queue's shared reader lock, polls the
    pipe, and releases it. A process SIGKILLed while holding that lock never
    releases it — it is a POSIX semaphore in shared memory and CPython has no
    robust-futex recovery — so **every surviving consumer then blocks on it
    forever.** When that happens the pool dutifully recovers the dead worker's
    requests, respawns it, reports the service healthy — and nothing is ever
    consumed again. A silent, permanent stall: exactly the failure the in-flight
    ledger was invented to prevent, one level down.

    An idle worker is inside `get()` continuously, but only one consumer holds the
    lock at a time, so the per-kill odds are roughly 1/N — measured at 3 of 8 trials
    with two workers (`examples/no_gpu_validation/mp_only.py`, ~20 lines of pure
    `multiprocessing` with none of this code involved). 1/N is not a reprieve: the
    consequence is total whenever it lands, it is unrecoverable, and a long run kills
    workers more than once.

    So the shared `request_queue` now has exactly ONE consumer — `_dispatch_loop`,
    here in the pool — and each worker reads a private inbox that only it consumes.
    A killed worker can still poison its own inbox; the pool throws that queue away
    and gives the replacement a fresh one. The poison is contained in the thing
    being discarded anyway.

    What that costs: one thread and one extra hop for a request (a few hundred
    microseconds against a model call measured in seconds). What it buys, beyond
    not hanging: the pool assigns the work, so it knows what every worker owes
    *before sending it*, which closes the checkout gap v1.9.1 documented as a known
    hole and makes ledger overflow unrepresentable. Load balancing is unchanged in
    effect — the dispatcher picks the least-loaded worker with free capacity, which
    is what the competing-consumers arrangement approximated.

    The residual case, stated honestly: workers still share the per-stage RESPONSE
    queues, so a worker SIGKILLed inside the window where its feeder thread holds
    that queue's write lock can still wedge the response path. The window is far
    smaller — microseconds of pipe write per response, against a reader lock held
    across a 0.25s poll — but it is the same bug, and closing it would mean routing
    every response through this process too. Left as a known limitation rather than
    paid for on the hot path, and recorded here so it is a decision rather than an
    oversight.
    """

    def __init__(self, spec: WorkerSpec, channel, logs_dir: str = "./logs"):
        self.spec = spec
        self.n_workers = spec.resolved_n_workers()
        self.max_inflight = spec.resolved_max_inflight()
        self.on_worker_death = spec.resolved_on_worker_death()
        self.max_respawns = max(0, int(spec.max_respawns))

        self.channel = channel
        self.request_queue = channel.request_queue
        self.response_queues = channel.response_queues
        self.stop_event = channel.stop_event
        self.logs_dir = logs_dir

        # Created HERE, in the process that will spawn the workers — the one
        # fd-inheritance-safe window. The status is published on the channel; the
        # ledger deliberately is not (v1.9.2 §3), and neither are the inboxes:
        # which worker holds which request is a conversation between a worker and
        # its supervisor, and nothing outside this pool has any use for it.
        self.status = ServiceStatus()
        self.ledger = Ledger(self.n_workers, self.max_inflight)

        self.logger = get_logger(logs_dir, "worker_pool")
        self._procs: Dict[int, mp.Process] = {}
        self._inboxes: Dict[int, mp.Queue] = {}
        self._respawns: Dict[int, int] = {r: 0 for r in range(self.n_workers)}
        # Ranks the pool has permanently given up on. Tracked so readiness can
        # still resolve while the pool runs degraded, instead of leaving cli()
        # blocked forever on a rank that is never coming back.
        self._abandoned: Set[int] = set()

        # The dispatch record: rank -> {slot -> (dispatch_seq, VLLMRequest)}. Plain
        # Python, because the pool is the only thing that writes OR reads it — the
        # inversion that let v1.9.1's shared byte arrays, their fixed field widths
        # and their write-ordering rules all disappear. A slot is outstanding while
        # its dispatch_seq differs from the worker's `ledger.done_seq`.
        self._assigned: Dict[int, Dict[int, tuple]] = {
            r: {} for r in range(self.n_workers)
        }
        self._dispatch_seq: Dict[int, Dict[int, int]] = {
            r: {i: 0 for i in range(self.max_inflight)}
            for r in range(self.n_workers)
        }
        # Guards _assigned/_dispatch_seq/_procs/_inboxes between the dispatcher and
        # the supervisor. A plain threading.Lock, because both are threads in this
        # one process — the thing shared memory could never be given.
        self._lock = threading.Lock()

        self._threads: List[threading.Thread] = []
        self._stop_threads = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.logger.info(
            f"starting WorkerPool: n_workers={self.n_workers}, "
            f"max_inflight_per_worker={self.max_inflight}, "
            f"mode={'async' if self.spec.is_async() else 'sync'}, "
            f"on_worker_death={self.on_worker_death}, "
            f"max_respawns={self.max_respawns}, devices={self.spec.devices}, "
            f"init_fn={getattr(self.spec.init_fn, '__name__', '?')}"
        )
        # Published before the first spawn. The channel is pickled at every spawn,
        # so anything placed in `extras` now reaches every process started later —
        # and nothing placed there afterwards reaches a process already started.
        self.channel.set_health(self.status)

        for rank in range(self.n_workers):
            self._spawn(rank)

        self._start_thread(self._status_loop, "pool-status")
        self._start_thread(self._dispatch_loop, "pool-dispatch")
        self._start_thread(self._supervise_loop, "pool-supervisor")

    def _start_thread(self, target, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _spawn(self, rank: int) -> None:
        """
        Start rank's process with a FRESH inbox.

        Fresh on every respawn, not reused: the previous one may have been left with
        its reader lock held by the process that just died, which would silently
        wedge the replacement on its first `get()`. The old queue is released with
        its join thread cancelled so a feeder blocked on that same poison cannot
        hold up this process's exit.
        """
        old = self._inboxes.pop(rank, None)
        if old is not None:
            try:
                old.cancel_join_thread()
            except Exception:
                pass

        inbox = mp.Queue()
        self._inboxes[rank] = inbox
        proc = mp.Process(
            target=_worker_main,
            args=(
                self.spec,
                rank,
                self.spec.device_for(rank),
                inbox,
                self.response_queues,
                self.stop_event,
                self.ledger,
                self.logs_dir,
            ),
            name=f"worker-{rank}",
            daemon=False,  # non-daemon so join() in stop() works
        )
        proc.start()
        self._procs[rank] = proc
        self.logger.info(f"spawned worker rank={rank} pid={proc.pid}")

    def stop(self, timeout: Optional[float] = 120.0) -> None:
        """
        Shut the pool down: signal, then wait, then insist.

        `term_fn` runs inside each worker's own `finally`, so a clean join is what
        releases GPU memory and terminates any launched subprocess. A worker that
        will not exit within `timeout` is terminated — a hung teardown must not hold
        up the run's exit.
        """
        self.logger.info("stopping WorkerPool")
        self.stop_event.set()
        self._stop_threads.set()

        for rank, proc in list(self._procs.items()):
            try:
                proc.join(timeout=timeout)
                if proc.is_alive():
                    self.logger.warning(
                        f"worker rank={rank} did not exit within {timeout}s — terminating"
                    )
                    proc.terminate()
                    proc.join(timeout=10.0)
            except Exception as e:
                self.logger.error(f"joining worker rank={rank} failed: {e}")

        for t in self._threads:
            if t.is_alive():
                t.join(timeout=5.0)
        self._threads.clear()

        # Every inbox was written by THIS process, so its feeder thread is ours to
        # release. Undelivered assignments are not lost quietly: whatever the pool
        # still holds outstanding is recovered below.
        for inbox in self._inboxes.values():
            try:
                inbox.cancel_join_thread()
            except Exception:
                pass
        self.logger.info(f"WorkerPool fully stopped — {self.status.describe()}")

    def is_alive(self) -> bool:
        return any(p.is_alive() for p in self._procs.values())

    def inflight_count(self) -> int:
        """How many requests the pool believes are outstanding across all ranks."""
        with self._lock:
            return sum(len(self._outstanding(r)) for r in range(self.n_workers))

    # ------------------------------------------------------------------
    # Dispatch — the shared request queue's one and only consumer
    # ------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        """
        Move one request at a time from the shared queue into a worker's inbox.

        Capacity is checked BEFORE the read, for the same reason the v1.9.1 worker
        acquired its semaphore before `get()`: a request pulled off the shared queue
        with nowhere to put it would sit in this process, hiding backpressure from
        `prediction.py`'s admission control and losing the request outright if the
        pool were killed.
        """
        while not self._stop_threads.is_set() and not self.stop_event.is_set():
            rank = self._pick_rank()
            if rank is None:
                # Everyone is full, or nobody is up yet. Short sleep rather than a
                # condition variable: the wake-up source would be a worker in
                # another process, which is what `done_seq` already is.
                time.sleep(0.005)
                continue
            try:
                req = self.request_queue.get(timeout=0.25)
            except Empty:
                continue
            except (OSError, ValueError, EOFError) as e:
                # The channel is being torn down underneath us.
                self.logger.info(f"dispatcher stopping — request queue closed: {e}")
                return
            if not self._dispatch(rank, req):
                # The chosen rank died between _pick_rank() and here. Re-queue and
                # let the next pass choose again; the request keeps its place at the
                # back rather than being lost.
                self.request_queue.put(req)

    def _pick_rank(self) -> Optional[int]:
        """
        The ready, alive rank with the most free capacity, or None if every worker
        is at its ceiling.

        Least-loaded rather than round-robin, which is what makes this equivalent to
        the competing-consumers arrangement it replaces: a worker stuck on a slow
        item simply stops being the least loaded, and the others keep taking work.
        """
        best, best_free = None, 0
        with self._lock:
            for rank, proc in self._procs.items():
                if rank in self._abandoned or not self.ledger.is_ready(rank):
                    continue
                if not proc.is_alive():
                    continue
                free = self.max_inflight - len(self._outstanding(rank))
                if free > best_free:
                    best, best_free = rank, free
        return best

    def _outstanding(self, rank: int) -> List[tuple]:
        """
        `[(slot, request), ...]` this rank has been given and not yet answered.

        Call with `self._lock` held. A slot is outstanding when the pool's dispatch
        counter for it is ahead of the worker's `done_seq` — which covers both "sent
        but never picked up" and "picked up and killed mid-work" without needing to
        tell them apart, because from the pipeline's point of view they are the same
        thing: a request nobody is going to answer.
        """
        out = []
        for slot, (seq, req) in self._assigned[rank].items():
            if seq != self.ledger.done_count(rank, slot):
                out.append((slot, req))
        return out

    def _dispatch(self, rank: int, req) -> bool:
        """Claim a free slot for `req` and send it. False if the rank went away."""
        with self._lock:
            if rank not in self._procs or rank in self._abandoned:
                return False
            inbox = self._inboxes.get(rank)
            if inbox is None:
                return False
            outstanding = {slot for slot, _ in self._outstanding(rank)}
            slot = next(
                (i for i in range(self.max_inflight) if i not in outstanding), None
            )
            if slot is None:
                return False
            # Record before sending. There is no window in which the worker holds a
            # request the pool does not know about.
            self._dispatch_seq[rank][slot] = self.ledger.done_count(rank, slot) + 1
            self._assigned[rank][slot] = (self._dispatch_seq[rank][slot], req)
        try:
            inbox.put((slot, req))
            return True
        except (OSError, ValueError) as e:
            self.logger.error(f"failed to dispatch to rank={rank}: {e}")
            with self._lock:
                self._assigned[rank].pop(slot, None)
            return False

    # ------------------------------------------------------------------
    # Status thread — the pool's own liveness, and readiness
    # ------------------------------------------------------------------

    def _status_loop(self) -> None:
        """
        Stamp the heartbeat and promote readiness.

        Separate from the supervisor on purpose: the supervisor blocks — a respawn
        backoff, a `Process.start()` that imports vLLM — and the heartbeat's only
        job is to be independent of whatever the pool happens to be busy with.
        Exactly the reasoning that put the v1.9.1 heartbeat on its own thread,
        applied to the one process that still needs one.
        """
        self.status.beat()
        while not self._stop_threads.is_set():
            self.status.beat()
            if not self.status.is_ready() and self._readiness_resolved():
                self.logger.info(
                    f"backend ready — {self.ledger.ready_count()}/{self.n_workers} "
                    f"worker(s) usable"
                )
                self.status.mark_ready()
            time.sleep(HEARTBEAT_INTERVAL_S)

    def _readiness_resolved(self) -> bool:
        """
        True when every rank has resolved — reported ready or been given up on — and
        at least one is usable.

        The `_abandoned` term matters: with a strict "all ranks ready" rule, one rank
        with a broken `init_fn` would exhaust its respawn budget, be left dead, and
        leave `cli()` blocked on `is_ready()` forever while the other 31 workers
        happily drained the queue. A pool that has lost *every* rank is not ready
        either, but that case is reported through `mark_failed`, which is what
        actually releases the wait.
        """
        ready = self.ledger.ready_count()
        return ready >= 1 and (ready + len(self._abandoned)) >= self.n_workers

    # ------------------------------------------------------------------
    # Supervisor — recovering a dead worker's requests, precisely
    # ------------------------------------------------------------------

    def _supervise_loop(self) -> None:
        """
        Watch for a worker exiting while `stop_event` is clear, every
        `SUPERVISE_INTERVAL_S`. `Process.is_alive()` in the process that started it
        is the whole mechanism — free, immediate, and correct for a SIGKILL, an
        OOM-kill and a clean crash alike.
        """
        while not self._stop_threads.is_set() and not self.stop_event.is_set():
            for rank, proc in list(self._procs.items()):
                if proc.is_alive():
                    continue
                if self._stopping():
                    return
                self._handle_worker_death(rank, proc)
            time.sleep(SUPERVISE_INTERVAL_S)

    def _stopping(self) -> bool:
        return self._stop_threads.is_set() or self.stop_event.is_set()

    def _handle_worker_death(self, rank: int, proc) -> None:
        """
        1. Recover every request this rank owed, as one `infra` response each on the
           correct response queue. Without this those req_ids sit in prediction.py's
           `inflight` dict forever: `_owned` never decrements, `fetch_n` walks to 0,
           and the pipeline stalls in silence.
        2. Apply `on_worker_death` — respawn within budget, or fail the service.
        """
        was_ready = self.ledger.is_ready(rank)
        reported_gone = proc.exitcode == EXIT_BACKEND_GONE
        self.logger.error(
            f"worker rank={rank} pid={proc.pid} exited unexpectedly "
            f"(exitcode={proc.exitcode}, had_completed_init={was_ready})"
        )

        self._recover_assigned(rank, proc)
        self.ledger.clear_ready(rank)

        # The exit status is the only thing that crosses from a dead process, so
        # translate it into the sentence an operator needs. `BackendGone` is a
        # worker's own diagnosis (its `vllm serve` exited, its engine core died) and
        # reads very differently from "killed with signal 9".
        if reported_gone:
            detail = (
                f"rank {rank} reported its backend resource is gone "
                f"(see worker_{rank}.log for the exact cause)"
            )
            code = REASON_SERVER_EXITED
        else:
            detail = f"rank {rank} exitcode={proc.exitcode}"
            code = REASON_WORKER_DIED
        if not was_ready:
            detail += " (died before init_fn completed)"
            code = REASON_STARTUP_FAILED

        with self._lock:
            self._procs.pop(rank, None)

        if self.on_worker_death == ON_DEATH_ABORT:
            self.logger.error(
                f"on_worker_death=abort — failing the service on the first "
                f"unexpected death ({detail}). Stages will abort with "
                "VLMServiceDown; in-flight rows are reset, nothing is marked FAILED."
            )
            self._fail(code, detail)
            return

        if self._respawns[rank] >= self.max_respawns:
            self.logger.error(
                f"worker rank={rank} has already been respawned "
                f"{self._respawns[rank]} times — leaving it dead. The pool runs "
                "degraded; if every rank reaches this point the run aborts with "
                "VLMServiceDown and no path is marked FAILED."
            )
            self._abandon(rank, detail, code)
            return

        self._respawns[rank] += 1
        time.sleep(RESPAWN_BACKOFF_S)
        if self._stopping():
            return
        try:
            with self._lock:
                self._spawn(rank)
            self.logger.warning(
                f"respawned worker rank={rank} "
                f"({self._respawns[rank]}/{self.max_respawns})"
            )
        except Exception as e:
            self.logger.error(f"failed to respawn worker rank={rank}: {e}")
            self._abandon(rank, f"{detail}; respawn failed: {e}", code)

    def _abandon(self, rank: int, detail: str, code: int) -> None:
        """Give up on one rank, and on the service if that was the last one."""
        self._abandoned.add(rank)
        if len(self._abandoned) < self.n_workers:
            return
        # If not one rank ever became usable this was a startup failure, however it
        # presented per-rank — that is the distinction an operator acts on ("it never
        # worked" vs "it worked and then died").
        if self.ledger.ready_count() == 0:
            code = REASON_STARTUP_FAILED
        self._fail(
            code,
            f"all {self.n_workers} worker(s) dead; last: {detail}; respawn budget "
            f"({self.max_respawns}) exhausted",
        )

    def _fail(self, code: int, detail: str) -> None:
        """
        Record why the service is gone, then stop everything.

        The pool is the only thing that can write this: it is alive after its workers
        die, it holds their `Process` objects, and it knows how much respawn budget
        was spent. That is the whole "exact reason for failure" mechanism — a live
        process reporting on dead ones.
        """
        self.status.mark_failed(code, detail)
        self.stop_event.set()

    def _recover_assigned(self, rank: int, proc) -> None:
        with self._lock:
            owed = self._outstanding(rank)
            self._assigned[rank] = {}
        if not owed:
            self.logger.info(
                f"worker rank={rank} owed no requests — nothing to recover"
            )
            return
        for slot, req in owed:
            queue = self.response_queues.get(req.stage_name)
            if queue is None:
                self.logger.error(
                    f"worker rank={rank} died owing req_id={req.req_id} for unknown "
                    f"stage {req.stage_name!r} — cannot recover it; that request "
                    "will stall its item"
                )
                continue
            queue.put(VLLMResponse.infra(
                req.req_id,
                f"worker {rank} died holding this request (exitcode={proc.exitcode})",
            ))
            self.logger.warning(
                f"recovered req_id={req.req_id} from dead worker rank={rank} as an "
                f"infra response on stage {req.stage_name!r} — it will be retried, "
                "not failed"
            )



__all__ = [
    "WorkerPool",
    "WorkerSpec",
    "ON_DEATH_RESPAWN",
    "ON_DEATH_ABORT",
]
