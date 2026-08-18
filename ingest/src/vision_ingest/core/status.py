"""
`ServiceStatus` — the only thing a consumer ever learns about a backend — and
`Ledger`, the two acknowledgements a worker sends its pool (v1.9.2 §§1, 3).

The problem this solves (v1.7, unchanged)
-----------------------------------------
Every backend reports a failed request the same way. `prediction.py` then
retried it `max_attempts` times and marked the path FAILED in the DB. That is
right for a bad prompt and catastrophically wrong for a dead engine: when
`vllm serve` or the AsyncLLM engine core died, *every* subsequent request came
back a failure, so the pipeline cheerfully walked the entire database into
state 3 (FAILED) at maximum speed. Those images were never actually attempted —
but a run that ends with 10M rows in state=3 is indistinguishable from one where
the model genuinely refused them.

`prediction.py` checks the status whenever a request comes back an infra failure
and raises `VLMServiceDown` instead of marking anything failed. That propagates
out of `run_iteration()` -> `process_pipeline()` -> `cli()`'s `finally`, which
resets every in-flight path from state 1 back to its fetch state. Nothing gets
marked FAILED, and the run stops immediately instead of burning through the
dataset.

Why this file is not `vllm_module/health.py` any more (v1.9.2)
-------------------------------------------------------------
Because nothing in it is about vLLM. A pool of 32 headless-chrome renderers
imported `VLMHealth` and published it under `extras["health"]`; the name was a
lie in every deployment except the first one. It now sits next to the thing that
writes it (`core/service.py`) and carries no backend in its name.

The split that makes this small: three facts, one writer, no per-rank state
----------------------------------------------------------------------------
v1.9.1 published a per-rank array of nine fields and asked consumers to compute
aggregates over it. That is what forced `attach_process()` (a `Process` handle
does not pickle, so it was permanently inert under StageWeaver), a per-rank
heartbeat in every worker, and a `ctypes.Structure` factory with a module
`__getattr__` so the generated class could be pickled by name.

None of it was needed, because a consumer does not care how many workers are up.
It asks two questions, and there is exactly one process — the pool — that can
answer both: it holds the `Process` handles, it knows the exit codes, and it
decides whether to respawn or give up. So the pool writes, everyone else reads,
and the record has three fields with three distinct jobs:

    ready       "the service is usable" — cli() waits on this once, at startup,
                so model-load time is excluded from the throughput clock. Set
                when every initial worker has finished `init_fn`. Deliberately
                latched: a respawn dipping capacity is a throughput event, not a
                return to "still loading".

    failed      "the pool gave up, and here is why" — the exact-failure-reason
    reason_code mechanism. Cooperative, written by a live pool about its dead
    detail      workers, so it can say things no flag could: which rank, what
                exit code, how many respawns were spent. This is the field that
                distinguishes a dead backend from a bad image, which is the
                whole reason this file exists.

    heartbeat   the one failure the pool cannot report cooperatively: its own
                SIGKILL or OOM-kill. Stamped every second by the pool process.
                If the process responsible for restarting workers is gone, the
                service is unsupervised, and aborting cleanly (rows reset to
                their fetch state, nothing marked FAILED) is the correct
                outcome even if some workers are still momentarily draining.

Note what is NOT here: per-worker liveness. The pool watches its own children
with `proc.is_alive()`, in the process that started them, where the answer is
free and immediate (≤0.5s instead of `STALE_S`). Nothing outside the pool has
any use for that fact.

Deliberately lock-free
----------------------
`mp.Value`/`mp.Array`'s default lock is a POSIX semaphore in shared memory, and
a process SIGKILLed while holding it NEVER releases it — there is no
robust-futex recovery in CPython. Every later reader would then block forever,
poisoning precisely the code path this module exists to serve: `failure_reason()`
asking "did the backend get killed?" would itself hang on the dead process's
lock, turning the one failure mode we must detect into an undetectable hang. A
heartbeat that can be killed by the kill it is watching for is worse than no
heartbeat.

No lock is needed because every field has a single writer — `ServiceStatus` is
written only by the pool process, and `Ledger`'s element for rank `r` only by rank
`r`'s own worker — and every field is a naturally-aligned scalar. A reader sees
the old value or the new one, never a mixture that means something different. The
one ordering a reader could observe mid-update (`mark_failed`) is noted at its
write site.

`STALE_S` is deliberately generous. It now guards only one thing — the pool
process itself vanishing — so a false positive requires the pool's own status
thread to starve for 30 seconds while its children keep running. Strictly rarer
than the v1.9.1 arrangement it replaces, where any worker blocked in GIL-holding
C code could trip it. A status that has never ticked is never stale, so a
five-minute model load is not a corpse.

Explicitly NOT solved with a per-request timeout: under StageWeaver several
stages share one backend, so a request can legitimately sit queued for a very
long time with nothing wrong. A deadline would have to be tuned against
contention it cannot observe. The process that knows the fact reports the fact.
"""

from __future__ import annotations

import ctypes
import logging
import multiprocessing as mp
import time
from ctypes import c_char, c_double, c_int8, c_int32
from typing import Optional

# Never silent, even when the caller has no logger to hand. The pipeline's own
# file loggers are built by get_logger(); this is the floor under them.
_log = logging.getLogger(__name__)

# reason codes — small ints so they fit in a shared ctypes field with no Manager.
REASON_UNKNOWN = 0
REASON_WORKER_EXITED = 1        # the worker loop returned/raised; its finally fired
REASON_ENGINE_FAILED = 2        # in-process engine raised / is dead
REASON_SERVER_EXITED = 3        # a launched subprocess (`vllm serve`) died
REASON_SERVER_UNREACHABLE = 4   # subprocess alive but not answering
REASON_STARTUP_FAILED = 5       # never came up at all
REASON_WORKER_DIED = 6          # worker process(es) died (v1.9)
REASON_SUPERVISOR_LOST = 7      # the POOL stopped heartbeating — SIGKILL/OOM (v1.9.2)

_REASON_TEXT = {
    REASON_UNKNOWN: "the backend stopped for an unknown reason",
    REASON_WORKER_EXITED: "the backend worker exited",
    REASON_ENGINE_FAILED: "the inference engine failed / is dead",
    REASON_SERVER_EXITED: "the backend's server subprocess exited",
    REASON_SERVER_UNREACHABLE: "the backend's server is not answering requests",
    REASON_STARTUP_FAILED: "the backend failed to start",
    # Deliberately not "every worker has died": with on_worker_death="abort" the
    # first death fails the service, and the `detail` the pool writes is what says
    # whether this was one rank or the last of eight.
    REASON_WORKER_DIED: "a backend worker process died and was not replaced",
    REASON_SUPERVISOR_LOST: (
        "the backend pool stopped heartbeating, so nothing is supervising or "
        "restarting its workers (SIGKILL/OOM-kill leaves no other trace)"
    ),
}

# How long the pool may go without stamping `heartbeat` before it is inferred
# dead. Generous on purpose — see the module docstring.
STALE_S = 30.0

# How often the pool stamps its own liveness tick.
HEARTBEAT_INTERVAL_S = 1.0

# Room for the pool's own account of what went wrong. Long enough for
# "all 8 workers dead; rank 3 exitcode=-9, respawn budget (3) exhausted" and
# then some; short enough that the whole status record stays a cache line or two.
DETAIL_FIELD_BYTES = 256

# Refuse an obviously-absurd declared in-flight ceiling at construction time, the
# way WorkerSpec.resolved_n_workers() already refuses a devices/n_workers
# mismatch. This is about catching a typo/units mistake (someone reaching for
# batch_size), not about the memory: `k` is how many requests the pool will let
# ONE worker hold, and an absurd value silently disables the backpressure the
# number exists to provide.
MAX_INFLIGHT_HARD_CEILING = 512


class VLMServiceDown(RuntimeError):
    """
    Fatal: the backend is gone. Raised into the pipeline loop instead of failing
    the in-flight paths, so they stay in state=1 and are reset to their fetch
    state by cli()'s shutdown path (i.e. replayed on the next run).

    Keeps its v1.7 name: it is `cli()`/`prediction.py`'s exception, part of a
    contract with the DB state machine, and renaming it would touch the one place
    in this codebase where a name is load-bearing across three modules.
    """


# ---------------------------------------------------------------------------
# ServiceStatus — pool writes, everyone reads
# ---------------------------------------------------------------------------
#
# A single fixed structure defined at module level, which is the whole reason
# v1.9.1's `_RankState_{k}` factory and its module `__getattr__` are gone:
# `mp.Value(cls)` shares `cls` across `spawn` by pickling it BY REFERENCE, so
# the child re-imports this module and looks the class up by qualified name. A
# class built inside a function has no such name; this one does.


class _Status(ctypes.Structure):
    _fields_ = [
        ("ready",       c_int8),                      # latches 1, never cleared
        ("failed",      c_int8),                      # 1 once the pool has given up
        ("reason_code", c_int8),                      # REASON_* for that failure
        ("heartbeat",   c_double),                    # monotonic() of the pool's last tick
        ("detail",      c_char * DETAIL_FIELD_BYTES), # the pool's own explanation
    ]


class ServiceStatus:
    """
    One backend's liveness, as seen from anywhere.

    Created by `WorkerPool.start()`, which also publishes it on the
    `ServiceChannel` — user code never constructs, sizes or publishes one. That
    is deliberate: v1.9.1 asked `init_fn` to build a health object whose
    `n_workers`/`max_inflight` matched the pool's spec, and validated the match
    with a `ValueError`. A mismatch is no longer representable, so the check and
    the ceremony that needed it are both gone.

    A group with no backend publishes nothing, and `channel.health is None`
    still means "no backend to wait for". The framework must never pre-create
    one: `cli()` blocks in an unbounded loop on `is_ready()`, escapable only by
    `shutdown_event` or `failure_reason()`, so a status object belonging to a
    group that starts no workers would hang that stage at startup instead of
    running it.
    """

    def __init__(self):
        self._s = mp.Value(_Status, lock=False)

    # ------------------------------------------------------------------
    # pool side — the single writer
    # ------------------------------------------------------------------

    def mark_ready(self) -> None:
        """
        The service can serve requests. Called by the pool once every initial
        worker has finished `init_fn`.

        Latches and never un-sets: cli() waits on this once at startup to decide
        when to start the clock, and a worker that dies and respawns mid-run must
        not send the whole service back to "still loading".
        """
        self._s.ready = 1

    def mark_failed(self, code: int, detail: str = "") -> None:
        """
        The pool has given up. Idempotent; the first reason wins, because the
        first thing that went wrong is the one worth reporting.

        `detail` is the pool's own account — which rank, what exit code, how much
        respawn budget was spent — and is what makes a log line actionable
        instead of a shrug. Only the pool can write it: it is alive after its
        workers die, and it is the only thing holding their `Process` objects.
        """
        s = self._s
        if s.failed:
            return
        # detail and reason BEFORE the flag: `failure_reason()` keys off `failed`,
        # so a reader that catches the gap sees a service not yet failed and
        # simply polls again. The other order would briefly advertise a failure
        # with reason UNKNOWN and no text, losing the explanation from the report.
        s.detail = detail.encode("utf-8", "replace")[:DETAIL_FIELD_BYTES - 1]
        s.reason_code = int(code)
        s.failed = 1

    def beat(self) -> None:
        """
        Stamp the pool's liveness tick.

        `time.monotonic()` is CLOCK_MONOTONIC on Linux, which is system-wide, so
        a timestamp written by the pool process is directly comparable in a stage
        worker. A `ServiceStatus` never crosses a machine boundary, which is the
        whole requirement.
        """
        self._s.heartbeat = time.monotonic()

    # ------------------------------------------------------------------
    # any process — read-only
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return bool(self._s.ready)

    def failure_reason(self) -> Optional[str]:
        """
        A human-readable reason if the backend is known to be gone, else None.

        Two independent signals, in priority order:

          1. The pool said so (`failed`). Precise, and it arrives as fast as the
             pool can notice — ≤0.5s for a killed worker, because the supervisor
             polls `Process.is_alive()` in the process that started it.
          2. The pool stopped saying anything (`heartbeat` older than `STALE_S`).
             Covers the pool's own SIGKILL/OOM, which by definition cannot be
             reported cooperatively.

        A status that has never ticked is never stale — otherwise a slow `spawn`
        import or a long model load would be misreported as a dead backend before
        the pool's status thread had run once.
        """
        s = self._s
        if s.failed:
            code = int(s.reason_code)
            text = _REASON_TEXT.get(code, _REASON_TEXT[REASON_UNKNOWN])
            detail = s.detail.decode("utf-8", "replace")
            return f"{text}: {detail}" if detail else text

        hb = float(s.heartbeat)
        if hb and (time.monotonic() - hb) > STALE_S:
            return _REASON_TEXT[REASON_SUPERVISOR_LOST]
        return None

    def raise_if_dead(self, context: str = "") -> None:
        reason = self.failure_reason()
        if reason:
            suffix = f" [{context}]" if context else ""
            raise VLMServiceDown(f"{reason}{suffix}")

    def describe(self) -> str:
        """One-line summary for a startup or shutdown log."""
        s = self._s
        return (
            f"ServiceStatus(ready={bool(s.ready)}, failed={bool(s.failed)}, "
            f"reason={int(s.reason_code)}, heartbeat={float(s.heartbeat):.1f})"
        )


# ---------------------------------------------------------------------------
# Ledger — what a worker tells its pool
# ---------------------------------------------------------------------------


class Ledger:
    """
    The two facts a worker reports upward: "I finished initialising" and "I am done
    with slot i".

    Note how little is here. v1.9.1 kept the whole in-flight record in shared
    memory — a `req_id` and a `stage_name` per slot, in fixed-width byte arrays,
    with a documented write ordering so a torn read could not produce an
    unroutable entry. None of that is needed once the POOL, not the worker, is
    what assigns work (see `WorkerPool._dispatch_loop`): the pool already knows
    which request it handed to which slot, in ordinary Python, in its own process.
    All that has to cross a process boundary is the acknowledgement.

    That inversion also closed two holes by construction rather than by comment:

      - **The checkout gap is gone.** v1.9.1 recorded the checkout immediately
        after `queue.get()` and documented the microseconds in between as a known,
        bounded hole in which a kill would lose the request. The pool now records
        the request *before sending it*, so there is no window at all.
      - **Ledger overflow is unrepresentable.** The pool allocates the slots, so it
        cannot hand out more than `k`. The v1.9.1 "ledger full, dropping the
        recovery record" error path — and the reason `set_inflight` needed a
        boolean return — has nothing left to report.

    Both arrays are single-writer per element, in the strict sense this module
    needs: element `rank` / slot `(rank, i)` is written only by rank's own worker
    process, and only ever read by the pool. `lock=False` for the reason in the
    module docstring — a lock a SIGKILL can leave held would poison the exact path
    that has to work when a worker is SIGKILLed.
    """

    def __init__(self, n_workers: int, k: int):
        self.n_workers = max(1, int(n_workers))
        self.k = max(1, int(k))

        # worker -> pool: "my init_fn returned".
        self.worker_ready = mp.Array(c_int8, self.n_workers, lock=False)
        # worker -> pool: monotonically increasing per slot. The pool bumps its own
        # private dispatch counter when it sends work; the worker bumps this one
        # when it has answered. Equal means the slot is free; unequal means that
        # slot holds a request nobody has answered yet.
        #
        # A counter rather than a flag so a slot can be reused indefinitely with no
        # reset step and no ABA ambiguity: the pool never has to distinguish "not
        # started" from "finished long ago".
        self.done_seq = mp.Array(c_int32, self.n_workers * self.k, lock=False)

    def index(self, rank: int, slot: int) -> int:
        return rank * self.k + slot

    # ------------------------------------------------------------------
    # worker side
    # ------------------------------------------------------------------

    def mark_worker_ready(self, rank: int) -> None:
        """This worker's `init_fn` returned; it can serve requests."""
        if 0 <= rank < self.n_workers:
            self.worker_ready[rank] = 1

    def release(self, rank: int, slot: int) -> None:
        """
        This worker has answered the request in `slot` — its response is already on
        the response queue.

        Called after the `put()`, deliberately. The other order would let the pool
        reuse the slot for a new request while the answer to the old one was still
        in flight, so a kill in between would leave the pool believing the new
        request had been answered.
        """
        if 0 <= rank < self.n_workers and 0 <= slot < self.k:
            self.done_seq[self.index(rank, slot)] += 1

    # ------------------------------------------------------------------
    # pool side
    # ------------------------------------------------------------------

    def done_count(self, rank: int, slot: int) -> int:
        return int(self.done_seq[self.index(rank, slot)])

    def all_ready(self) -> bool:
        return all(self.worker_ready[i] for i in range(self.n_workers))

    def ready_count(self) -> int:
        return sum(1 for i in range(self.n_workers) if self.worker_ready[i])

    def is_ready(self, rank: int) -> bool:
        return bool(self.worker_ready[rank])

    def clear_ready(self, rank: int) -> None:
        """Called by the pool when a rank dies, so a respawn has to earn it back."""
        if 0 <= rank < self.n_workers:
            self.worker_ready[rank] = 0



__all__ = [
    "ServiceStatus",
    "Ledger",
    "VLMServiceDown",
    "STALE_S",
    "HEARTBEAT_INTERVAL_S",
    "MAX_INFLIGHT_HARD_CEILING",
    "REASON_UNKNOWN",
    "REASON_WORKER_EXITED",
    "REASON_ENGINE_FAILED",
    "REASON_SERVER_EXITED",
    "REASON_SERVER_UNREACHABLE",
    "REASON_STARTUP_FAILED",
    "REASON_WORKER_DIED",
    "REASON_SUPERVISOR_LOST",
]
