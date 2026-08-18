"""
`ServiceChannel` — the one bundle of shared IPC primitives that crosses a
process boundary (v1.9.1 §4.2).

The problem
-----------
Through v1.9 every shared primitive was threaded by hand through three locked,
flat, positional signatures:

    init_fn(args, logger, request_queue, response_queues, stop_event,
            shm_name, free_slots) -> init_vars
    term_fn(init_vars, stop_event) -> None
    function(args, request_queue, response_queue, shutdown_event,
             router_coordinator, stage_specific_log_dir, shm_name, free_slots)

Eight arguments, with no slot for `vlm_health` and two slots (`shm_name`,
`free_slots`) carrying a subsystem v1.8 already marked vestigial. That is why
`vlm_health` never reached StageWeaver at all: there was nowhere to put it, so
`cli()` ran with `vlm_health=None` under StageWeaver from v1.7 onward —
silently, because every consumer treats it as optional. Both the readiness wait
and crash detection were dead code in that deployment
(`docs/v1.7_stageweaver_gap.md`). And every *future* shared primitive would cost
the same three-signature migration across two repos.

The fix
-------
One object carries the primitives; new ones go in `extras` instead of into a
signature:

    init_fn(args, logger, channel: ServiceChannel) -> init_vars
    term_fn(init_vars, stop_event) -> None
    function(args, channel: ServiceChannel, router_coordinator,
             stage_specific_log_dir) -> None

Ordering is what makes `extras` work. The channel is created in the
fd-inheritance-safe window (StageWeaver's group-leader process, or standalone's
`__main__`) and `init_fn` populates `channel.extras["health"]` *before* any stage
worker is spawned — so every worker inherits the already-populated channel. The
shared primitives inside (`mp.Queue`, `mp.Event`, `ServiceStatus`'s `mp.Value`)
survive pickling across `spawn`; verified two levels deep, main -> group leader ->
leaf worker.

`extras` is deliberately an opaque `dict`, which is what preserves StageWeaver's
"imports nothing from `vision_ingest`" rule: the framework creates the channel and
passes it along without ever needing to know `ServiceStatus` exists.

Who publishes the status (v1.9.2)
---------------------------------
`WorkerPool.start()` does, via `set_health()`. `init_fn` no longer builds, sizes
or publishes anything — it constructs a pool and starts it, and the pool owns the
one object it is the only writer of.

What has NOT changed is why the *framework* must not create it. `cli()` blocks in
an unbounded loop on `is_ready()`, escapable only by `shutdown_event` or
`failure_reason()`. A framework-created status handed to a group whose `init_fn`
starts no workers would hang that stage at startup instead of running it. The
framework cannot know which groups have a backend — only `init_fn` does, and a
CPU-only group publishes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ServiceChannel:
    """
    The shared primitives one init_source group's processes have in common.

    Attributes:
        request_queue:   Shared mp.Queue — every stage in the group puts
                         requests here; the backend is the only consumer.
        response_queues: stage_name -> that stage's private mp.Queue. The whole
                         dict travels, not one queue, because the backend routes
                         by `stage_name` and `cli()` looks up its own.
        stop_event:      mp.Event — signals the backend process(es) to stop.
        extras:          Anything else the group's own code needs to share, by
                         convention keyed by a short string. `"health"` is the
                         only key `vision_ingest` itself reads.
    """

    request_queue: Any
    response_queues: Dict[str, Any]
    stop_event: Any
    extras: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Interop with StageWeaver's own channel
    # ------------------------------------------------------------------

    @classmethod
    def adopt(cls, channel) -> "ServiceChannel":
        """
        Return `channel` as a `ServiceChannel` of THIS class.

        StageWeaver defines its own channel type — it depends on nothing, so it
        cannot import this one — carrying the same four fields and no
        conveniences. Rather than duplicating the conveniences on both sides of
        that dependency boundary, where they would quietly drift, `cli()` adopts
        whatever it is given: already one of ours, use it; otherwise wrap the
        same primitives, sharing `extras` BY REFERENCE so nothing is copied and
        a later `set_health` is still visible through either object.

        This is the whole compatibility layer, and it is why StageWeaver's class
        can stay a bare four-field dataclass.
        """
        if isinstance(channel, cls):
            return channel
        try:
            return cls(
                request_queue=channel.request_queue,
                response_queues=channel.response_queues,
                stop_event=channel.stop_event,
                extras=channel.extras,
            )
        except AttributeError as e:
            raise TypeError(
                f"cli() expects a ServiceChannel (or anything carrying "
                f"request_queue / response_queues / stop_event / extras); got "
                f"{type(channel).__name__}: {e}"
            ) from None

    # ------------------------------------------------------------------
    # Convenience — thin, so nothing has to remember the extras key
    # ------------------------------------------------------------------

    @property
    def health(self):
        """The group's `ServiceStatus`, or None if nothing published one (a stage
        with no backend of its own — a CPU-only stage, say).

        Keeps the key and the property name it had in v1.9.1 even though the object
        behind them changed: `is_ready()` / `failure_reason()` are the only two
        things any consumer calls, and both mean exactly what they meant before."""
        return self.extras.get("health")

    def set_health(self, status) -> None:
        """Publish the status object. Called by `WorkerPool.start()`, which runs
        inside `init_fn` and therefore before any stage worker is spawned — a
        channel pickled earlier carries a snapshot of `extras` as it was at that
        moment."""
        self.extras["health"] = status

    def response_queue(self, stage_name: str):
        """This stage's own response queue."""
        try:
            return self.response_queues[stage_name]
        except KeyError:
            raise KeyError(
                f"no response queue for stage {stage_name!r}; the channel knows "
                f"about {sorted(self.response_queues)}. The stage_name passed to "
                "cli() must match the key its response queue was created under."
            ) from None

    def cancel_join_threads(self, *, include_responses: bool = True) -> None:
        """
        Opt out of the exit-time flush-and-join for every queue in this channel.

        A queue with unread items leaves its writer's feeder thread blocked on a
        full pipe with no reader, hanging interpreter exit — and taking any
        `join()` on that process with it. Cancellation is process-local, so this
        must be called in each process that put() to these queues, from a
        `finally` that always runs. Idempotent and always safe.
        See `docs/v1.9.1_changes.md` §4.4 and the exit-hang history behind it.
        """
        queues = [self.request_queue]
        if include_responses:
            queues.extend(self.response_queues.values())
        for q in queues:
            try:
                q.cancel_join_thread()
            except Exception:
                pass

    def describe(self) -> str:
        """One-line summary for a startup log."""
        return (
            f"ServiceChannel(stages={sorted(self.response_queues)}, "
            f"extras={sorted(self.extras)})"
        )


def build_channel(stage_names, mp_ctx=None) -> ServiceChannel:
    """
    Create a channel and everything in it, in one call.

    MUST be called in the process that will spawn the backend and the stage
    workers — under the `spawn` start method that is the only window where file
    descriptors are inherited correctly. StageWeaver calls this inside the group
    leader; a standalone `main.py` calls it in `__main__`.

    `extras` starts empty; `init_fn` fills it in before anything is spawned.
    """
    import multiprocessing as mp

    ctx = mp_ctx or mp
    return ServiceChannel(
        request_queue=ctx.Queue(),
        response_queues={name: ctx.Queue() for name in stage_names},
        stop_event=ctx.Event(),
    )


__all__ = ["ServiceChannel", "build_channel"]
