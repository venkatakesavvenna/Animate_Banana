"""
v1.9 — the backend-agnostic half of the pipeline.

Everything in here is NEW code, added rather than migrated: `core` exists
because a function backend needs three things the vLLM path never did — a task
object that builds an opaque request (`task.py`), a pool of N worker processes
each holding its own resource (`service.py`), and somewhere for image bytes to
go (`shards.py`).

v1.9.2 finished the job: `service.py` now runs EVERY backend, including vLLM
(`vllm_module/specs.py` is a pair of `WorkerSpec` builders), and `status.py`
carries the liveness record that `vllm_module/health.py` used to — under a name
that is not a lie in the 31-of-32 deployments that are not a VLM.

`prediction.py`, `cli.py`, `writer.py` and `recovery.py` stay exactly where they
were; `core` is imported by them, never the other way around. That direction is
now unbroken: v1.9 had to apologise for one exception, the wire types every
backend emits, which lived in `vllm_module/wire.py` and so were imported *up* into
`core`. They are `core/wire.py` now, for the same reason `status.py` is here —
nothing about them was ever vLLM's.

Importing this package must never require vLLM. See `vllm_module/vllm.py` for
the graceful-degradation import that makes that true for the rest of the chain.
"""

from .channel import ServiceChannel, build_channel
from .task import BackendGone, FunctionTask, InputRejected
from .service import ON_DEATH_ABORT, ON_DEATH_RESPAWN, WorkerPool, WorkerSpec
from .shards import ShardAllocator, ShardReservation
from .status import ServiceStatus, VLMServiceDown

__all__ = [
    "ServiceChannel",
    "build_channel",
    "FunctionTask",
    "InputRejected",
    "BackendGone",
    "WorkerPool",
    "WorkerSpec",
    "ON_DEATH_RESPAWN",
    "ON_DEATH_ABORT",
    "ServiceStatus",
    "VLMServiceDown",
    "ShardAllocator",
    "ShardReservation",
]
