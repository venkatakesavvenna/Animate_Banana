"""
`FunctionTask` — the project-specific half of a function backend (v1.9 §2).

This is `PromptAndValidation`'s counterpart for `backend="function"`. It
deliberately reuses the method names `prediction.py` already calls, so the
predictor's own code is identical for both backends: the only branch anywhere in
`prediction.py` is `get_prompts()` calling `build_request()` instead of the
`get_image_specific_prompt` -> `get_prompt_with_image` two-step.

The contract, in one place:

    build_request(path, logger) -> dict | None
        Build the opaque payload for one input path. A payload is ALWAYS a
        JSON-able dict — that is the only structural constraint v1.9 imposes on
        it, and it exists because the payload crosses an mp.Queue and is echoed
        into the JSONL. Return None to fail this one path terminally (the same
        meaning a None prompt has on the vLLM path).

        Runs on the reader_pool, in the MAIN process. This is where a task that
        produces image output calls `self.reserve(name)` — folder assignment is
        a main-process decision, never a worker's (see shards.py).

    wrap_and_validate_output(output)
        `output` is whatever the worker's `call_fn` returned. Raise ValueError to
        retry on the validation budget, exactly as a bad VLM answer does. Return
        the value to record in the JSONL's `vlm_output`.

    build_retry_request(payload, params, attempt, last_error)
        Unchanged from v1.8 and inherited for free — it already took the payload
        as a dict and already received private copies. `params` is None for
        function backends.

    write_local_json(obj, logger)
        Optional per-item sidecar. The shared JSONL is written regardless.

Failure classification is the worker's job, not this file's — see
`core/service.py`. `InputRejected` lives here only because a task's `call_fn`
is the thing that raises it.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


class InputRejected(Exception):
    """
    Raised by a worker's `call_fn` to mean: THIS input is bad and always will be.

    Maps onto the v1.8 `reject` kind — terminal on first occurrence, never
    retried, because re-running the identical call cannot change the answer.
    "This HTML is unrenderable", "this file is not a PDF", "this mask is empty".

    Anything else a `call_fn` raises is `infra` instead (a chrome crash, a CUDA
    OOM), which is retried on the infra budget and NEVER marks a path FAILED.
    That default bias is deliberate: infrastructure is the far likelier cause of
    an unexpected exception, and guessing wrong in the other direction is what
    marches a dataset into state=3.
    """


class BackendGone(RuntimeError):
    """
    Raised by a worker's `init_fn` or `call_fn` to mean: MY RESOURCE is dead, and
    this worker process cannot serve any more requests (v1.9.2).

    The third and last thing a worker can discover, and the counterpart of
    `InputRejected`: one says the input is dead, an ordinary exception says the
    call failed, and this says the worker is dead. `vllm serve` exited, the
    AsyncLLM engine core died, chrome's WebDriver session is unrecoverable.

    The worker responds `infra` for the request in hand (so that item is retried,
    never marked FAILED), stops accepting new work, drains whatever it is already
    holding, runs `term_fn`, and exits. The pool then applies
    `WorkerSpec.on_worker_death` — respawning a fresh replica, or ending the run
    cleanly with in-flight rows reset.

    This replaces the per-backend death plumbing v1.9.1 needed: a 5-second
    `proc.poll()` timer in the online worker's accept loop, an
    `_is_engine_dead_error()` check writing a shared flag in the async worker, and
    a `REASON_*` code for each. A worker that knows its resource is gone says so
    once, in the language of the framework, and the framework does the rest.
    """


class FunctionTask:
    """
    Base class for a `backend="function"` task. Subclass and override
    `build_request` / `wrap_and_validate_output` at minimum.

    A task opts into image-output shard allocation simply by calling
    `self.reserve(...)` inside `build_request()` and putting the returned path
    into the payload. Nothing else in the pipeline branches on it — a task that
    never reserves is a plain text-output task and its JSONL is written exactly
    the same way.
    """

    # Mirrors PromptAndValidation.enable_thinking, which prediction.py reads via
    # getattr. Meaningless for a function backend; defined so the attribute
    # exists rather than relying on the getattr default.
    enable_thinking = False

    def __init__(self):
        self._shard_allocator = None

    # ------------------------------------------------------------------
    # Shard allocation (see core/shards.py). Wired by cli(); a task that
    # produces no image output never touches any of this.
    # ------------------------------------------------------------------

    def attach_shard_allocator(self, allocator) -> None:
        """Called once by cli() during startup. The allocator is the single
        authority on folder assignment for this process."""
        self._shard_allocator = allocator

    @property
    def shard_allocator(self):
        return self._shard_allocator

    def reserve(self, input_path: str, name: str) -> Tuple[str, dict]:
        """
        Reserve one output slot for `input_path`, to be written under the member
        name `name` (e.g. "a.png").

        Returns `(abs_output_path, record)` where record is
        `{"shard": "shard_00007.tar", "member": "a.png"}` — naming the FINAL TAR
        from the moment of reservation, not the folder the worker actually
        writes into. That is what keeps a JSONL line correct both before and
        after its shard seals, and therefore what keeps "JSONL shards are
        immutable after fsync" true.

        Every reservation must resolve: either the item succeeds and the
        allocator is told how many bytes were written, or it fails terminally
        and the reservation is released. cli_utils does both automatically from
        the terminal results — a task never has to.
        """
        if self._shard_allocator is None:
            raise RuntimeError(
                "build_request() called reserve() but no ShardAllocator is attached. "
                "Set args.image_output_path so cli() builds one for this run."
            )
        return self._shard_allocator.reserve(input_path, name)

    # ------------------------------------------------------------------
    # The four hooks prediction.py actually calls
    # ------------------------------------------------------------------

    def build_request(self, path: str, logger) -> Optional[dict]:
        """
        Build the opaque payload for one input path, or return None to fail this
        path terminally.

        Runs on the reader_pool (main process). Keep it cheap and side-effect
        free apart from `reserve()`: it is on the prep thread's critical path,
        and a path that never reaches a worker must not have left anything
        behind.

        The convention the shipped example follows — and the one `core/service.py`
        documents for `call_fn` — is:

            {"input_path": "/in/a.html", "output_path": "/out/<host>/shard_00007/a.png",
             "shard": "shard_00007.tar", "member": "a.png"}

        but only `input_path` is load-bearing for the framework; the rest is
        between your `build_request` and your `call_fn`.
        """
        raise NotImplementedError

    def wrap_and_validate_output(self, output):
        """
        Decide whether the worker's output is GOOD. Raise ValueError to retry on
        the validation budget (`max_attempts`), exactly like a failed
        `validate_output()` on the vLLM path.

        Default: accept whatever came back. Override to reject a zero-byte
        render, an empty mask set, a truncated document.
        """
        return output

    def build_retry_request(self, sent_payload: dict, params, attempt: int, last_error: str):
        """
        How a retry differs from the attempt before it. Both arguments are
        private copies (`dict(payload)` and a deepcopy of `params`), so mutating
        them in place is safe — `prediction.py` owns that guarantee.

        Default: resend the identical payload. That is the right default here,
        unlike on the vLLM path: a function is usually deterministic, so the
        thing worth varying is generally nothing at all, and the retry exists to
        ride out a transient (a chrome tab that hung, a GPU that was briefly
        out of memory). Override it if your task has a cheaper/stricter second
        mode worth falling back to.

        Note `params` is None for function backends — sampling params are a vLLM
        concept and `prediction.py` short-circuits their resolution entirely.
        """
        return sent_payload, params

    def write_local_json(self, obj, logger) -> None:
        """Optional per-item sidecar file. The shared JSONL is the real output
        and is written regardless; exceptions here are logged, never fatal."""
        pass

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def member_name(input_path: str, ext: str) -> str:
        """
        Default member name for an input path: its basename with `ext`
        substituted (`/in/deep/a.html` -> `a.png`).

        Member names must be unique WITHIN a shard folder. Basenames usually are
        not unique across a 100M-file dataset, so a task whose inputs can collide
        should build something derived from the full path instead (a hash, or the
        relative path with separators replaced). The allocator does not
        de-duplicate — it hands out folders, not names.
        """
        stem = os.path.splitext(os.path.basename(input_path))[0]
        return f"{stem}{ext}"
