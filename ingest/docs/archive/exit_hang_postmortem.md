# Postmortem: Pipeline Hangs on Exit, Needs Multiple Ctrl+C

## Symptom

The pipeline finishes all work, prints its final log lines, and then just sits
there. `Ctrl+C` doesn't exit cleanly either — it takes several presses, and
what finally comes out is a pile of `KeyboardInterrupt` tracebacks instead of
a normal shutdown:

```
------ Pipeline process ended due to complete. Please wait before exiting ------
...
--- Closing Shm connection in VLLMService ---
--- Successfully slosed Shm connection in VLLMService ---

^CProcess VLLMService-1:
Traceback (most recent call last):
  File ".../multiprocessing/process.py", line 317, in _bootstrap
    util._exit_function()
  File ".../multiprocessing/util.py", line 363, in _exit_function
    _run_finalizers()
  File ".../multiprocessing/util.py", line 303, in _run_finalizers
    finalizer()
  File ".../multiprocessing/util.py", line 227, in __call__
    res = self._callback(*self._args, **self._kwargs)
  File ".../multiprocessing/queues.py", line 219, in _finalize_join
    thread.join()
  File ".../threading.py", line 1147, in join
    self._wait_for_tstate_lock()
KeyboardInterrupt

Traceback (most recent call last):
  File ".../main.py", line 259, in <module>
    model_term(vllm_proc, stop_event, shm)
  File ".../main.py", line 150, in model_term
    vllm_proc.join()
  File ".../multiprocessing/popen_fork.py", line 27, in poll
    pid, sts = os.waitpid(self.pid, flag)
KeyboardInterrupt
```

...followed by a resource-tracker warning about leaked shared-memory objects
(a side effect of the forced kill, not the root cause).

The confusing part: the two print statements at the top of the log
(`Closing Shm connection` / `Successfully s[c]losed Shm connection`) are the
**last two lines of `VLLMService.run()`**. Seeing both printed proves
`_worker()` returned and `shm.close()` ran. It looks like the child process
is completely done — so why is `model_term()`'s `vllm_proc.join()` still
blocked?

## Root Cause

**`run()` returning is not the same as the OS process exiting.**

After `run()` returns, `multiprocessing.Process._bootstrap()` still runs the
module's own `atexit`-registered cleanup (`multiprocessing.util._exit_function`
→ `_run_finalizers()`). Among those finalizers: for *every* `mp.Queue` a
process has ever called `.put()` on, that queue registered a finalizer in
this process to join its background **feeder thread** — the thread that
asynchronously drains `Queue`'s internal buffer into the underlying OS pipe
via `os.write()`.

Concretely, at process-exit time, for each such queue:

1. `_finalize_close` (higher priority, runs first) appends a sentinel to the
   queue's internal buffer and wakes the feeder thread.
2. The feeder thread drains whatever's left in the buffer, writing it to the
   pipe with `os.write()`, then exits its loop once it hits the sentinel.
3. `_finalize_join` (runs second) does `thread.join()` — waits for step 2 to
   actually finish.

This is fast and invisible almost all the time. It becomes a **hang** if
`os.write()` in step 2 blocks — which happens when the pipe's kernel buffer
(a few tens of KB on Linux) is already full of bytes nobody is reading. This
is the well-documented stdlib gotcha ("Joining processes that use queues" in
the `multiprocessing` programming guidelines):

> If a child process has put items on a queue ... then that process will not
> terminate until all buffered items have been flushed to the pipe. This
> means that if you try joining that process you may get a deadlock unless
> you are sure that all items which have been put on the queue have been
> consumed.

Both processes in the traceback above are stuck in exactly this path —
`VLLMService`'s own exit finalizer (`Process-1` traceback), and the parent's
`vllm_proc.join()` waiting for that exit to complete.

## Why This Pipeline Hits It: `free_slots`

`request_queue` and each `response_queue` are drained to empty by design —
every request eventually gets exactly one response, and `generate_with_validation()`
/ `_collect()` block until every submitted `req_id` has been retrieved. By the
time the pipeline reaches its final flush, those queues are logically empty.

`free_slots` (see [`shm.py`](shm.py)) is different **on purpose** — it's a
free-list pool, not a request/response channel:

- `Shm.acquire()` → `free_slots.get()`
- `Shm.release()` → `free_slots.put()`

Both `VLLMService._worker()` (releasing request slots, acquiring/releasing
response slots) **and** the main process (`retry_validate.py`'s `_submit()` /
`_collect()`) call `.get()`/`.put()` on this *same* queue throughout the run.
Its steady-state occupancy is "most of the pool," not empty — by design,
nobody needs to drain it, because unused slots are supposed to just sit there
until acquired again.

Once processing ends, nothing calls `.get()` on `free_slots` anymore — there's
no more work to acquire a slot for. Whatever was written to the pipe but never
read (or was buffered but not yet flushed) just sits in the kernel pipe
buffer. With `n_slots` in the hundreds (`total_size // slot_size`, e.g.
10 GB / 10 MB = 1024 slots), it's entirely plausible for the last few
`.put()` calls — made by whichever process happens to release a slot last —
to land on an already-full pipe buffer with no reader left to drain it. That
write blocks forever inside the feeder thread, and that's the thread
`_finalize_join` is waiting on.

## The Fix

`Queue.cancel_join_thread()` opts a process out of the flush-and-join step at
exit for a given queue. It's safe to call once that process has no more data
it needs delivered through that queue — which is exactly the situation here:
by the time `VLLMService.run()`'s `finally` block executes, `_worker()` is
done producing, so nothing further needs to reach `free_slots` or
`response_queues[stage_name]` reliably.

```python
# VLLMService.run(), after self.shm.close():
self.free_slots.cancel_join_thread()
for q in self.response_queues.values():
    q.cancel_join_thread()
```

The parent process has the identical exposure — it also calls `.put()` on
`request_queue` (via `submit()`) and `free_slots` (via `release()`) throughout
the run, so it needs the same treatment on its own exit path, in
`model_term()` after `shm.cleanup()`:

```python
# main.py, model_term(), after shm.cleanup():
request_queue.cancel_join_thread()
free_slots.cancel_join_thread()
for q in response_queues.values():
    q.cancel_join_thread()
```

## General Lesson

Any `mp.Queue` used as a **pool / free-list** (put-and-get from multiple
processes, steady-state non-empty, no natural "drained to zero" point) is a
latent exit-hang risk — unlike a queue used as a **channel** (every put is
matched by exactly one get, drained to empty by construction). If a future
change adds another such pool-style queue, it needs `cancel_join_thread()` on
every process that writes to it, called once that process is done producing.
