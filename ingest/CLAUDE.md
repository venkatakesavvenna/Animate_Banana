# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How the Pipeline Actually Works (End-to-End)

### Entry Point: `examples/vllm_testing/main.py`

Everything starts here. `main.py` does three things before the pipeline even begins:

1. **Creates the `ServiceChannel`** (`build_channel([stage_name])`) — `mp.Queue()` for requests, one `mp.Queue()` per stage for responses, `mp.Event()` for stop signaling, and an `extras` dict. This must happen in the parent process before any `spawn` so file descriptors are inherited correctly. As of v1.9.1 this is the *only* IPC bundle: the shared-memory pool is gone (see below), and standalone's `__main__` now plays exactly the part StageWeaver's group leader plays, with the same shape.
2. **Starts the backend** (`model_init(args, logger, channel)`) — two lines: build a `WorkerSpec` and hand it to a `WorkerPool`. The pool spawns one *separate process* per worker and publishes its own `ServiceStatus` on the channel before spawning any of them (v1.9.2 — see "The Backend" below; the v1.9.1 build-attach-publish ordering dance is gone). Workers receive prompts via `request_queue` — through the pool's dispatcher — and send results back via `response_queues[stage_name]`. The stop_event signals shutdown. Critically, **the worker processes are the only things that ever touch the GPU**. The main pipeline process is entirely CPU-bound. Running the VLM in its own process (rather than a thread in the pipeline process, as it originally was) also matters for StageWeaver: with up to 10 stages' worth of CPU-bound stage code sharing one process, keeping the VLM's own Python-side bookkeeping out of that process avoids adding more GIL contention on top. `args.backend` picks the spec builder in `vllm_module/specs.py` and reaches nothing else: **`"online"`** (default in `examples/vllm_testing/main.py`) launches a local `vllm serve` subprocess and drives it over HTTP; **`"async"`** builds an in-process AsyncLLM continuous-batching engine. **`"batch"` was retired in v1.9.2** — it was a batch-boundary `LLM.generate()` A/B timing baseline and cannot be a call-per-request worker (`docs/archive/timed_worker.py.removed`). Full design: `docs/v1.6_changes.md`, `docs/v1.9.2_changes.md`.
3. **Calls `main_cli()`** which optionally ingests image paths into the DB (`ingest_images()`), then calls `cli()`.

On exit (normal or exception), `model_term(pool, channel)` calls `pool.stop()` — which sets the stop event and joins every worker, each running its own `term_fn` in a `finally` — then clears GPU memory and calls `channel.cancel_join_threads()`. Since v1.9.1 this mirrors StageWeaver's `term_fn(init_vars, stop_event)` — with `shm` gone there is no ownership asymmetry left between the two, only "clean up the queues in whichever process created them."

---

### Orchestrator: `src/vision_ingest/drivers/cli.py`

`cli()` is the central coordinator for one pipeline stage. It initializes everything in a strict order and then hands off to `process_pipeline()`.

**Initialization sequence:**
```
MainLogger → DB → Recovery → JSONLWriter → VLMPrediction → PipelineMetrics
```

Key decisions made here:
- **`fetch_state`**: set from `args.fetch_state` (0 for standalone, 4 for downstream stages). This tells the DB which state to fetch from. This is the single knob that makes multi-stage pipelines work — you don't change any other code.
- **`router_coordinator`**: `Optional[RouterCoordinator]` — passed in from the caller (StageWeaver in multi-stage mode, or `None` in standalone). No stub needed; `JSONLWriter` guards both calls with `if self.router_coordinator is not None`. The stageweaver import is `TYPE_CHECKING`-only so the package does not need to be installed in standalone deployments.
- **`stage_name`**: identifies this stage in the response queues and in router_coordinator calls.
- **`channel`** (v1.9.1): the `ServiceChannel` carrying `request_queue` / `response_queues` /
  `stop_event` / `extras`. `cli()` reads two things off it — its own response queue
  (`response_queues[stage_name]`) and `extras["health"]` — so `request_queue`, `response_queue`,
  `shm_name`, `free_slots` and `vlm_health` are all gone from the signature. `ServiceChannel.adopt()`
  accepts StageWeaver's own four-field channel class as well as this repo's.
- **`vlm_health`** (v1.7; reaches StageWeaver only since v1.9.1): read off `channel.extras["health"]`,
  written by the pool. Lets `VLMPrediction` distinguish "the backend died" from "this image
  failed" and abort instead of marking every remaining path FAILED. **Through v1.9 this never
  reached StageWeaver at all** — the locked signatures had no slot for it, so `cli()` silently ran
  with `None` there and both the readiness wait and crash detection were dead code. Build one in
  `init_fn` and publish it; without it only the heuristic consecutive-failure fuse protects the DB.
- **`args.allowed_media_roots`** (v1.7; scope widened in v1.8): the directories images may be read
  from. Becomes vLLM's own `allowed_local_media_path` (carried inside `engine_args`, so it reaches
  both the in-process engine and `vllm serve`) *and* the roots every image path is validated against
  before submission — one arg so the two can never drift. **Required for any vision model on every
  backend** as of v1.8, since the worker now resolves `file://` paths itself everywhere and vLLM
  refuses local files outright when it is unset. vLLM accepts a single directory, so several roots
  are collapsed to their common ancestor for the engine while the exact list stays enforced locally;
  `cli()` logs that widening at startup.

On any exception or shutdown, the `finally` block resets all in-flight paths (state 1 → fetch_state)
back to their original state, closes the writer, shuts down thread pools, and logs a final summary.
The whole block runs inside `_shielded_teardown()`, which ignores SIGINT/SIGTERM until it finishes —
repeated Ctrl+C can no longer interrupt the reset half-way and strand rows in state=1. After
`writer.close()` a second sweep resets `writer.unflushed_paths()` (anything accepted but never
fsynced), which is safe because the commit protocol guarantees an uncommitted record was never
written durably.

---

### Main Loop: `src/vision_ingest/drivers/cli_utils.py:process_pipeline()`

This is the pipeline's heartbeat. As of v1.5, all batching/overlap machinery lives inside `VLMPrediction` (see below) — this loop is a plain sequential `while True` that just sizes fetches and hands paths off:

```
loop:
  fetch fresh_paths = db.fetch_batch(fetch_n)   # fetch_n comes from the previous iteration
  failed, successful, fetch_n = vlm_predictor.run_iteration(fresh_paths)
  db.mark_failed(failed); writer.enqueue(each successful obj)
```

`fetch_n` is computed by the predictor, not the loop — it is `desired_total (3·batch_size) - owned`, where `owned` is an exact conservation counter (incremented when paths are handed to the predictor, decremented when results come back). This is what keeps at most ~3·batch_size DB rows sitting in state=1 at any time, regardless of how the predictor's internal threads are actually pacing work.

**Bootstrap / steady-state / flush collapse into one rule** ("fetch keeps the owned total near `desired_total`; each call to `run_iteration` drains whatever has finished and feeds whatever is new"):
- First iterations: `fetch_n` primes at `2·batch_size` so the engine fills quickly.
- Steady state: `run_iteration` blocks internally until it has a chunky return (see below), so the loop never spins.
- Drain/flush: once `fetch_batch()` returns empty, the loop keeps calling `run_iteration(None)` — draining whatever is still pending/in-flight/in-prep inside the predictor — until nothing remains.
- Multi-stage waiting: exits the sleep-1s-and-retry branch only when fetch is empty **AND** `pending_count() == 0` **AND** `inflight_count() == 0` **AND** `in_prep_count() == 0` — i.e., don't sleep while the predictor still has work streaming back.

The DB fetch loop also handles the **waiting case**: if `fetch_batch()` returns empty but state=0 count is non-zero, it means upstream work is still in flight. The loop sleeps 1 second and retries — this is how downstream stages wait for StageWeaver to activate their rows.

---

### VLM Inference: `src/vision_ingest/modules/prediction.py:VLMPrediction`

`VLMPrediction` streams work against the AsyncLLM continuous-batching worker via **two persistent background threads**, started once in `__init__` and running for the object's whole lifetime — there is no per-call thread pool and no "batch" boundary:

```
paths_in ──► PREP THREAD ─────────────► request_queue / inflight{}
retry_q ──►  (retries first — cheap           │
             re-stage, no prompt prep;        │  the ONLY submitter
             then fresh paths in chunks       ▼
             of ≤ batch_size)            AsyncLLM worker
                                              │
response_queue ◄──────────────────────────────┘
     │
     ▼
COLLECTOR THREAD — drain, validate, classify:
  success / attempts exhausted ─────────────► terminals ("ok"/"fail")
  failed with attempts left ────────────────► retry_q
```

- **Prep thread** (`_prep_loop`): drains `retry_q` first — re-staging a retry is a trivial re-put (v1.8: the payload is paths + text) and lets it rejoin the running engine immediately. Retries go through the project's `build_retry_request()` hook; fresh items skip it entirely. Only when there's nothing to retry does it pull up to `batch_size` fresh paths off `paths_in`, call `get_prompts()` (via `reader_pool`) to build the prompt text and **validate** each image path, then stage and submit each (`stage()` + `submit_token()` from `retry_validate.py`). It inserts into `self.inflight` **before** the `request_queue.put()` so a response can never race back to an unknown `req_id`.
- **Collector thread** (`_collect_loop`): calls `drain_available()` (blocks ≤0.25s for the first response, then greedily drains everything already queued — may return empty). Each response is a `VLLMResponse` already tagged `ok`/`reject`/`infra` by the worker, so the collector only decides whether an `ok` response's *output* passes `wrap_and_validate_output()`: valid → `terminals`; invalid with attempts remaining → bumped attempt count/error, pushed onto `retry_q`; attempts exhausted, or `reject` → terminal failure; `infra` → retried on its own budget, never a terminal failure.
- **`run_iteration(fresh_paths)`** — cli's only entry point, called once per loop pass in `cli_utils.py`. Hands `fresh_paths` to the prep thread instantly (non-blocking `paths_in.put`), then blocks on the `terminals` queue collecting a chunk of finished results. Returns when **any** of: ≥`batch_size` results collected (a chunky return, so the next DB fetch is worthwhile), or nothing is left active (`owned - collected <= 0`), or `MAX_LINGER_S` (5s) has elapsed since the first result of this call (bounds how stale a finished result can get before reaching the writer). Returns `(failed_paths, successful_objs, fetch_n)`.
- **`fetch_n`** is exact by conservation: `self._owned` is incremented by `len(fresh_paths)` on hand-in and decremented by however many terminal results are returned; `fetch_n = desired_total (3·batch_size) - owned`. `pending_count()` / `inflight_count()` read `retry_q`/`terminals` qsize and `len(inflight)` directly; `in_prep_count()` is the derived remainder (`owned - inflight - pending`) rather than tracked separately.

Because submission runs on its own thread, **the GPU is fed independent of how long cli spends in fetch/writer/fsync**, and because the collector streams per-request responses from AsyncLLM (not a per-batch `generate()` call), one slow/hallucinating item never holds up classification or writing of the rest of its cohort — it delays only itself.

**Loose coupling with vLLM**: `VLMPrediction` never imports or calls vLLM directly. It only uses `request_queue` and `response_queue` (via the `stage`/`submit_token`/`drain_available` helpers in `retry_validate.py`). Any inference backend (a plain Hugging Face model, an API call, ...) can replace vLLM as long as it consumes the v1.8 payload and emits a `VLLMResponse` — see `core/wire.py` for the whole contract. **The `"online"` backend is exactly this in practice**: it drives a `vllm serve` HTTP server instead of an in-process engine, changing only what one `call_fn` does.

**v1.9 cashes that in: `backend="function"` runs an ordinary Python function.** The payload is now *opaque* — `prediction.py` touches its contents in exactly two places (it records `send` on the record for the output line, and hands `response.output` to `wrap_and_validate_output()`), and the only structural constraint left is that a payload is a JSON-able dict. Set `function_task` instead of `vlm_config` and `get_prompts()` calls `FunctionTask.build_request(path, logger)` per path instead of the `get_image_specific_prompt` → `get_prompt_with_image` two-step; sampling params resolve to `None`. Everything else — the DB state machine, the commit protocol, recovery, admission control, the retry/infra taxonomy, `PipelineMetrics`, the StageWeaver hooks — is untouched and unaware. See `docs/v1.9_changes.md` and the `core/` package.

**The wire contract (v1.8)** — one shape each way, so no process has to infer anything about the other:

```
request   {"prompt": <raw user text>, "image_paths": [abs, ...], "enable_thinking": bool}
          NOT chat-templated, no decoded pixels. The worker renders it with vLLM's own code
          (renderer.render_chat_async / LLM.chat / POST /v1/chat/completions), so online and
          offline run identical templating rather than two implementations that agree by luck.

response  VLLMResponse(req_id, kind, text, stats, error)   kind ∈ {ok, reject, infra}
          The worker tags it — it is the only thing that knows. `reject` means "this exact input
          is bad and always will be" (terminal, first occurrence); `infra` means "the backend
          could not answer" and must NEVER mark a path FAILED. `stats` is built worker-side, so
          nothing downstream duck-types a response object.
```

This is what removed `_payload_kind`/`assert_payload_kind()`/`BackendPayloadMismatch` (nothing left to mismatch), the `_OnlineFailure` type-name string match, the `.usage` duck-typing, and `post_process_full_vllm_object()`. Full rationale: `docs/v1.8_changes.md`.

---

### The Backend: `src/vision_ingest/core/service.py:WorkerPool`

**One class runs every backend.** vLLM is a `WorkerSpec` (`vllm_module/specs.py`) exactly as a pool of
32 headless-chrome renderers is. v1.9 had `FunctionService` *beside* `VLLMService` — two supervision
stories for one problem — and v1.9.2 deleted the left column along with everything that existed only
to keep the two in sync: two `from_channel`s, a health object named "VLM" that chrome workers
published, a per-`k` ctypes factory with a module `__getattr__` so it could be pickled, per-worker
heartbeats, 30-second SIGKILL detection.

An `init_fn` is two lines, and that is the whole contract:

```python
pool = WorkerPool(vllm_spec(args.backend, engine_args=..., gpus=..., max_inflight=args.batch_size,
                            logs_dir=args.logs_path), channel, logs_dir=args.logs_path)
pool.start()      # publishes its own ServiceStatus on the channel, before spawning anything
```

`term_fn` is `pool.stop()`. Nothing builds, sizes, attaches or publishes a health object any more; the
v1.9.1 five-step dance and the `ValueError` that policed its sizes are gone because a mismatch is no
longer representable.

- **`(rank, device, ctx)` is the abstraction.** `init_fn(args, rank, device) -> ctx` builds one
  worker's resource (a chrome, a SAM3 replica on `device`, a `vllm serve` subprocess, an AsyncLLM
  engine); `call_fn(ctx, payload, params) -> output` does one item; `term_fn(ctx)` tears it down. All
  three must be module-level (they are pickled into a spawned child). `params` is that request's
  inference parameters — a `SamplingParams` for vLLM, `None` for everything else.
- **Sync or async is the loop's only degree of freedom.** `inspect.iscoroutinefunction(call_fn)` picks
  between get-one/answer-one and an event loop holding `max_inflight_per_worker` calls at once. That
  single fact is all vLLM ever needed that a chrome worker did not, which is why the two hand-rolled
  asyncio queue loops (`online_worker._amain`, `async_worker._amain`), their four bridge threads and
  their `_SENTINEL` are one framework loop now. An async `init_fn`/`term_fn` is awaited on the
  worker's own loop — a hard requirement for AsyncLLM, which schedules its output loop on the running
  one.
- **Two replicas of one model are `n_workers=2`**; two different models remain two pools on two
  channels. Each rank gets its own GPU group via `devices` and (online) its own ephemeral port.
- **The pool dispatches; workers do not share the request queue.** A dispatcher thread is the shared
  `request_queue`'s only consumer and hands each request to the least-loaded worker with free capacity
  through a private per-worker inbox. This is not an optimisation — it is the fix for a bug that made
  respawn useless: `mp.Queue.get()` holds the queue's shared reader lock while polling, a process
  SIGKILLed in there never releases it, and every surviving consumer then blocks forever — so the pool
  would recover the dead worker's requests, respawn it, report healthy, and consume nothing again.
  Silent, permanent stall. Only one consumer holds the lock at a time, so the per-kill odds are
  roughly 1/N (measured 3 of 8 trials with two workers — `examples/no_gpu_validation/mp_only.py`),
  which is not a reprieve: the consequence is total and unrecoverable whenever it lands. A killed
  worker can now only poison its own inbox, which the pool discards when it respawns it.
  *Residual, unfixed:* workers share the per-stage response queues, so a kill inside the microseconds
  where a feeder thread holds one's write lock can still wedge the response path. A much smaller
  window than the request side's — microseconds of pipe write per response, against a lock held across
  a 0.25s poll — but the same class of bug; closing it would route every response through the pool
  process.
- **Supervision is `proc.is_alive()` every 0.5s**, in the process that started the workers, which is
  the only place the answer is free. On a death: emit one `infra` response per request that worker
  owed (on the right response queue), then apply `on_worker_death`. Without that recovery those
  req_ids sit in `prediction.py`'s `inflight` dict forever, `_owned` never decrements, `fetch_n` walks
  to 0, and the pipeline stalls in silence.
- **`on_worker_death` is explicit**: `"respawn"` (default, with `max_respawns=3`) for pools of many
  cheap independent workers, where one death is throughput and not the run; `"abort"` for both vLLM
  specs, because a vLLM death is usually an OOM or driver fault that respawning reproduces while
  burning GPU-minutes invisibly. Either way the run ends with in-flight rows at state=1, reset on
  shutdown, nothing marked FAILED.
- **`BackendGone`** (`core/task.py`) is what a worker raises when its *own resource* is dead — the
  third and last thing a worker can discover, next to `InputRejected`. It answers the request in hand
  as `infra`, drains, runs `term_fn` and exits with a distinct status so the pool reports why. It
  replaced the online worker's 5-second `proc.poll()` timer and the async worker's engine-dead flag
  write.
- **The ledger is two acknowledgements, not a shared record.** Because the *pool* assigns work, it
  already knows what each worker owes, in ordinary Python, in its own process. All that crosses a
  boundary is `worker_ready` and a per-slot `done_seq` counter (`core/status.py:Ledger`). v1.9.1's
  fixed-width `req_ids`/`stages` byte arrays, their write-ordering rules, the "microseconds between
  `get()` and the ledger write" hole and the ledger-overflow path all ceased to exist rather than
  being fixed — the record is written before the request is sent, and the pool cannot hand out more
  slots than it has.

---

### Backend Liveness: `src/vision_ingest/core/status.py:ServiceStatus`

One fixed 5-field `ctypes.Structure` in an `mp.Value`, **written only by the pool**, read by anyone.
It is not in `vllm_module/` because nothing in it is about vLLM — a chrome pool published the old
`VLMHealth` too, and the name was a lie in every deployment but the first.

A consumer does not care how many workers are up; it asks two questions, and the pool is the only
process that can answer either (it holds the `Process` handles, knows the exit codes, and decides
whether to respawn or give up). So:

- **`ready`** — cli()'s "wait for the backend, then start the clock", so model-load time stays out of
  the throughput numbers. Set once every initial worker has finished `init_fn`. Latched: a respawn
  dipping capacity is throughput, not a return to "still loading". A rank the pool has permanently
  given up on still counts as resolved, so one broken rank cannot leave cli() blocked forever while
  the other 31 drain the queue.
- **`failed` + `reason_code` + `detail`** — the exact-failure-reason mechanism. A live pool writing
  about its dead workers can say things no flag could: `"a backend worker process died and was not
  replaced: rank 3 exitcode=-9"`, or `"the backend failed to start: all 8 worker(s) dead; last: rank 1
  exitcode=1 (died before init_fn completed); respawn budget (3) exhausted"`. `prediction.py` turns
  this into `VLMServiceDown` instead of marking anything FAILED.
- **`heartbeat`** — the one failure the pool cannot report cooperatively: its own SIGKILL/OOM-kill.
  Stamped ~1s by a thread that does nothing else, so a blocking respawn backoff cannot starve it. Stale
  past `STALE_S` (30s, still unmeasured) means nothing is supervising the workers any more, and
  aborting cleanly is right even if some are still momentarily draining. A status that has never ticked
  is never stale, so a five-minute model load is not a corpse.

Deleted with the per-rank array: per-rank `ready`/`dead`/`reason` flags and their read-time aggregates,
per-worker heartbeats, `attach_process` / `attach_worker_process` (a `Process` handle does not pickle,
so under StageWeaver `cli()` could never use it — that inertness is why the heartbeat had to exist),
the `_RankState_{k}` factory and the module `__getattr__` that made it picklable.

**Deliberately lock-free.** `mp.Value`/`mp.Array`'s default lock is a POSIX semaphore, and a process
SIGKILLed while holding it never releases it — every later reader would block forever, poisoning
exactly the path this file exists to serve. A heartbeat that can be killed by the kill it is watching
for is worse than no heartbeat. Every field has one writer and is a naturally-aligned scalar, so a
reader sees the old value or the new one; `mark_failed` writes `detail` before the flag so a reader
that sees `failed=1` always finds the text.

CPU-only groups publish nothing, and `channel.health is None` still means "no backend to wait for" —
which is why the framework must never pre-create a status object.

---

### Shared IPC: `src/vision_ingest/core/channel.py:ServiceChannel`

`request_queue`, `response_queues` (keyed by stage name), `stop_event`, and an opaque `extras` dict.
Created in the one fd-inheritance-safe window — StageWeaver's group-leader process, or standalone's
`__main__` via `build_channel()` — and pickled into every child at spawn.

**Ordering is the rule that matters**: `init_fn` runs to completion before any stage worker is
spawned, so anything it puts in `extras` (in practice `extras["health"]`) reaches every stage.
Anything added *after* a spawn does not reach the process already spawned — pass it explicitly
instead. Since v1.9.2 this is not something an `init_fn` author can get wrong: `WorkerPool.start()`
publishes its `ServiceStatus` before it spawns a single worker, because the pool is the only thing
that writes it.

`extras` being opaque is what preserves StageWeaver's "imports nothing from VIE" rule: the framework
passes a `dict` along and never needs to know `ServiceStatus` exists. StageWeaver defines its own bare
four-field `ServiceChannel` (it depends on nothing and cannot import ours); `ServiceChannel.adopt()`
wraps whatever `cli()` is handed, so the conveniences live on one side of the boundary only.

> **The `shm` subsystem was deleted in v1.9.1.** It existed because pre-v1.8 offline prompts embedded
> decoded PIL images (3-12 MB each), where the kernel copy through `mp.Queue` dominated everything.
> Since v1.8 a request is paths + text and a response is text + stats, so nothing justified it; v1.9
> already called it vestigial and kept it only because `shm_name`/`free_slots` were part of
> StageWeaver's locked signature, which this release breaks anyway. Design notes are preserved in
> `docs/archive/`. **The exit-hang lesson is still live and still implemented**: every process that
> `put()` to a queue calls `cancel_join_thread()` from an always-runs `finally`, now via
> `channel.cancel_join_threads()` — see `docs/archive/exit_hang_postmortem.md`.

### Writing & Crash Safety: `src/vision_ingest/modules/writer.py:JSONLWriter`

The writer runs on a **dedicated background thread** consuming from an internal `queue.Queue`. The main loop just calls `writer.enqueue(obj)` and moves on immediately — the queue is unbounded by default so it never blocks the pipeline.

The background thread accumulates raw output dicts in `_objs` (and the corresponding paths in `_paths`). When `len(_objs) >= fsync_every_lines` (default 250), it executes the **atomic commit sequence**:

```
0. router_coordinator.get_current_path_taken(paths)  # one SELECT for the batch
   → stamps route_taken: [stage_names…] onto each object in _objs
1. f.write(payload)                          # json.dumps each enriched obj; write to OS buffer
2. f.flush() + os.fsync(f.fileno())          # force to disk (durable)
3. router_coordinator.mark_done(paths, stage_name, successful_objs=objs)  # tell StageWeaver
4. db.mark_done(paths, current_shard_path)   # mark state=2 in DB
```

Steps 0 and 3 are skipped when `router_coordinator is None` (standalone mode). The order of steps 1–4 is non-negotiable. If the process crashes:
- After step 1 but before step 2: recovery will truncate the partial write
- After step 2 but before step 4: recovery sees the fsync boundary as committed; DB will be rewound on next startup
- After step 4: fully committed, never replayed

Shards auto-rotate at 1 GB. The writer opens its own DB connection (separate from the one in `cli.py`) to avoid cross-thread contention.

**`writer.flush()` (since v1.9.3)** — forces the commit sequence above out-of-band, without
waiting for `fsync_every_lines` or `close()`. A batch smaller than `fsync_every_lines` (e.g. a
short trailing run, or a smoke test with a low image count) would otherwise sit buffered
indefinitely, and step 3's downstream `mark_done()` activation with it. `cli_utils.py`'s
`process_pipeline()` calls it on every idle tick (fetch empty, nothing pending/inflight/in-prep)
right before deciding whether to exit or sleep — a no-op if nothing's buffered. Implemented as a
sentinel enqueued on the writer's own queue, handled by the same background thread, so it stays
inside the writer's single-thread-owns-`_objs` invariant.

---

### StageWeaver Integration

`router_coordinator` is `Optional[RouterCoordinator]`. Pass `None` for standalone mode — no stub required. When provided (multi-stage StageWeaver pipeline), `_flush_group` makes two calls per batch:

1. **`get_current_path_taken(paths)`** — one SELECT against `router_items` that decodes each item's `done_mask` into the list of stage names that completed before this one. The result is stamped as `route_taken` on every output object before the JSONL is written, so every record carries its full DAG route.

2. **`mark_done(paths, stage_name, successful_objs=objs)`** — runs routing logic (if any), writes `state=4` into downstream stage DBs, and ORs this stage's bit into `done_mask`. `successful_objs` carries the full per-item output dicts (including `route_taken`) and is available to user-defined `routing_fn` implementations.

The downstream stage is configured with `fetch_state=4`. Its `db.fetch_batch()` only returns rows once StageWeaver has activated them (state 0→4). VIE has zero knowledge of what's upstream or downstream — it processes whatever rows its DB exposes.

**Locked signatures (StageWeaver ≥ 1.1.0, this repo ≥ v1.9.1).** Requires updating both repos
together; the old flat 7/8-argument forms are gone:

```
init_fn(args, logger, channel) -> init_vars
term_fn(init_vars, stop_event) -> None
function(args, channel, shutdown_event, router_coordinator, stage_specific_log_dir) -> None
```

`init_fn` is where a group's backend is started — `WorkerPool(spec, channel).start()` — and the pool
publishes its own `ServiceStatus` from inside `start()`. Two rules still hold, and v1.9.2 made both
of them unmissable rather than something an author has to remember:

- **The framework must not create the status object.** `cli()` blocks in an unbounded loop on
  `is_ready()`, escapable only by `shutdown_event` or `failure_reason()`. A framework-created status
  handed to a group whose `init_fn` starts no workers would hang that stage at startup. Only
  `init_fn` knows whether a group has a backend — a CPU-only stage publishes none, and
  `channel.health is None` is the correct, working answer.
- **Publish before the spawn that reads it.** The channel is pickled into each stage worker at spawn
  time; `init_fn` returns before any of them start, so its `extras` reach all of them. The pool
  publishes before it spawns its *own* workers, which is strictly earlier still.

Also worth knowing: because `init_fn` runs in the group leader, that is where the pool's dispatcher,
`Process` handles, 0.5s liveness polling and ~1s status heartbeat all live. A worker dying is
therefore invisible to the stages (recovered as `infra`, respawned in under a second); the *group
leader* dying is what the `STALE_S` heartbeat check exists for.

Reference implementations: `examples/vllm_testing/main.py` (vLLM, standalone shape),
`examples/html_render/main.py` (32 chrome workers), and in the parent repo
`src/digital_twin/project_specific/*.py` (four-stage DAG, two models, two `vllm serve` processes).

---

### User Customization: `PromptAndValidation`

Users subclass this (example: `examples/vllm_testing/project_specific.py`) and override:

- `get_sampling_params()` — optional override, layered on top of `VLLMConfig.get_sampling_params()` (which merges the model's own `generation_config.json` for temperature/top_p/top_k/min_p/repetition_penalty/max_tokens, exactly like `vllm serve`'s HTTP layer does for any field a request leaves unset — see `docs/vllm_serve_offline_parity.md` and `docs/v1.6_changes.md`'s 2026-07-28 addendum). Return `None` (default — most tasks should do this) to use that generation_config-derived base as-is; a `dict` to override only specific fields (everything else still resolves from generation_config.json); or a full `SamplingParams` for the old full-replacement behavior (only worth it if a task genuinely wants every field pinned regardless of the model's own defaults).
- `get_image_specific_prompt(image_path, logger)` → `(prompt_str, [image_paths])` — build the per-image prompt; can return multiple image paths for multi-image prompts
- `preprocess_vlm_output(raw)` — strip thinking tokens, parse JSON fences, etc. Raise `ValueError` to trigger a retry.
- `validate_output(processed)` — check correctness. Raise `ValueError` to trigger a retry.
- `build_retry_request(sent_prompt, sampling_params, attempt, last_error)` → `(prompt, sampling_params)` — **new in v1.8.** Decides how a retry differs from the attempt before it. The default reproduces what `prediction.py` used to hardcode for every project: append `"Previous attempt failed: ..."` to the prompt text and bump `repetition_penalty` by 0.5 per attempt. Override it to lower temperature instead, switch to a stricter instruction on attempt 2, add a few-shot correction example, or drop the penalty bump entirely — the default reaches ~2.0 by attempt 2, which actively degrades long-form OCR transcription. Both arguments are private copies (`dict(orig_prompt)` and `deepcopy(base_sp)`), never the shared originals, so mutating them in place is safe; `prediction.py` keeps that guarantee precisely because `base_sampling_params` is resolved once and reused for every record forever.
- `write_local_json(image_path, output, logger)` — optional per-image sidecar file; the shared JSONL is always written regardless

`wrap_and_validate_output()` chains preprocess → validate and is called by `VLMPrediction`'s collector thread (`_validate_one`) on every `ok` response, retry or fresh. Do not override it; override the two methods above. Note that as of v1.8 it is the *only* thing `_validate_one` still decides — `reject`/`infra` come tagged from the worker.

---

## Project Overview

**Vision-Ingestion-Engine** is a distributed pipeline for processing massive image datasets (100M–1B+ images) with Vision-Language Models (VLMs) via vLLM. Core design goal: crash-safe, exactly-once processing that can be stopped/restarted at any point with guaranteed data consistency.

## Installation & Setup

```bash
# Install package in editable mode (from repo root)
pip install -e .

# Docker-based setup (recommended for multi-node)
bash bash_scripts/docker_env_setup.sh
```

## Running the Pipeline

### Quick local test
```bash
bash local_testing_pipeline.sh
```

### Production (PostgreSQL, multi-node)

**Step 1 — Start PostgreSQL server (once, on host node):**
```bash
python -m vision_ingest.drivers.db_driver_postgres serve \
  --data-dir /opt/dlami/nvme/<user>/db \
  --pg-host <HOST_IP> --pg-port 5432 --pg-dbname vllm_testing --pg-user pipeline
```

**Step 2 — Run workers (on each node):**
```bash
cd examples/vllm_testing && python main.py
```

`main.py` calls `build_args()` to configure paths, model, batch size, etc.

### Database maintenance
```bash
# Reset stuck paths (state 1→0) after a crash
python -m vision_ingest.drivers.db_driver_postgres reset-stuck \
  --pg-host <HOST> --pg-port 5432 --pg-dbname vllm_testing

# Verify DB consistency and recalculate state_counts
python -m vision_ingest.drivers.db_driver_postgres verify \
  --pg-host <HOST> --pg-port 5432 --pg-dbname vllm_testing

# Both at once (run every 10–20 processing runs)
python -m vision_ingest.drivers.db_driver_postgres full-maintenance \
  --pg-host <HOST> --pg-port 5432 --pg-dbname vllm_testing
```

### Performance benchmarks
```bash
python examples/db_performance_testing/bench_ops_postgres.py
python examples/db_performance_testing/bench_ops_sql.py
```

There is no automated test suite; correctness is validated by running the actual pipeline on small datasets.

## Architecture

### Component Initialization Order (cli.py)
```
MainLogger → Database → Recovery → Writer → VLMPrediction → PipelineMetrics
```
Each component fails fast on initialization errors — no partial-startup state.

### Core Components

| Component | File | Role |
|-----------|------|------|
| Orchestrator | `src/vision_ingest/drivers/cli.py` | Lifecycle management, ties everything together |
| Database | `src/vision_ingest/db/db.py` | Auto-selects Postgres or SQLite backend |
| Recovery | `src/vision_ingest/modules/recovery.py` | Validates JSONL/DB consistency at startup |
| Writer | `src/vision_ingest/modules/writer.py` | Async queue → buffer → fsync → DB.mark_done |
| Prediction | `src/vision_ingest/modules/prediction.py` | Streaming submit/collect against the backend — dedicated prep + collector threads |
| vLLM Specs | `src/vision_ingest/vllm_module/specs.py` | `vllm_online_spec` / `vllm_async_spec` — vLLM as an ordinary `WorkerSpec`; the one place `args.backend` is still interpreted (v1.9.2) |
| Online Backend | `src/vision_ingest/vllm_module/online_worker.py` | `init`/`call`/`term` for a local `vllm serve` driven over HTTP (`backend="online"`); `engine_args`→serve-flag translation. Since v1.9.4, `vllm_bin` (threaded through `vllm_online_spec`) picks WHICH venv's `vllm` executable to launch — an absolute path serves a model from a different vllm/torch build than the one driving the pipeline (e.g. two models needing incompatible CUDA versions on one node), with `_venv_nvidia_lib_dirs()` auto-fixing that venv's `LD_LIBRARY_PATH` so its own pip-installed `nvidia-*` wheel libs are actually found by the bare subprocess. `launch_vllm_serve`/`terminate_vllm_serve` are the same launch/health-check/terminate logic exposed as public functions, for a `backend="function"` stage that needs a native `vllm serve` without the rest of the online `WorkerSpec`'s call-per-request contract (see `digital_twin/project_specific_translation.py`) |
| Wire Contract | `src/vision_ingest/core/wire.py` | The one request/response shape every backend shares — `VLLMRequest`, `VLLMResponse`, `CallResult`, stats builders, chat-message builder (v1.8; moved into `core` in v1.9.2) |
| Async Backend | `src/vision_ingest/vllm_module/async_worker.py` | `init`/`call`/`term` for an in-process AsyncLLM engine, continuous batching (`backend="async"`) |
| Service Channel | `src/vision_ingest/core/channel.py` | The one bundle of shared IPC primitives that crosses a process boundary — queues, stop_event, `extras` (v1.9.1) |
| Service Status | `src/vision_ingest/core/status.py` | The pool's one shared record: readiness, failure reason + detail, heartbeat, plus the worker→pool acknowledgements. Tells "the backend died" apart from "this image failed" (v1.7; rewritten and de-vLLM'd in v1.9.2) |
| Image Validation | `src/vision_ingest/vllm_module/media.py` | Path/pixel checks + safe `file://` encoding, one place for every backend (v1.7) |
| Function Task | `src/vision_ingest/core/task.py` | `PromptAndValidation`'s counterpart for `backend="function"`: `build_request` / `wrap_and_validate_output` / `InputRejected` / `BackendGone` (v1.9) |
| Worker Pool | `src/vision_ingest/core/service.py` | `WorkerSpec` + the ONE backend manager: N worker processes, a dispatcher, and a supervisor that recovers a dead worker's requests and respawns or buries it. Runs vLLM too (v1.9; unified in v1.9.2) |
| Shard Allocator | `src/vision_ingest/core/shards.py` | Single authority on image-output folders; byte-rolls, seals to uncompressed tar, recovers on startup (v1.9) |

### Database State Machine
```
0 (PENDING)        → ready to process
1 (PROCESSING)     → fetched by worker
2 (DONE)           → written to JSONL
3 (FAILED)         → error logged
4 (UPSTREAM-READY) → multi-stage: ready for next stage

On crash: state=1 → state=0 (replayed at next startup)
```

**state=3 means the MODEL or the INPUT was at fault — never the infrastructure.** A dead engine, a
dead `vllm serve`, or a killed worker process raises `VLMServiceDown`, which aborts the run with the
in-flight paths still at state=1 so cli()'s shutdown resets them to their fetch state. This is what
stops one backend crash from marching the whole dataset into FAILED. See `vllm_module/health.py`
and `core/status.py`, `docs/v1.7_changes.md`.

### FSYNC_EVERY_LINES = 250
This constant is the synchronization primitive across all components:
- Writer flushes every 250 images (write → fsync → DB.mark_done)
- Recovery only needs to validate the last 250 lines of the active shard
- At most 250 images are replayed after any crash

### CPU/GPU Parallelism Pattern (prediction.py)
```
Prep thread:      retries (re-stage, no prep) first, then fresh paths in
                   chunks of ≤ batch_size — prep + stage + submit, continuously.
Collector thread:  drain whatever responses have streamed back, validate,
                   classify (terminal / retry), continuously.
```
Two persistent daemon threads started once in `__init__`, not a per-batch thread pool — the GPU (AsyncLLM engine) is fed as long as either thread has work, independent of how long `cli` spends fetching/writing/fsyncing. Because the worker streams per-request (not per-batch `generate()` calls), there is no batch boundary for a straggler to stall behind.

### Write-Fsync-Commit Protocol (writer.py)
The commit sequence per flush group: `get_current_path_taken() → enrich objs → write() → fsync() → router.mark_done(successful_objs) → DB.mark_done()`  
The first and fourth steps are skipped when `router_coordinator` is `None`. If the process crashes at any point, recovery truncates the JSONL to the last fully committed boundary. JSONL and DB always remain synchronized.

### User Customization Interface
Users extend `PromptAndValidation` (base in `src/vision_ingest/project_specific.py`, example in `examples/vllm_testing/project_specific.py`) to define:
- `get_sampling_params()` — VLM inference parameters; see the fuller `None`/`dict`/`SamplingParams` contract under "User Customization: `PromptAndValidation`" above
- `get_image_specific_prompt(path)` — per-image prompt construction
- `preprocess_vlm_output(raw)` / `validate_output(processed)` — output cleaning and validation
- `build_retry_request(sent_prompt, sampling_params, attempt, last_error)` — how a retry differs from
  the attempt before it (v1.8). Default appends the error to the prompt and bumps
  `repetition_penalty` by 0.5/attempt, exactly as `prediction.py` used to hardcode. Override it to
  lower temperature instead, change the instruction on attempt 2, or drop the penalty bump — that
  default reaches ~2.0 by attempt 2, which actively degrades long-form OCR. Both arguments are
  private copies, so mutating them in place is safe.
- `write_local_json(obj, logger)` — optional per-image JSON output. `obj` is the full output
  record (`path` / `prompt_str` / `vlm_output` / `full_vlm_object`). Exceptions are logged
  (v1.7) rather than vanishing into an unread Future, but they never fail the image — the
  shared JSONL is the real output.

## Output Structure

```
{jsonl_output_path}/{hostname}/
├── results_00000.jsonl   # Auto-rotates at 1GB
├── results_00001.jsonl
└── ...

{logs_path}/{hostname}/{timestamp}/
├── main.log              # Orchestrator events
├── vlm_prediction.log    # Inference timing + metrics
├── recovery.log          # Startup recovery details
├── jsonl_writer.log      # Write/fsync/DB timing
├── worker_pool.log       # WorkerPool: spawns, dispatch, deaths, respawns (v1.9.2)
└── worker_0.log          # one per rank: init_fn, per-request failures, term_fn (v1.9.2)
```

Each JSONL line: `{"path": "...", "prompt_str": "...", "vlm_output": "...", "full_vlm_object": {...}, "route_taken": ["stage0", ...], "shard_path": "results_00000.jsonl"}`

**The JSONL is always the output — every backend, every job (v1.9).** Images are *additional*, and only images go into folders-then-tars. A `backend="function"` job may produce both: the bytes land in a shard and the JSONL record says which shard and which member. A function record carries `sent_payload` (with `shard`/`member`) in place of a meaningful `prompt_str`; the vLLM record is unchanged, key for key. Nothing about the JSONL mechanism, its rotation, its fsync boundary or its recovery branches on any of this.

```
{image_output_path}/{hostname}/
├── image_folders/shard_00007/a.png   # what the worker writes — KEPT by default
└── shards/
    ├── shard_00007.tar               # the packed shard (uncompressed)
    └── shard_00007.csv               # byte-offset index + the seal's completion marker
```

`shards/*.csv` carries `image_name_in_shard, original_image_path, img_tar_shard_path, byte_offset, byte_length`, where the offset/length span the file's **content only** — so a consumer can seek (or S3 Range-read) one image straight out of the tar with no untarring. Its presence also marks the shard as fully sealed, which is what keeps startup cheap when folders are retained.

**Sealed folders are retained by default** (`delete_folders_after_seal=False` / `args.delete_shard_folders`), so every image exists twice — a deliberate ~2x disk cost that buys directly-readable files, the ability to rebuild a bad tar, and a folder path that stays resolvable forever. Turn it on to reclaim the space.

`route_taken` lists the stages that completed before the current one. It is `[]` for root-stage items and omitted entirely in standalone mode (`router_coordinator=None`).

`full_vlm_object` (built worker-side by `wire.stats_from_usage()` / `wire.stats_from_request_output()`, which produce key-identical dicts) carries per-item diagnostics: `prompt_tokens`, `completion_tokens`, `mm_tokens` (per-modality breakdown, online backend only — see `docs/v1.6_changes.md`'s 2026-07-28 addendum), `finish_reason`/`stop_reason` (check for `"length"` to catch silently truncated output), and `attempts` (retries consumed before this success). The same per-item values also feed `PipelineMetrics`' running `token_metrics` averages, logged at every checkpoint.

### Input Validation and Failure Classification (v1.7)

Every image path is validated once, in `VLLMConfig.get_prompt_with_image` via
`vllm_module/media.py`, before any backend sees it: absolute, UTF-8-encodable, exists, regular
file, non-empty, inside `allowed_media_roots`, header-decodable, and within the pixel ceiling.
Failures name the reason and the `repr()` of the path. As of v1.8 the header read is
**unconditional** rather than online-only: the stage decodes no images on any backend now, so this
is the only local chance to reject a corrupt or oversized file before it becomes an opaque
worker-side failure.

Two kinds of problem are handled very differently:

- **Per-image** (missing / corrupt / oversized file) → that one image fails terminally, with the
  reason logged. The run continues.
- **Per-configuration** (`is_llm: true` on a model being sent images, more images than
  `limit_mm_per_prompt` allows) → `PromptConfigError`, which aborts the run. These affect every image
  equally, so failing them one at a time would quietly march the whole dataset into state=3. v1.8
  adds a startup check in the same spirit: a sampling-param/backend combination the backend cannot
  honour (`structured_outputs` on `backend="online"`) is refused before a single row is fetched.
  `BackendPayloadMismatch` is gone — there is one payload shape now, so it cannot happen.

`file://` URLs are percent-encoded (`media.to_file_url`). Interpolating a raw path is not safe: vLLM
parses with urllib3 and un-quotes with `url2pathname`, so `#`/`?` truncate the path and `%` decodes
into a different one.

**Image resolution:** the hard ceiling (`VLLM_MAX_IMAGE_PIXELS`, default ~179M px) is enforced on
every backend, not just online. Separately, the model's own HF image processor downscales above its
own budget (chandra-ocr-2: `size.longest_edge = 16777216` **pixels of area**) — that is part of the
model and applies identically online and offline. It cannot be disabled, but it is logged at startup
as `processor_downscales_above_pixels` so the quality trade-off is never invisible.

**`async_scheduling` defaults to `False`** (`vllm_config.DEFAULT_ASYNC_SCHEDULING`) on all backends.
vLLM ≥ 0.25 enables it when unset; it is pinned off here unless a model yaml explicitly opts in.

## Important Deployment Constraints

- **Use PostgreSQL for multi-node**: SQLite has lock contention on network filesystems (FSx Lustre).
- **Keep logs on local NVMe** (`/opt/dlami/nvme/`), not FSx — `os.fsync()` on FSx is slow.
- **Pre-generate image lists**: Use `find` to create `images.txt` once; `os.walk` on FSx for 100M+ images takes hours.
- **JSONL shards are immutable** after fsync — never edit them; recovery assumes they are.
- **Multi-stage pipelines**: Each stage has its own database; stage N writes `state=4` to stage N+1's DB; stage N+1 fetches from `state=4` instead of `state=0`.
