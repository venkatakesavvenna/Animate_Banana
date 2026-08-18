# Changelog

All notable changes to the Vision Ingestion Engine.

## [TODO]
- **v1.0 Post-Release**
  - Main README.md and QUICKSTART.md updates for v1.0 changes (PostgreSQL support, backend selection)
  - CLI testing needs to be done (database testing is complete)
  - Comprehensive backward compatibility testing:
    - Single-stage pipelines with SQLite and PostgreSQL
    - Multi-stage pipelines with mixed backends
    - Existing SQLite DBs with new unified interface
    - Cross-database migration scenarios
- change in toml file
- ⚠️ There is some issue in `os.path.getsize` in `db.get_db_health()`. Its returning wrong size.
- Main logger.shutdown() should log all the metrics.. including fetch_metrics, etc.
- image_lifecycle in README.md

#### ⚠️ NOTE on v1.5.0 (superseded — see the 1.6.0 cleanup entry below):-

By default, we ship with AsyncLLM (`backend="async"` or `"online"`) and shm disabled. If you want
to use shm, set `use_shm=True` in `build_args()` in your `main.py` (threaded down to
`VLLMService`/`VLMPrediction` — it's no longer a constant you edit inside the package). It's off by
default so that big prompts (>50MB) can also be processed and not skipped. If you want the older
batch-boundary `LLMEngine` instead of continuous batching, set `backend="batch"` in `build_args()`
(the old `use_async=False` toggle in `vllm_service.py` has been removed).

Please note that using AsyncLLM with shm enabled is the fastest way to process prompts. Please check `docs/v1.5_changes.md` for more details.

## [1.9.4] - 2026-08-13
### Cross-venv `vllm serve` — `vllm_bin` + public `launch_vllm_serve`/`terminate_vllm_serve`

Non-breaking, additive. `vllm_online_spec()` (and the `WorkerSpec` args it builds) gained an
optional `vllm_bin` parameter — an absolute path to a SPECIFIC venv's `vllm` executable,
threaded through to `online_worker._launch_server`'s `subprocess.Popen`. Default is unchanged
(bare `"vllm"`, resolved off the ambient PATH of whatever env the pipeline process itself runs
in), so every existing caller is unaffected.

**Why**: a single `main.py` process orchestrating several VLM stages sometimes needs to serve
models that require genuinely different vllm/torch builds — e.g. a newer model needing a
CUDA-13 torch build alongside older models on a CUDA-12 build, both on the same node in the
same run. Before this, that meant either one shared (and probably-broken-for-one-of-them) venv,
or manually pre-launching a second `vllm serve` outside the pipeline entirely (see digital_twin's
former `TRANSLATION_BASE_URL` — removed this release in favor of native hosting).

**The one wrinkle vllm_bin surfaces**: a bare `subprocess.Popen([vllm_bin, "serve", ...])` does
NOT put that venv's own pip-installed `nvidia-*` wheel libs on the dynamic linker's search
path — only `import torch`'s own preloading does that, which a bare subprocess skips entirely.
Confirmed live serving `google/gemma-4-31B-it` from a second venv: `vllm serve` died instantly
with `ImportError: libcudart.so.13: cannot open shared object file`, even though the .so was
sitting right there in `site-packages/nvidia/cu13/lib/`. Fixed generically, not per-model:
`online_worker._venv_nvidia_lib_dirs()` derives every `site-packages/nvidia/*/lib` directory
under `vllm_bin`'s venv root and prepends them to the subprocess's `LD_LIBRARY_PATH`
automatically.

**New public API** (`vllm_module.launch_vllm_serve` / `terminate_vllm_serve`): thin wrappers
around the same `_launch_server`/`_terminate_server` the "online" backend uses internally, for
a `backend="function"` stage that needs a native, health-checked `vllm serve` subprocess
without going through the rest of `vllm_online_spec()`'s call-per-request `WorkerSpec` contract
(e.g. a stage with its own custom multi-request-per-row shape that isn't "one prompt in, one
completion out" — see `digital_twin/project_specific_translation.py`, which launches its own
model's server this way and drives it with plain HTTP client workers).

Validated end-to-end via `src/digital_twin`'s translation stage (`google/gemma-4-31B-it` served
from a second venv, `/environments/gemma4_new`, alongside chandra/qwen in `ocr_env_vllm` on the
same 8-GPU node) — see that folder's CLAUDE.md for the full write-up.

## [1.9.3] - 2026-08-12
### `JSONLWriter.flush()` — force a commit out-of-band when a stage goes idle

Full write-up: `docs/v1.9.3_known_limitations.md`. `JSONLWriter` only ever committed a
buffered batch at an `fsync_every_lines` boundary or at `close()` — a batch smaller than
`fsync_every_lines` (a short trailing run, or a low-image-count smoke test) could sit
buffered with no fsync boundary in sight, and `mark_done()`'s downstream stage-DB activation
buffered right along with it.

New public `flush()`: enqueues a sentinel on the writer's own queue; the background thread
runs the same `_flush_group()` commit sequence (write → fsync → `router_coordinator.
mark_done()` → `db.mark_done()`) it would at a normal boundary, staying inside the writer's
single-thread-owns-`_objs` invariant. No-op if nothing's buffered.

`cli_utils.py`'s `process_pipeline()` calls it on every idle tick — fetch empty, nothing
pending/inflight/in-prep — right before deciding whether to exit or sleep, instead of just
sleeping blind.

Also: the `docs/v1.9.3_known_limitations.md` `state_counts`-desync bug (downstream stages
never self-exiting) is now fixed — but on the StageWeaver side, in `stage_db_batch_write_
fetch_state_4` (StageWeaver v1.1.2), not in this package. Re-validated end-to-end via
`src/digital_twin` alongside this release.

## [1.9.2] - 2026-08-12
### One `WorkerPool`, one channel-visible `ServiceStatus`, vLLM stops being special

Full design and rationale: `docs/v1.9.2_changes.md`. **Breaking**: every `init_fn` / `term_fn`
changes, and Patram-Ingest must be updated together with the parent repo's stage functions.
StageWeaver needs no code change at all (docs-only 1.1.1).

v1.9 added `FunctionService` *beside* `VLLMService` rather than under it, and the codebase then paid
for two parallel supervision stories for one problem. Almost every concept this release deletes —
two service classes, two `from_channel`s, a health object named "VLM" that chrome workers published,
a per-`k` `ctypes.Structure` factory with a module `__getattr__` so it could survive pickling,
30-second SIGKILL detection via `STALE_S`, per-worker heartbeats — existed only to keep those two
columns in sync.

**The one abstraction.** `core/service.py:WorkerPool` (the renamed `FunctionService`) is now the only
thing that spawns, watches, restarts or buries a backend process. vLLM is a `WorkerSpec` like a
headless-chrome renderer is (`vllm_module/specs.py`: `vllm_online_spec` / `vllm_async_spec`). An
`init_fn` is two lines:

```python
pool = WorkerPool(vllm_spec(args.backend, engine_args=..., gpus=..., max_inflight=args.batch_size,
                            logs_dir=args.logs_path), channel, logs_dir=args.logs_path)
pool.start()      # publishes its own ServiceStatus on the channel, before spawning anything
```

and `term_fn` is `pool.stop()`. The five-step v1.9.1 dance — build a `VLMHealth` with a matching
`n_workers`/`max_inflight`, hand it to the service directly, `start()`, `attach_process()`,
`set_health()` — is gone, along with the `ValueError` that policed the sizes. A mismatch is no longer
representable.

**Sync or async is the loop's only degree of freedom.** `inspect.iscoroutinefunction(call_fn)` picks
between a get-one/answer-one loop and an event loop holding `max_inflight_per_worker` calls at once.
That single fact is all vLLM ever genuinely needed that a chrome worker did not, so the two
hand-rolled asyncio queue loops (`online_worker._amain`, `async_worker._amain`), their four bridge
threads and their `_SENTINEL` are replaced by one framework loop. `online_worker.py` /
`async_worker.py` are now just `init` / `call` / `term` triples.

**Two `vllm serve` replicas are `n_workers=2`.** Nothing else changes — each rank gets its own GPU
group via `devices` and its own ephemeral port, and the pool balances across them.

**`ServiceStatus` replaces `VLMHealth`** (`core/status.py`, moved out of `vllm_module` because
nothing in it was about vLLM). One fixed 5-field struct, written only by the pool, carrying the three
facts a consumer actually needs: `ready`, `failed` + `reason_code` + a 256-byte human-readable
`detail`, and one `heartbeat`. Per-rank flags, per-rank aggregates, `attach_process` /
`attach_worker_process` and per-worker heartbeats are all deleted — the pool holds the `Process`
handles, so `proc.is_alive()` every 0.5s answers the same question exactly, in the process where the
answer is free. Hard-kill detection drops from ~30s to under a second; `STALE_S` now guards only the
pool's own death.

**Fixed: a SIGKILLed worker permanently wedged the whole pool.** Through v1.9.1 all N workers called
`get()` on one shared `request_queue`. `mp.Queue.get()` holds the queue's shared reader lock while it
polls, and a process SIGKILLed in there never releases it, so **every surviving consumer blocks
forever**: the pool would recover the dead worker's requests, respawn it, report the service healthy,
and consume nothing ever again. A silent, permanent stall — precisely the failure the in-flight ledger
was invented to prevent, one level down, and it made `on_worker_death="respawn"` a promise the code
could not keep. Only one consumer holds the lock at a time, so the per-kill odds are roughly 1/N;
measured at 3 of 8 trials with two workers in ~20 lines of pure `multiprocessing`
(`examples/no_gpu_validation/mp_only.py`). 1/N is not a reprieve — the outcome is total and
unrecoverable whenever it lands, and a long run kills workers more than once.

The shared queue now has exactly one consumer, a dispatcher thread in the pool, which assigns each
request to the least-loaded worker with free capacity via a private per-worker queue. A killed worker
can only poison its own inbox, which the pool discards when it respawns it. Cost: one thread and one
extra hop (hundreds of microseconds against a model call measured in seconds). Beyond not hanging,
the inversion deleted three things: the worker-side concurrency semaphore (the dispatcher enforces the
ceiling, so two components can no longer disagree about it), the ledger's shared byte arrays with
their fixed field widths and write-ordering rules (the pool records assignments in ordinary Python,
because it is the only reader and writer), and the "microseconds between `get()` and the ledger write"
hole v1.9.1 documented — the record now exists before the request is sent. Ledger overflow became
unrepresentable and its error path is gone.

Still open, and stated rather than papered over: workers share the per-stage *response* queues, so a
worker SIGKILLed inside the microseconds where its feeder thread holds one's write lock can still
wedge the response path. The window is far smaller than the request side's — microseconds of pipe
write per response, against a reader lock held across a 0.25s poll — but it is the same class of bug,
and closing it would mean routing every response through the pool process too.

**`on_worker_death` is an explicit policy** (`"respawn"` | `"abort"`, plus `max_respawns`). Pools of
many cheap independent workers respawn — one chrome crash is throughput, not the run. Both vLLM specs
default to `"abort"`: a vLLM death is usually an OOM or driver fault that respawning reproduces, and
a relaunch costs GPU-minutes while the operator sees only a gap in throughput. Aborting resets
in-flight rows to their fetch state and marks nothing FAILED.

**`BackendGone`** (`core/task.py`) is the third and last thing a worker can discover, next to
`InputRejected`: "my resource is dead". The worker answers the request in hand as `infra`, drains,
runs `term_fn` and exits with a distinct status so the pool can say *why*. It replaces the online
worker's 5-second `proc.poll()` timer, the async worker's `_is_engine_dead_error` shared-flag write,
and a `REASON_*` code for each.

**`call_fn(ctx, payload, params)`** — one new argument, that request's inference parameters
(`SamplingParams` for vLLM, `None` for everything else). A `call_fn` may return a bare value or a
`CallResult(output, stats)` when it counted something the record should carry; vLLM's token counts
arrive that way. Every vLLM JSONL record now also carries `worker_rank` / `device` / `duration_ms` in
`full_vlm_object`, which is the first thing anyone asks once `n_workers > 1`.

**Retired.** `backend="batch"` and `timed_worker.py` (a batch-boundary A/B timing baseline that cannot
be a call-per-request worker), `debug_big_payload` and its x10 payload inflation, and
`lazy_loading` — the engine is built eagerly, so "ready" means ready instead of "ready to start
loading". `vlm_service.py` and `vllm_module/health.py` are gone; `VLLMRequest` and the wire contract
moved to `core/wire.py`, which finally makes the `core` -> `vllm_module` import direction clean.
Archived under `docs/archive/`.

**Unchanged, and still authoritative:** the whole `ok`/`reject`/`infra` failure taxonomy in
`prediction.py` — the model-attempt budget, `reject` terminal on first occurrence, the separate
`MAX_INFRA_ATTEMPTS` budget with its `others_are_working` blame-the-item rule, and the
consecutive-infra fuse at `max(2·batch_size, 64)`. It never knew what backend produced a response and
still does not. The fuse stays deliberately independent of `ServiceStatus`: it catches "alive and
failing everything", which no liveness mechanism can see.

**Validated with no GPUs** (the constraint on this work), on a fake-backend harness covering: a clean
sync run; an async spec proving real concurrency up to its declared ceiling; SIGKILL of a busy worker
(request recovered as `infra` on the right queue in 0.45s, rank respawned, service never reported
failed); a crash-looping `init_fn` exhausting its budget and reporting a startup failure; SIGKILL
under `"abort"` (0.45s, naming the rank and exit code); `BackendGone`; two async workers
load-balancing one queue with one then killed (the case that exposed the poisoning bug); and SIGKILL
of the *group leader*, detected by its stage at exactly `STALE_S` with the right reason. Not covered,
as in v1.9.1: a scripted kill against a real `vllm serve`, and measuring `STALE_S`.

## [1.9.1] - 2026-08-11

### 🎯 CONSOLIDATE — `VLMHealth` as one shared array, a heartbeat, `ServiceChannel`, and `shm` deleted

Design doc: `docs/v1.9.1_changes.md`. Resolves `docs/v1.7_stageweaver_gap.md`, open since v1.7.

**This release is BREAKING across two repos.** It changes StageWeaver's locked `init_fn` /
`term_fn` / stage-`function` signatures and `cli()`'s own. Requires StageWeaver ≥ 1.1.0. Every
`main.py` and every stage module must be updated together; the ones in this repo already are.

#### 🐛 Fix — `vlm_health` never reached StageWeaver (since v1.7)

`init_fn` / `term_fn` / stage `function` were flat positional signatures with no slot for it, so
under StageWeaver `cli()` **always** ran with `vlm_health=None` — silently, because every consumer
treats it as optional. Both the readiness wait (`cli.py`) and crash detection (`prediction.py`) were
dead code in that deployment; only the heuristic consecutive-failure fuse protected the DB. A dead
`vllm serve` degraded to that fuse, and a SIGKILL/OOM-kill of the backend was never noticed at all —
the stage hung forever.

Fixed by `ServiceChannel` (below) plus the heartbeat. Verified by reproducing StageWeaver's exact
topology (main → group leader → {backend, stage worker}, `spawn` throughout) with no GPU:

| | Before | After |
|---|---|---|
| Cooperative death, seen by the **stage worker** | never (`vlm_health` was None) | specific reason, ~1s |
| SIGKILL, seen by the **stage worker** | never — hangs | detected via heartbeat staleness |

#### 🐛 Fix — the v1.9 in-flight tracker silently destroyed recovery records

`VLMHealth._inflight` was one fixed slot per rank, overwritten unconditionally. A worker holding
more than one request at once (internal micro-batching, an async/threaded worker — nothing did this
yet, but `WorkerSpec` was written with SAM3/YOLO in mind, which naturally want to) would destroy the
first request's recovery record with no error: exactly the failure mode the ledger exists to
prevent, moved one layer down.

Each rank now gets `k` slots (`WorkerSpec.max_inflight_per_worker`, default `1` — today's exact
behaviour for every worker that exists). Overflow **logs an error and drops that one record**
instead of corrupting another. `k` is per-pool rather than one large shared constant on purpose: a
ceiling generous enough to "never matter" is also generous enough that a real bug — a leaked
semaphore permit inflating concurrency 5-10x — sails through undetected. `WorkerSpec` now rejects a
declared value above 512 at construction, the way `resolved_n_workers()` already rejects a
`devices`/`n_workers` mismatch.

#### 🔧 `VLMHealth` — nine attributes across three sharing mechanisms → one `mp.Array`

Was: `dead_event`, `ready_event`, `_reason`, `_ready_flags`, `_dead_flags`, a `Manager()` dict (and
therefore **an extra SyncManager process per instance**), plus `proc`/`procs` stripped by hand from
every pickle. Now: one `mp.Array` of a per-rank ctypes structure
(`ready`/`dead`/`reason`/`heartbeat`/`n_inflight`/`req_ids[k]`/`stages[k]`), built by a cached
factory so `k` can vary per instance.

- **`dead_event` / `ready_event` are gone.** Both were pure aggregates over the per-rank flags,
  hand-synchronised in every mutator — and **nothing in the codebase ever `.wait()`ed on either**.
  Every consumer polls, because it has to interleave the check with `shutdown_event` and
  `failure_reason()` anyway. Computing the aggregate on read is behaviourally identical and deletes
  the two-step lock dance everywhere, including `mark_alive`'s manual un-latch and reason rollback.
- **The extra Manager process is gone.** v1.9's `VLMHealth.__init__` started an `mp.Manager()`, and
  `VLLMService` substitutes a throwaway `VLMHealth()` when none is passed — so **every VLM stage
  silently spawned a SyncManager process that nothing ever read.**
- **`req_ids` / `stages` are separate fixed-width parallel arrays**, not one delimited blob. A
  `req_id` is always exactly 36 chars (`str(uuid4())`), so 37 bytes can never truncate one;
  `stage_name` is the only genuinely variable field, and giving it its own array means an oversized
  name can only truncate *itself* rather than corrupting a neighbouring `req_id`.
- **The array is deliberately lock-free** (`lock=False`). This is a correctness requirement, not an
  optimisation: `mp.Array`'s lock is a POSIX semaphore, and a process SIGKILLed while holding it
  never releases it — every later reader would block forever. That would poison precisely the path
  this class exists to serve, turning "was the backend SIGKILLed?" into an undetectable hang. No
  lock is needed because access is single-writer per rank (a rank's own process, plus the supervisor
  only *after* that worker is dead), ranks never touch each other's slots, and every field is a
  naturally-aligned scalar or fixed-width byte array. The two orderings a reader could observe
  mid-update are both benign and documented at their write sites.

#### ✨ Heartbeat — hard-kill detection that works in **any** process

Every backend process now runs one daemon thread writing `monotonic()` into its own rank slot every
~1s, started as the *first* thing it does — before `init_fn`, before loading a model — so a
five-minute model load is never mistaken for a corpse.

    dead(rank) = ranks[rank].dead                                  # cooperative
              or (has ever ticked and now - heartbeat > STALE_S)    # inferred

Hard-kill detection previously ran through `attach_process()`/`_all_procs_exited()`, which needs a
live `Process` object — and a `Process` handle only means something in the process that started it,
so `__getstate__` drops it. Under StageWeaver that is the group leader, while `cli()` runs in a
separately-spawned StageWorker, where the check was permanently `False`. A shared `mp.Array` *does*
survive `spawn` (verified two levels deep), so `failure_reason()` now gives the same correct answer
anywhere. `attach_process` / `attach_worker_process` are **kept** as a faster, exact-exit-code
accelerant wherever a live handle happens to exist.

`STALE_S` = 30s, deliberately generous: a worker blocked in GIL-holding C code can miss ticks, so
this is a backstop for "gone", not a tight SLA. A rank that has **never** ticked is never called
stale — otherwise a slow `spawn` import or a long model load would be misreported as a dead backend;
that window is covered by the cooperative flag and by `attach_process`.

#### ✨ `ServiceChannel` — one bundle instead of eight positional arguments

```
init_fn(args, logger, channel) -> init_vars
term_fn(init_vars, stop_event) -> None
function(args, channel, shutdown_event, router_coordinator, stage_specific_log_dir) -> None
```

`ServiceChannel` carries `request_queue` / `response_queues` / `stop_event` / `extras`. `init_fn`
publishes anything else its group needs into `extras` — `channel.set_health(...)` — **before** any
stage worker is spawned, which is when the channel is pickled. A future shared primitive is an
`extras` key, not a fourth signature migration across two repos.

`cli()` reads its own response queue (`response_queues[stage_name]`) and the health object off the
channel, so `request_queue` / `response_queue` / `vlm_health` are gone from its signature.

- **StageWeaver defines its own `ServiceChannel`** — it depends on nothing and cannot import ours —
  as a bare four-field dataclass. `vision_ingest.core.channel.ServiceChannel.adopt()` wraps whatever
  `cli()` is handed, so the conveniences live on exactly one side of the dependency boundary and
  there is no parallel surface to drift.
- **`shutdown_event` stays an explicit argument**, deviating from the design doc's proposed
  signature. It is created per stage process and set by that process's own SIGTERM handler — "stop
  this stage loop", a different question from `stop_event`'s "stop the group's shared resource".
  Folding a per-stage event onto a group-wide object pickled once per spawn would be a category
  error, and dropping it outright would break graceful shutdown: it is what lets a stage reset its
  in-flight rows from state=1 instead of stranding them.
- **`VLLMService` / `FunctionService` constructors keep their explicit parameters** (§6 Q1): a
  constructor is not a process boundary that must survive a pickle, so forcing the bundle past that
  point would only hide which primitives each class uses. Both gain a `from_channel()` classmethod
  that removes the call-site boilerplate without giving up that explicitness.

#### 🗑️ REMOVE — the `shm` subsystem is deleted

`src/vision_ingest/shm/` is gone, along with `shm_name` / `free_slots` / `use_shm` from every
signature: `cli()`, `VLMPrediction`, `VLLMService`, `retry_validate.stage()/drain_available()`,
`async_worker`, `timed_worker`, and every `main.py`.

It existed because pre-v1.8 offline prompts embedded decoded PIL images (3-12 MB each), where the
kernel copy through `mp.Queue` dominated everything. Since v1.8 a request is paths + text and a
response is text + stats, so there has been no workload to justify it — v1.9 already called it
vestigial and scoped its removal out only because `shm_name`/`free_slots` were part of StageWeaver's
locked signature. This release breaks that signature anyway, so removing it now is strictly cheaper
than a second migration later.

The design notes are preserved as `docs/archive/shm_readme.md`, `docs/archive/exit_hang_postmortem.md`
and `docs/archive/shm.py.removed`. **The exit-hang lesson is still live and still implemented**:
every process that `put()` to a queue calls `cancel_join_thread()` in an always-runs `finally`, now
via `channel.cancel_join_threads()`.

Both `model_term` variants converge as a result — with `shm` gone there is no ownership asymmetry
left, only "clean up the queues in whichever process created them."

#### 🔧 StageWeaver-side (requires StageWeaver ≥ 1.1.0)

`StageConfig` now rejects an empty name, and one longer than 255 bytes — the fixed-width field a
stage name travels through for crash recovery. An over-long name would truncate silently and then
fail to match any response queue precisely when it matters most, deep inside a recovery path.

#### ⚠️ Still open

- `docs/v1.9.1_changes.md` §6 Q2: `STALE_S = 30s` is an instinct, not a measurement against a real
  worker's worst-case GIL-holding call. Worth revisiting against chrome rendering / SAM3.
- No automated test suite exists in this repo. The wiring and failure paths above were verified with
  throwaway harnesses reproducing StageWeaver's topology without GPUs; the scripted-kill test
  `docs/v1.7_stageweaver_gap.md` proposed still has not been run against a real vLLM backend.

## [1.9.0] - 2026-08-10

### 🎯 EXTEND — Arbitrary Python functions as a backend, multiprocess workers, image outputs

v1.8 unified the request payload and gave responses a worker-decided `ok`/`reject`/`infra` kind.
v1.9 cashes that in: the pipeline can now run **an ordinary Python function** — HTML→PNG rendering
through a headless browser, Set-of-Mark annotation, SAM3, a YOLO or HF model — with the same
crash-safety, retry, recovery and StageWeaver behaviour it already gives vLLM.

**v1.9 does not add a second pipeline. It makes the payload opaque and adds a second implementation
of the process-side of the contract v1.8 already defined.** The DB state machine, the
write→fsync→router→`mark_done` commit protocol, `recovery.py`, admission control (`_owned`/`fetch_n`),
the retry and infra-failure taxonomy, `PipelineMetrics` and the StageWeaver hooks are all unchanged
in shape. Full write-up (including where the implementation diverged from the proposal, and what was
and was not verified): `docs/v1.9_changes.md`.

**The output rule, stated once: the JSONL is always the output — every backend, every job.** Images
are *additional*, and only images go into folders-then-tars.

#### Added
- **`core/` package** — all new code, nothing migrated into it. `vllm_module/`, `prediction.py`,
  `cli.py`, `writer.py`, `recovery.py` and `health.py` stay exactly where they were.
- **`core/task.py`** — `FunctionTask`, the `PromptAndValidation` counterpart for
  `backend="function"`. Deliberately reuses the method names `prediction.py` already calls
  (`wrap_and_validate_output`, `build_retry_request`, `write_local_json`), so the predictor's code is
  identical for both backends; adds `build_request(path, logger) -> dict | None`. Plus
  **`InputRejected`**, the exception a `call_fn` raises to mean "this input is bad and always will
  be".
- **`core/service.py`** — `WorkerSpec(init_fn, call_fn, term_fn, args, devices, n_workers)` and
  `FunctionService`, which spawns N processes each holding its own resource and satisfies exactly
  the contract `VLLMService` does (one shared `request_queue`, per-stage `response_queues`,
  `stop_event`, `VLMHealth`). `(rank, device, ctx)` is the whole abstraction: `devices=None,
  n_workers=32` gives one headless Chrome per process; `devices=["cuda:0", …, "cuda:7"]` gives one
  model replica per GPU, with **no branching in the framework**. Load balancing is work-stealing by
  construction — N processes competing on `request_queue.get()`, no scheduler, no per-worker queues.
- **A supervisor thread that recovers a dead worker's in-flight request *exactly*.** On a worker
  exiting while `stop_event` is clear it reads that rank's shared slot, emits
  `VLLMResponse.infra(req_id, "worker <rank> died")` on the correct response queue, and respawns.
  Without it the `req_id` sits in `prediction.py`'s `inflight` dict forever: `_owned` never
  decrements, `fetch_n` walks to 0, and the pipeline stalls in silence. Respawns are bounded
  (`MAX_RESPAWNS_PER_RANK = 3`, 2s backoff) so a broken `init_fn` cannot crash-loop for a whole run,
  and so "every worker died" converges on a clean `VLMServiceDown` abort.
- **`core/shards.py`** — `ShardAllocator`, the single authority on image-output folder assignment.
  `reserve` / `commit` / `release_failed` / seal, all in the cli process; workers only perform I/O on
  paths handed to them and never choose a filename, pick a folder, or scan a directory. Rolls by
  byte size (default 10 GB, the same trigger JSONL rotation uses), seals to an **uncompressed** tar
  via `.tar.tmp` → fsync → `os.rename` → dir-fsync → DB pointer-swap hook → CSV index → *optional*
  `rmtree`, on a background thread, continuously through the run rather than batched to the end.
- **Two output trees, and folder deletion is opt-in (`delete_folders_after_seal`, default `False`).**

  ```
  {root}/{host}/image_folders/shard_00007/a.png   <- what the worker writes; KEPT by default
  {root}/{host}/shards/shard_00007.tar            <- the packed shard
  {root}/{host}/shards/shard_00007.csv            <- its byte-offset index
  ```

  By default a sealed folder is **retained**, so every image exists both loose and inside its tar.
  That is a deliberate ~2x disk cost, bought for: loose files stay directly readable with no
  untarring, a bad tar can be rebuilt from its folder, and a downstream stage that resolved the
  *folder* path keeps working forever rather than only until the seal. Set
  `args.delete_shard_folders = True` to reclaim the space once the tars are trusted — flipping it on
  later also cleans up previously-retained folders at the next startup. Keeping the trees separate
  makes `shards/` exactly the set of shippable artifacts, so `rsync`/`aws s3 cp --recursive`/a glob
  need no filtering.
- **The CSV doubles as the seal's completion marker.** With folders retained, "is there a folder next
  to this tar?" stops being a crash signal (it is now the permanent normal state), so the CSV is
  written last of the durable steps and startup recovery keys off it: `.tar` + `.csv` means finished,
  skipped in one `os.path.exists`; `.tar` with no `.csv` means the seal was interrupted after the
  rename, so the tail is re-run. Without a marker, a run with 10,000 retained folders would re-call
  the pointer-swap hook for all 10,000 on every startup.
- **Reservation-time tar naming.** The JSONL record names the **final tar** from the moment of
  reservation (`{"shard": "shard_00007.tar", "member": "a.png"}`), so a line is correct both before
  and after its shard seals and sealing never has to touch an immutable JSONL shard.
- **Startup recovery for shards** — a leftover folder with no tar is sealed and allocation resumes at
  `NN+1`; a tar with no CSV means the seal was interrupted after the rename, so the
  (idempotent-by-contract) `on_shard_sealed` hook is re-called and the index rebuilt; a stray
  `.tar.tmp` is deleted.
- **Byte-offset CSV index** — `shard_00007.csv` written next to every sealed `shard_00007.tar`, with
  `image_name_in_shard`, `original_image_path`, `img_tar_shard_path`, `byte_offset`, `byte_length`.
  `byte_offset`/`byte_length` span the file's **content only** (tar header excluded), so a consumer
  can seek/Range-read one image straight out of the tar — locally or as an S3 Range request — with no
  untarring. Column names follow Unified-Vision-Dataset-Repo's `images.csv` convention
  (`standardisation_spec.md` §7.3) for the subset that applies. Offsets are read back **from the
  just-sealed tar itself** (`TarInfo.offset_data`/`.size`) rather than tracked during the write,
  because `TarFile.addfile()` copies the `TarInfo` it is given and never reports position back to the
  caller (verified empirically); that also makes the same method reusable unchanged from the
  crash-recovery path. Framework-level — it applies to any `backend="function"` task with image
  output, not just the example.
- **`VLMHealth` generalized to N workers** — per-rank ready/dead flag arrays, with `is_ready()` true
  when **all** ranks are ready (so cli's "wait for the backend, then start the clock" still excludes
  model-load time) and `failure_reason()` non-`None` only when **all** ranks are dead (one Chrome
  worker dying costs throughput, not the run). Adds `mark_alive(rank)` and
  `attach_worker_process(rank, proc)`. Plus one shared `mp.Array` slot per rank holding the
  `(req_id, stage_name)` that rank is currently processing, written immediately after `get()` and
  cleared immediately after `put()`.
- **`VLLMResponse.output`** — property alias for `text`, so `prediction.py` reads a backend-agnostic
  name. The field keeps the name `text` so every worker construction site is untouched; its
  annotation widened to `Optional[Any]`.
- **`examples/html_render/`** — the worked example: `render_engine.py` (a verbatim port of
  Digital-Twin-Pipeline's SMART render path — CDP-driven viewport control, a JS TEXT AUTO-FIT pass
  that grows the box, then the page, then shrinks the font rather than silently clipping text, and an
  optional content-tight crop, off by default as in the source), `worker.py` (one persistent headless
  Chrome per worker process via selenium/CDP, reused across every call that process handles),
  `task.py` (`HtmlRenderTask`, reserves a shard path and rejects a blank render), and `main.py`
  (`devices=None, n_workers=32`). Structurally the same shape as `examples/vllm_testing/main.py`,
  with `FunctionService` where `VLLMService` stands.
- **`backend="function"` end-to-end wiring** — one `if` in `cli()`, plus reservation resolution in
  `cli_utils.process_iteration` (the one place that already sees both terminal outcomes).

#### Changed
- **The request payload is now opaque.** `prediction.py` touches its contents in exactly two places:
  it records `send` on the record (`rec["sent_payload"]`, was `rec["sent_prompt_str"]`), and it hands
  `response.output` to `wrap_and_validate_output()`. It never reads `"prompt"`, never reads
  `"image_paths"`, and never assumes the response is text. The only structural constraint left:
  **a payload is always a JSON-able dict.**
- **`VLMPrediction(function_task=...)`** — one new optional constructor arg, mutually exclusive with
  `vlm_config` / `model_name` / `config_path`. When set, `get_prompts()` calls
  `build_request(path, logger)` per path through the same `reader_pool` instead of the
  `get_image_specific_prompt` → `get_prompt_with_image` two-step, and `_base_sampling_params()` /
  `_check_backend_supports_sampling_params()` short-circuit to `None`. **Existing vLLM projects
  change nothing** — the two-step path is untouched, just now inside an `else`.
- **`vllm_module/vllm.py` degrades gracefully when vLLM is not installed** — `try: from vllm import
  … except ImportError:` now falls through to the dummy `SamplingParams`/`StructuredOutputsParams`/
  `LLM` classes that already existed behind the hand-flipped `local_testing` constant, and exposes
  `VLLM_AVAILABLE`. Every vLLM-adjacent import in the package bottoms out here, so the whole chain —
  `drivers/cli.py`, `modules/prediction.py`, `vllm_module/retry_validate.py` — becomes importable on a
  CPU-only node. `local_testing` is kept as a manual override.
- **`cli()` logs `vlm_model_name`/`vlm_gpus` via `getattr`** and records `backend` +
  `image_output_path` in its startup config — a function backend has no model or GPU list, and
  requiring the args to carry empty ones would be a lie in the log.
- **`process_pipeline()` takes `shard_allocator=None`.** When present, `process_iteration` calls
  `release_failed(path)` for every terminal failure and `commit(path, vlm_output)` for every success.
  No-ops for items that never reserved, so a vLLM run passes through untouched.

#### Fixed
- **A `SIGKILL`ed worker no longer strands its in-flight request forever.** This is the same class of
  gap `docs/v1.7_stageweaver_gap.md` records for a `SIGKILL`ed `VLLMService`, which no amount of
  health-flag threading could reach; the per-rank in-flight slot is the shape that closes it, and is
  worth considering for `VLLMService` separately.

#### ⚠️ Deliberately NOT added: a per-request timeout
The obvious way to notice a lost request is a deadline on it. Under StageWeaver many stages share one
backend (`initialization_source = [0, 0, 0]` puts three stages on one service), so a request can
legitimately sit queued for a very long time with nothing wrong. A timeout would have to be tuned
against contention it cannot observe, and would begin failing healthy work the moment another stage
joined. The health channel is the right mechanism for the same reason it was in v1.7: **the process
that knows the fact reports the fact**, instead of a second process inferring it from a clock.
Recorded here so it is not "simplified" back in later.

#### ⚠️ Known limitations
- **The DB pointer swap is stubbed.** `on_shard_sealed(shard_id, folder_path, tar_path)` ships as a
  documented no-op (not `raise NotImplementedError`, which would fail every seal and take tar
  creation with it). Everything around it is real and tested — it fires exactly once per shard,
  strictly between `os.rename` and `rmtree`, and recovery re-calls it. Wiring it to a real schema is
  the one open item: until then the "downstream reads the folder path before the seal, the tar path
  after" half of §5 is unimplemented.
- **A hard `kill -9` can leave orphan members in a sealed tar.** Workers may have fsynced images whose
  JSONL records were never committed; recovery cannot tell those apart from legitimate files (the
  thing that would distinguish them is the DB record that was never written), so it seals them. Those
  rows are replayed into a *new* shard, leaving the originals unreferenced. Deliberate trade: an
  unreferenced member costs disk, discarding possibly-legitimate files costs data. **Every JSONL
  record still resolves** — the tar is a superset, never a subset.
- **`transformers` is still required** for a function-only deployment (`vllm_config.py` imports it at
  module scope and `prediction.py` imports `VLLMConfig` for `PromptConfigError`) — just not
  `vllm`/CUDA. Making that import lazy is a small follow-up, not done here because it is not what
  blocked a CPU-only worker.
- **The vLLM GPU regression run is still owed.** See Verification below.

#### Verification
Run against real worker processes, real queues, the real `cli()` path, `JSONLWriter` and DB —
using a PIL-based stand-in `call_fn` (no GPU was touched), plus real runs of
`examples/html_render/main.py` against real HTML through local Postgres.

- **Function backend, 8 workers** — 8 processes confirmed; JSONL commits on the `fsync_every_lines`
  boundary; DB settles at 2/3 with **none stuck in 1**. `InputRejected` → terminal on first
  occurrence, no retry; an unexpected exception → retried on the infra budget, blamed on the item
  only once other requests were demonstrably succeeding.
- **Shards** — 8–10 shards rolled and sealed *continuously* through a run; uncompressed verified by
  magic bytes; no `.tar.tmp` and no unsealed folder survive; rolls only past the byte threshold; a
  folder seals only after its last outstanding request returns; a terminally-failed reservation's file
  is deleted and never appears in the tar; `on_shard_sealed` fires exactly once per shard with the
  correct `(folder_path, tar_path)`; **every `{shard, member}` pair in the JSONL resolves** (198/198,
  1974/1974, 298/298).
- **CSV index** — verified by actual byte-range reads back out of the produced tars (not log
  inspection), on both the normal seal path and the crash-recovery path.
- **Crash safety** — `kill -9` mid-shard, then restart: the leftover folder sealed, allocation resumed
  at `NN+1`, `recover_last_shard` ran, and the member check still passed for every record. The three
  recovery branches were additionally driven as direct unit cases.
- **Worker death** — `kill -9` one worker of four mid-request: `infra` emitted for exactly that
  `req_id`, retried with **attempt count untouched**, respawned, run finished 40/40 in state=2 with
  **zero in state=3**. Then killing every worker until the respawn budget was exhausted: aborted with
  `VLMServiceDown` and **reset 12 in-flight paths to `fetch_state`** — final DB
  `{state_0: 24, state_2: 16, state_3: 0}`. **Nothing marked FAILED**, which is the entire point.
- **`html_render`** — two full end-to-end runs against real HTML files (25 total, 0 failures) through
  local Postgres, plus a standalone render-engine smoke test and a direct
  `model_init`/`model_call`/`model_term` test.
- **vLLM regression — PARTIAL.** No GPU run was performed (the node's GPUs were in use). What was
  proven is what §2 actually put at risk: driving `_classify()` with a vLLM-shaped payload yields
  `['path', 'prompt_str', 'vlm_output', 'full_vlm_object']` — the same four keys in the same order
  with the same values as v1.8, `prompt_str` still exactly `send["prompt"]`, **no new key**. (The
  function path adds `sent_payload` and nothing else.) vLLM-blocked import checks cover the
  graceful-degradation change. **Still owed before this is called done: an actual
  `examples/vllm_testing/main.py` run on `backend="online"` and `backend="async"`, byte-compared
  against a pre-change run.**
- **StageWeaver — not run.** §6 argues no StageWeaver change is required and nothing in the
  implementation contradicts it, but the two-stage run with `route_taken` stamped was not performed.

#### Dependencies
- `selenium` (plus `trio`, `trio-websocket`, `outcome`, `wsproto`, `pysocks`, `sortedcontainers`)
  installed into `/environments/ocr_env_vllm` — required by `examples/html_render/worker.py`, which
  uses selenium/CDP rather than playwright. Also needs a chrome/chromedriver build
  (`CHROMEDRIVER_PATH` / `CHROME_BINARY_PATH`, defaulting to the Chrome for Testing build under
  `digital_twin_stage_weaver/chrome_cft/`). Example-only — the framework itself adds no new
  dependency.

## [1.8.0] - 2026-08-06

### 🎯 SIMPLIFY — One payload shape, tagged responses, per-project retries

v1.7 gave every silent failure mode its own explicit channel. v1.8 deletes most of those channels
by deleting the ambiguity they guarded. Full write-up: `docs/v1.8_changes.md`; original proposal:
`docs/v1.8_suggested_changes.md`.

#### Added
- **`vllm_module/wire.py`** — the stage↔worker contract in one place: `VLLMResponse(req_id, kind,
  text, stats, error)` with `kind ∈ {ok, reject, infra}`, the two uniform stats builders, and
  `build_chat_messages()` (shared by all three workers).
- **`PromptAndValidation.build_retry_request()`** — overridable retry construction. Default
  reproduces v1.7 behaviour exactly (append error to prompt, +0.5 `repetition_penalty` per attempt),
  so no project has to opt in. `prediction.py` still owns copy-safety: the hook is handed a fresh
  dict and a deepcopy, never the shared originals.
- **`VLLMConfig.engine_media_root()` / `media_root_warning()`** — derive vLLM's single
  `allowed_local_media_path` from `allowed_media_roots`, logging any widening.

#### Changed
- **One request payload for every backend** — `{"prompt": raw text, "image_paths": [...],
  "enable_thinking": bool}`. Not chat-templated, no decoded pixels. Each worker renders it with
  vLLM's own code (`renderer.render_chat_async` for async, `LLM.chat` for batch,
  `/v1/chat/completions` for online), so online and offline now run *identical* templating rather
  than two implementations that happened to agree.
- **`VLMPrediction._validate_one` reads `response.kind`** instead of reverse-engineering an untyped
  union (`None`, a `type(...).__name__` string match, `.usage` duck-typing).
- **`allowed_media_roots` is required for any vision model on every backend** (was online-only).
  vLLM refuses `file://` without it, in-process as much as in `vllm serve`. It now travels inside
  `engine_args`, so `VLLMService(allowed_media_roots=...)` is gone.
- **Image-header validation is unconditional** — the stage decodes nothing on any backend now, so
  this is the only local chance to reject a corrupt/oversized file before submission.
- **Batch backend** renders inline before `LLM.chat()`, groups by `enable_thinking`, and retries a
  failed batch one item at a time so a single poison item cannot fail its whole cohort.

#### Fixed
- **A `structured_outputs` misconfiguration marched the dataset into state=3** (v1.8 Defect 1). It
  failed per item as a "rejection", resetting the infra fuse each time. Now refused at startup by
  `VLMPrediction`, before a single row is fetched.
- **Retries were materially different requests on the two backends** (Defect 2). Offline, the
  "Previous attempt failed: ..." text was appended to the already-templated string — i.e. *after*
  `<|im_start|>assistant\n`, inside the model's own turn. Templating now always happens after the
  append.
- **A deterministic per-input failure was classified oppositely by backend.** A prompt over
  `max_model_len` was a clean `reject` online but a bare `None` offline, burning all 5 infra retries
  and finally logging as a *backend* failure. Both report `reject` now.
- **The second `--allowed-local-media-path` silently overwrote the first.** vLLM models it as a
  single `str`, so a run with two roots had every image under the first one rejected by the server.
  `examples/chandra_testing` was configured this way and does send from both roots.
- **Engine-death detection missed `EngineDeadError`'s own message** (`"EngineCore encountered an
  issue..."` — the `"engine core"`/`"died"` pairing matched neither half). Broadened, and
  `chat_batch()` additionally downgrades `reject`→`infra` when nothing in a batch succeeded, so the
  heuristic is no longer load-bearing on its own.

#### Removed
- `_payload_kind`, `assert_payload_kind()`, `BackendPayloadMismatch`, `VLLMConfig(backend=...)`,
  `VLLMConfig._assert_image_placeholders()` (vLLM's multimodal processor enforces it),
  `online_worker`'s `_OnlineOutput`/`_OnlineCompletion`/`_OnlineFailure` shims,
  `retry_validate.post_process_full_vllm_object()` and `collect()` (dead since v1.5),
  `VLLMService(allowed_media_roots=...)`.

#### ⚠️ NOTE — shm is now vestigial
The shared-memory pool exists because offline prompts embedded 3–12 MB of decoded PIL image per
request and the kernel copy through `mp.Queue` dominated everything else. Requests are paths + text
now. It is left in place and still works (`use_shm` defaults to `False`), but it no longer has a
workload that justifies it. Deleting the subsystem is a clean follow-up, deliberately not bundled
into this change.

## [1.7.0] - 2026-08-06

### 🎯 FIX — Infra failures no longer march the DB into state=3; unified image validation

Every backend previously reported every failure the same way (`output = None`), so a dead
engine, a missing file, a wrong `is_llm`, and a backend mismatch were all indistinguishable
from "the model produced a bad answer" — walking the whole DB into state=3 on any
infrastructure failure. Full write-up: `docs/v1.7_changes.md` (design log),
`docs/v1.7_reported_issues.md` (issue-by-issue root cause → fix).

#### Added
- **`vllm_module/health.py`** — `VLMHealth` shared liveness flag (`mp.Event` + `mp.Value`
  reason code) between `VLLMService` and the stage. `VLMServiceDown` is raised instead of
  marking in-flight paths FAILED whenever the backend itself is gone (engine death, `vllm
  serve` subprocess exit, SIGKILL/OOM-kill via `proc.is_alive()`, undecodable request,
  payload-kind mismatch).
- **Four-status classification in `prediction.py`** (`_validate_one`/`_classify`): `ok` /
  `fail` (bad output, normal retry budget) / `reject` (deterministic per-input refusal —
  terminal immediately) / `infra` (backend produced nothing — own `MAX_INFRA_ATTEMPTS`
  budget, never marks a path FAILED while the backend is at fault). Infra failures also trip
  a fuse (`max(2*batch_size, 64)` consecutive failures with zero successes) even when
  nothing set the health flag.
- **`vllm_module/media.py`** — single validation path for both backends: absolute path,
  UTF-8-encodable, exists, regular file, non-empty, inside `allowed_media_roots`, and
  (online only) header-decodable and within the pixel ceiling (`VLLM_MAX_IMAGE_PIXELS`,
  ~179M px, now enforced offline too). `to_file_url()` percent-encodes `file://` URLs — raw
  interpolation let `#`/`?` truncate a path and `%` silently decode into a different file.
- **`PromptConfigError` / `BackendPayloadMismatch`** (`vllm_config.py`) — configuration
  errors (`is_llm=true` + images, over `limit_mm_per_prompt`, mismatched `args.backend`
  across the `main.py`→`VLLMService` and `cli.py`→`VLLMConfig` chains) abort the run instead
  of failing images one at a time, since they'd affect every image identically.
  `_payload_kind` + `assert_payload_kind()` guard the backend/payload agreement on the wire.
- **`_assert_image_placeholders()`** (`vllm_config.py`) — offline prompts must contain
  exactly one image-placeholder token per attached image; a chat-template mismatch used to
  silently answer from text alone.
- **`processor_downscales_above_pixels`** logged at startup — the model's own HF image
  processor's downscale threshold (e.g. chandra-ocr-2's `size.longest_edge`), real and
  undisableable, but previously invisible.
- **`_shielded_teardown()`** (`cli.py`) — ignores SIGINT/SIGTERM for the whole shutdown
  `finally` block so repeated Ctrl+C can't strand rows mid-reset. A second sweep resets
  `writer.unflushed_paths()` (accepted but never fsynced) in addition to in-flight paths.
- **Ephemeral port + `/v1/models` identity check** (`online_worker.py`) — a hardcoded port
  let any stale/foreign listener silently serve the whole run with the wrong model.
- **Real logging for `VLLMService`/`online_worker.py`/`async_worker.py`** — replaced
  `print()` with `get_logger()`, built inside the spawned child. `vlm_service.log` now lands
  in the same `{logs_path}/{hostname}/{timestamp}/` directory as every other component's log.

#### Fixed
- `async_scheduling` now defaults to `False` on all backends (`DEFAULT_ASYNC_SCHEDULING`) —
  vLLM ≥0.25 enables it implicitly when unset, implicated in node-level hangs/crashes.
  `_engine_args_to_serve_flags()` now emits `--no-<flag>` for `False` (vLLM's
  `BooleanOptionalAction` dropped bare `False` silently, so the yaml setting worked offline
  and was silently ignored online).
- `online_worker.py` now polls the `vllm serve` subprocess every 5s and stops on exit,
  instead of continuing to POST into a closed port and failing the rest of the DB at full
  speed.
- `raise_for_status()` no longer discards the server's 4xx response body — `_OnlineFailure`
  now carries the server's own explanation.
- `structured_outputs` / `min_p` / `stop_token_ids` now forwarded (or explicitly rejected)
  online — previously dropped silently, producing unconstrained output that failed the
  task's own validation on every retry.
- `get_state_pct(0) == 0.0` (rounds to one decimal) no longer used as the exit condition —
  could read a 10M-row DB with ~5000 pending rows as "0.0%" and declare the dataset
  finished. Now `db.get_state_count()`.
- `write_local_json` Future is now read/logged instead of submitted-and-ignored — a raised
  exception there used to fail silently for the whole run.
- Writer duplicate suppression is now an O(1) set (was an O(n) list scan that dropped
  silently); logs when it fires.
- `VLLMService._worker()` no longer dereferences `self.shm` unconditionally on the default
  `use_shm=False` path.
- `args.backend` now threaded consistently through all three examples (was defaulting
  silently in `chandra_testing/main_test.py`).

#### Verified
`scratchpad/verify.py` — 22 assertions against a stubbed `vllm` + the real chandra
processor: media validation error messages, `file://` round-trip (7 path shapes), serve-flag
translation (`False → --no-<flag>`), payload-kind guard both directions, `VLMHealth` reason
reporting, end-to-end `VLMPrediction` vs a fake worker (dead backend → `VLMServiceDown`,
nothing marked failed).

**Not verified — needs one GPU run before production:** `async_scheduling: false`'s
engine-side effect, the `vllm serve` launch path end to end, real throughput.

## [1.6.1] - 2026-07-28

### 🎯 FIX — `vllm serve` parity for offline AsyncLLM + sampling-param overlay

Closes two of the three divergence mechanisms documented in `docs/vllm_serve_offline_parity.md`
(Divergence 3, image loading, remains deferred). Full write-up: `docs/v1.6_changes.md`'s
2026-07-28 addendum.

#### Fixed
- **Offline `AsyncLLM` scheduler defaults now match `vllm serve`.** `AsyncLLM.from_engine_args()`
  was defaulting `usage_context` to `ENGINE_CONTEXT`, which isn't a key in vLLM's scheduler-defaults
  table and silently fell through to `max_num_batched_tokens=2048` / `max_num_seqs=128` — far below
  what `vllm serve` picks (`8192`/`1024` on this hardware class). Now passes
  `usage_context=UsageContext.OPENAI_API_SERVER` explicitly. Since vLLM's kernels are not
  batch-invariant, this affects output quality, not just throughput.
- **`generation_config.json` is now merged into sampling params on both backends**, not just
  online. `VLLMConfig.get_sampling_params()` used to hardcode `temperature=0.2` / `max_tokens=4096`
  / `repetition_penalty=1` as fallbacks — leftover constants from an old bakeoff script, not the
  model's own defaults. Now merges `GenerationConfig.from_pretrained(model).to_diff_dict()`
  (`temperature`/`top_p`/`top_k`/`min_p`/`repetition_penalty`/`max_new_tokens`) as the base, with
  precedence `explicit yaml value > generation_config.json > vLLM neutral default` — model-agnostic,
  every future model gets `vllm serve` parity automatically.
- **`max_tokens` no longer silently caps output at 4096.** When nothing sets it, it's now an
  explicit `max_tokens=None` (not an omitted kwarg — `SamplingParams`'s own dataclass default is
  `16`, not `None`) so vLLM's `InputProcessor` fills it as `max_model_len - prompt_len`, exactly
  like a `vllm serve` client that never sends `max_tokens`.
- **`online_worker.py` now forwards `top_p`/`top_k`/`repetition_penalty`** to the HTTP request —
  previously only `temperature`/`max_tokens` were sent, so `prediction.py`'s per-retry
  `repetition_penalty` bump never actually reached the online backend.
- **`get_engine_args()` no longer leaks sampling-only yaml keys** (`sampling_temperature`,
  `sampling_max_tokens`) into `AsyncEngineArgs(**engine_args)` / serve-flag translation. Was latent
  (nothing set those keys yet) until this change started actively reading them.

#### Added
- **`get_sampling_params()` override contract**: a task's `PromptAndValidation.get_sampling_params()`
  can now return `None` (default — defer entirely to the generation_config-merged base), a `dict`
  (override only those fields), or a full `SamplingParams` (legacy full-replacement, unchanged
  behavior). Applied the `None`-by-default pattern to `gemma_testing`, `vllm_testing`, and the
  in-package template; left `chandra_testing`/`archive/*` alone (deliberately-tuned values).
- **Per-item VLM output diagnostics.** `retry_validate.post_process_full_vllm_object()` (a
  `return None` stub since v1.5) now returns `prompt_tokens` / `completion_tokens` / `mm_tokens`
  (online only, per-modality) / `finish_reason` / `stop_reason`, written into every JSONL record's
  `full_vlm_object` field alongside `attempts` (retries consumed before success). `online_worker.py`'s
  response shim now carries the server's `usage` object and `finish_reason` instead of discarding
  them.
- **`PipelineMetrics.record_tokens()`** — new `token_metrics` block in checkpoint logs:
  `avg_prompt_tokens`, `avg_completion_tokens`, `avg_mm_tokens_by_modality`,
  `finish_reason_counts` (surfaces truncation — `finish_reason == "length"` — instead of it being
  silent), `avg_retries_per_success`.

#### Deferred
- Divergence 3 (image loading: `cv2.imread` vs vLLM's PIL + white-background RGBA compositing).
- Cached-token ratio and per-request latency metrics.
- Thinking-token counts — no offline equivalent to vLLM's `--reasoning-parser` without extra
  machinery, and nothing currently sets `enable_thinking=True` in production.

## [1.6.0] - 2026-07-27

### ⭐ FEATURE — Online-Serving Backend (`vllm serve` over HTTP)

Adds a third inference backend: a locally-launched `vllm serve` (OpenAI-compatible HTTP API),
driven over `/v1/chat/completions`. Online serving has shown better speed/quality than the offline
in-process engine in testing. Everything outside `VLLMService` — DB, writer, retries, StageWeaver,
shm — is untouched: this honours the "any backend that speaks `request_queue`/`response_queue`"
design from CLAUDE.md. Only *how* `VLLMService` fulfils a request changes (subprocess + HTTP instead
of in-process `engine.generate()`). Full design notes: `docs/v1.6_changes.md`.

#### Added
- **`backend` selector** (`args.backend`) with three values, threaded from `main.py` → `model_init`
  → `VLLMService` (worker dispatch) and via `cli.py` → `VLLMConfig` (prompt-payload shape):
  - `"online"` — launch a local `vllm serve` subprocess and drive it over HTTP (new default in
    `examples/vllm_testing/main.py`).
  - `"async"` — in-process `AsyncLLM` continuous batching (the v1.5 default worker).
  - `"batch"` — batch-boundary `LLM.generate()` (`timed_worker` / `_worker`).
  The old module-level `use_async` no longer selects the worker (kept only as a legacy note);
  `debug_timing` is now consulted only when `backend == "batch"`.
- **`vllm_module/online_worker.py`** — the `"online"` branch of `VLLMService.run()`:
  - Launches `vllm serve` via `subprocess.Popen` in the **same venv** (no Docker), polls `/health`
    until ready, and on `stop_event` tears the subprocess down (`terminate()` → `wait(30)` →
    `kill()`).
  - Reuses `async_worker.py`'s receiver/sender bridge threads unchanged (they already branch on
    `use_shm`); one `asyncio` task per in-flight request via `httpx.AsyncClient`.
  - **`_engine_args_to_serve_flags()`** translates the model-yaml `engine_args` dict into serve
    CLI flags (snake_case → `--kebab-case`; bool `True` → bare flag; dict/list → JSON), so the
    online server loads with the same `max_model_len` / `gpu_memory_utilization` /
    `limit_mm_per_prompt` / `trust_remote_code` / `async_scheduling` / … as offline. `model` is
    positional; `tensor_parallel_size` is derived from `gpus` (mirrors `_load_model`).
  - Minimal `RequestOutput` shim (`.outputs[0].text`) wrapping the HTTP response — the exact
    surface `prediction._validate_one` reads.
- **Online prompt-payload shape** (`vllm_config.get_prompt_with_image`, gated on `backend`): sends
  raw user text + **image file paths** (`file://` URLs, allowed root `/fsxvision_new`) — **no PIL
  load, no `apply_chat_template`**; the server templates the prompt and reads the files itself.
  Text-only LLMs (`is_llm`) send content as a plain string with no images.

#### Wire contract preserved (why this is a safe swap)
- Exactly one response `(req_id, output)` per request; `output=None` on any HTTP failure → drives
  the existing retry in `prediction.py` (unchanged).
- The online prompt dict keeps the `"prompt"` key, so `prediction._build_send_prompt`'s retry
  error-append is backend-agnostic — **`prediction.py` / `retry_validate.py` are untouched**.
- Payloads carry only paths + text (tiny), so `use_shm=False` queue transport is used directly —
  no shm slot sizing / oversize handling on this path.

#### Unchanged (no work needed)
- `prediction.py`, `retry_validate.py`, DB, `JSONLWriter`, StageWeaver, the shm pool, and
  `main.py`'s `model_init()` / `model_term()` lifecycle. `project_specific.py` hooks are called
  exactly as before.

#### Deferred
- `repetition_penalty` (retry bump) and structured-output / guided decoding via `extra_body`.
- Per-request `usage` → token-count metrics (`post_process_full_vllm_object` still returns `None`).

### 🔧 Cleanup — `use_async` removed, `use_shm` now a real threaded arg

- **Removed `use_async`** from `vlm_service.py` entirely — `backend` alone has driven `run()`'s
  worker dispatch since the change above; `use_async` was never read anywhere. No behavior change.
- **`use_shm` is no longer a hardcoded module constant** in `timed_worker.py`. It's now threaded as
  a real parameter, the same way `backend` is: `args.use_shm` (default `False`) → `model_init()` →
  `VLLMService(use_shm=...)` and `cli()` → `VLMPrediction(use_shm=...)`; `retry_validate.stage()` /
  `collect()` / `drain_available()` / `_decode_response()` all take it as an explicit argument
  instead of importing the global.
- **The shm block itself is only allocated when `args.use_shm=True`.** Previously every example's
  `main.py` called `Shm(create=True, ...)` unconditionally in `__main__`, and both
  `VLLMService.run()` and `VLMPrediction.__init__` unconditionally reconnected to it
  (`Shm(create=False, ...)`) — even though `use_shm=False` (the default) meant the pool was never
  actually read from or written to. Now, with `use_shm=False`, `main.py` skips creating the
  shared-memory segment entirely and `self.shm` stays `None` in both `VLLMService`/`VLMPrediction`.
  `shm_name`/`free_slots` are still passed down unconditionally (cheap: a name string + an empty
  `mp.Queue`), so the call shape is unchanged either way.
- Default behavior is unchanged: `use_shm=False` everywhere, same as the old hardcoded constant.

## [1.5.0] - 2026-07-17

### ⭐ MAJOR UPGRADE — Continuous-Batching VLM Inference (~29% faster)

Full design history, problem analysis, and session-by-session decisions live in
`docs/v1.5_changes.md` (kept as a permanent working-notes record). Nothing here changes the DB
schema, the JSONL output format, or `project_specific.py` — see the Backward Compatibility Notice
below.

#### The story: v1.0 → v1.5 (multiprocessing → shm → async)
- **v1.0**: `VLLMService` and the pipeline stage ran on the same process/thread — direct calls,
  no IPC. Doesn't scale for StageWeaver, where up to 10 stages' worth of CPU-bound stage code
  shares one process; putting the VLM's own bookkeeping there too just adds more GIL contention.
  **v1.1** moved `VLLMService` to its own process, talking over `mp.Queue` request/response
  queues instead of direct calls.
- Queues pickle the payload and copy it across a kernel pipe boundary — fine for small messages,
  expensive for VLM prompts carrying multi-MB images. **v1.2** added a shared-memory (`shm`) pool:
  queues carry only a slot index, the payload is pickled directly into shared RAM, skipping the
  kernel copy.
- Neither change touches the inference call itself: `LLM.generate()` returns only when the *last*
  item in a batch finishes, so N−1 already-finished results sit idle behind one slow/hallucinating
  item, and the GPU idles as the batch drains toward empty with nothing new able to join mid-call.
  **v1.5** replaces the batch-boundary worker with vLLM's `AsyncLLM` engine for true continuous
  batching — a slow item now only delays itself, and because submission is fully asynchronous,
  communication and computation are genuinely overlapped rather than merely interleaved. This is
  where the ~29% win comes from. Full Problem 1–4 breakdown and per-phase design: `docs/v1.5_changes.md`.

#### Changed (see `docs/v1.5_changes.md` for full rationale)
- **Single batching point (Phase A/A2)** — batching used to live in three places and run in
  lock-step; now lives in exactly one (`VLMPrediction` in `prediction.py`). Failed items re-enter
  as retries riding along with fresh work; prompts are pre-staged into shm ahead of submission.
- **AsyncLLM continuous-batching worker (Phase C)** — `vlm_service.py` dispatches by default to
  `async_worker.py`'s `AsyncLLM`-based worker; `VLMPrediction` runs a dedicated prep thread +
  collector thread that stream continuously instead of waiting on batch boundaries.
- Old `use_shm=False` / `debug_timing` / `use_async=False` paths are kept (not deleted) as the A/B
  baseline for the benchmark numbers below and for re-measuring on future vLLM versions/hardware.

#### Verified — `examples/chandra_testing/testing_result.txt`
1024-image end-to-end run, Qwen3.5-VL via vLLM 0.19.0, identical dataset across all five configs,
0 failures and a fully consistent DB (state_2 = 100%) in every case:

| Configuration                                                          | Runtime  | Throughput   |
|-------------------------------------------------------------------------|---------:|-------------:|
| v1.0 (pre-v1.1 baseline — single process, multithreaded, direct calls) | 588.31 s | 1.74 img/s   |
| v1.5, `use_async=False`, shm off (mp.Queue only)                       | 577.47 s | 1.77 img/s   |
| v1.5, `use_async=False`, shm on                                        | 570.4 s  | 1.80 img/s   |
| v1.5, `use_async=True`, shm off                                        | 460.1 s  | 2.23 img/s   |
| **v1.5 (AsyncLLM + shm, default)**                                     | **456.75 s** | **2.24 img/s** |

Async is what drives the win: turning it on alone (shm off) already reaches 460.1s (~29%
faster); shm on top adds only ~3s more. mp.Queue and shm each contribute a small real win on their
own (~2%, ~1%) but nowhere near async's. Reasoning behind these numbers: `docs/v1.5_changes.md`.

#### ⚠️ Backward Compatibility Notice

**Upgrading from v1.0 to v1.5 requires no data migration**, but it does require a `main.py`
rewrite. The DB schema, state machine, JSONL output format, and `project_specific.py` interface
are all untouched. To move an existing v1.0 deployment to v1.5:
- Update the package (`pip install -e .` again after pulling v1.5).
- **`main.py` must be rewritten** — this is the one non-drop-in part of the upgrade. v1.0's
  `main.py` ran `VLLMService` in-process and passed `vlm_config, vlm_service` straight into
  `cli()`. v1.5 runs `VLLMService` as its own spawned process talking over queues + shared memory,
  so `main.py` now must: create `request_queue` / `response_queues` / `stop_event` and the shm
  pool *before* spawning anything (required for `spawn` fd inheritance); call the new
  `model_init(args, logger, request_queue, response_queues, stop_event, shm_name, free_slots)`,
  which just `.start()`s the process instead of handing back a callable service object; pass
  `request_queue` / `response_queue` / `shm_name` / `free_slots` / `stage_name` / `shutdown_event`
  into `cli()` instead of `vlm_config, vlm_service`; and shut down via the new `model_term()`
  (stop_event → join → GPU cleanup → `shm.cleanup()` → `cancel_join_thread()` on every queue)
  inside a `finally`, replacing the old `vlm_service.shutdown()`. Also size the shm pool to
  `>= 5*batch_size` slots (up from `3*batch_size`) — Phase C's continuous streaming holds more
  slots in flight than the old batch-boundary worker did; undersizing silently serializes
  throughput rather than deadlocking. Full reference implementation:
  `examples/chandra_testing/main_test.py` and `examples/vllm_testing/main.py`.
- **Existing JSONL shards and databases are directly compatible** — nothing needs re-processing,
  re-ingesting, or migrating. A run stopped mid-way under v1.0 and resumed under v1.5 (or vice
  versa) picks up exactly where it left off, same as any other v1.x → v1.y upgrade covered by the
  crash-safety guarantees elsewhere in this file.
- `project_specific.py` (your `PromptAndValidation` subclass) needs no changes — the hooks
  (`get_sampling_params`, `get_image_specific_prompt`, `preprocess_vlm_output`, `validate_output`,
  `write_local_json`) are called exactly the same way as before.

## [1.4.0] - 2026-07-10

- Added `use_shm` debug flag to `retry_validate_timed.py` / `timed_worker.py` — when `False`, bypasses shm and puts/gets the full payload directly on `request_queue`/`response_queue`, for A/B timing shm vs. no-shm.

## [1.3.0] - 2026-07-09

### Fixed — Exit Hang on Shutdown (Needs Multiple Ctrl+C)

- The pipeline would finish all work, print its final log lines, and then hang — `Ctrl+C` needed several presses and produced a pile of `KeyboardInterrupt` tracebacks instead of a clean exit. Root cause and full trace analysis: `src/vision_ingest/shm/exit_hang_postmortem.md`.
- **Root cause**: `Process.run()` returning is not the same as the OS process exiting. `multiprocessing`'s own atexit finalizers still need to run, and for every `mp.Queue` a process ever called `.put()` on, that includes joining the queue's background feeder thread — which blocks if the pipe's kernel buffer is full and nobody is left reading. `free_slots` is a free-list *pool* (both `VLLMService._worker()` and the stage process `.get()`/`.put()` it constantly, never drained to empty by design), so its last few writes at shutdown could land on a full pipe with no reader and hang forever — taking `vllm_proc.join()` in `model_term()` down with it. The same exposure applies to `request_queue` / `response_queues` in abnormal-exit paths (a Ctrl+C or exception mid-`collect()`/mid-`submit()`), even though they're normally drained to empty by the synchronous `submit()`/`collect()` pairing.
- **Fix**: `Queue.cancel_join_thread()` called in every process that writes to a given queue, inside a `finally` block that always runs on that process's exit (clean or crashed) — this opts the process out of the flush-and-join step, since by that point nothing further needs guaranteed delivery through the queue.
  - `vlm_service.py`: `VLLMService.run()`'s `finally`, right after `self.shm.close()`, cancels `free_slots` and every `response_queues` value.
  - `main.py`: `model_term()` now takes `request_queue`, `free_slots`, `response_queues` and cancels all three after `shm.cleanup()`.
  - `cli.py`: the stage process's own `finally`, right after `vlm_predictor.close(wait=False)`, cancels `request_queue` and `free_slots` too — redundant with `model_term()` in today's single-process layout (`cancel_join_thread()` is idempotent, safe to call twice), but added as **future-proofing** for when `main_cli()`/`cli()` runs in its own spawned process, separate from `model_term()`'s process: the queue-writer's exit finalizer is process-local, so one process calling `cancel_join_thread()` has no effect on another process's copy of the same queue — without this, splitting `cli()` into its own process would silently reintroduce the exit hang.

## [1.2.0] - 2026-07-04

- In the previous version, we were using mp.queue for interprocess communication between VLLM_Service and other processes. This involves a pickle.dump(), os kernel copy via pipe and then a pickle.load()
- We want to optimize this further using a shared memory approach, in which case we will be able to avoid the os kernel copy, which is the most expensive part of above interprocess communication.

- Here in main.py we are creating shm and we are only passing the shm name and free pool queue to all the processes this is so that all the processes can access the same shared memory and can read/write to it. This also helps and naturally extends to stageweaver where we can intialize the shared memory in main process and only pass the name to all other processes that a given VLLM Service shares.

### Implemented — Shared-Memory IPC

- Added `Shm` (`src/vision_ingest/shm/shm.py`): a fixed-name, fixed-slot pool backed by `multiprocessing.shared_memory.SharedMemory`. `request_queue` / `response_queues` now carry only an `int` slot index; the actual `VLLMRequest` / `(req_id, output)` payload is pickled directly into shared RAM, skipping the kernel pipe copy entirely.
  - One pool, bidirectional, one shared free-list (`mp.Queue` of free slot indices) — used for both request and response payloads.
  - `acquire()` / `release()` manage the free-list; `put()` raises `ValueError` on oversized payloads instead of corrupting adjacent slots.
  - Stale-block reclaim on `create=True` (handles a block left behind by a `SIGKILL`/OOM-killed previous run).
- **`vlm_service.py`**: `VLLMService` reconnects to the pool by name in `run()`; `_worker()` reads/writes shm slot indices instead of full payloads, releasing request slots immediately after unpickling and acquiring response slots after `model.generate()` completes.
- **`retry_validate.py`**: `submit()` / `collect()` / `generate_with_validation()` thread `shm` through. An oversized prompt that can't fit in a slot is marked as a failed attempt (`req_id=None`) rather than crashing or misaligning results with the rest of the batch.
- **`prediction.py` / `cli.py` / `main.py`**: `shm` threaded through `VLMPrediction` → `cli()` → `main_cli()` / `model_init()` / `model_term()`. The pool is created in `main.py` before `VLLMService` is spawned and `shm.cleanup()` (unmap + unlink) runs in `model_term()` after the process is joined.
- See `src/vision_ingest/shm/shm_ipc.md` and `v1.md` for the full design writeup (OS-level mechanics, slot sizing, ownership protocol, leak analysis).

## [1.1.0] - 2026-04-30

- **Upgrade VLM service from in-process multithreading to a separate multiprocessing worker**
  - Previously `VLLMService` and the pipeline stage ran on the same process, sharing the GIL via
    threads — no IPC, just direct function calls into the VLM. This doesn't scale for
    StageWeaver: running up to 10 stages' worth of CPU work and the VLM's own Python-side
    bookkeeping on one process just means they all fight over the same GIL. `LLM.generate()`
    itself releases the GIL during the actual generate call, but everything around it (request
    prep, response handling) doesn't, and that's exactly the work StageWeaver multiplies by
    stage count.
  - Moved `VLLMService` to its own process. Added a request queue, a response queue, and an
    event, all from `multiprocessing`. `VLLMService` now interacts with all other processes only
    through these queues and the event — no direct function calls.

## ⚠️ Backward Compatibility Notice

**v1.0.0 uses PostgreSQL, solving all multi-node and slow db issues we faced with SQLite**
- No change to project_specific.py. So, older version of project_specific.py will work with this version.
- Major change in db(SQLite -> PostgreSQL). Therefore main.py and args passed to main.py will change. So, older version of main.py will not work with this version. 
- For now support for older SQLite DBs has been retained. But can not guarantee that future changes/upgrades to Postgres based DB will be provided for SQlite DBs. So, it is recommended to migrate to Postgres DBs as soon as possible.

## [1.0.0] - 2026-03-30

### ⭐ MAJOR UPGRADE — Multi-Backend Database Support

#### Added
- **Unified Database Layer with Pluggable Backends**
  - Refactored DB module with single `DB` class exposing unified interface
  - Backend-agnostic API: identical interface for SQLite and PostgreSQL
  - Pluggable backend submodules: `db_sql/` (SQLite) and `db_postgres/` (PostgreSQL)
  - Automatic backend selection: provide `sqlite3_db_path` for SQLite or `pg_config` dict for PostgreSQL
  - Single initialization call handles backend differences transparently

- **PostgreSQL Support**
  - Full PostgreSQL backend implementation with identical feature parity to SQLite
  - Supports all existing operations: state management, batching, path fetching, metrics
  - Tested and verified working at scale (12M+ image paths)
  - Production-ready performance for large-scale deployments

- **DB Performance Testing Suite** (`examples/db_performance_testing/`)
  - Benchmarking tools for SQLite and PostgreSQL side-by-side comparison
  - `bench_ops_sql.py` — SQLite operation benchmarks
  - `bench_ops_postgres.py` — PostgreSQL operation benchmarks
  - Documented results at 12M-image scale with comprehensive metrics
  - Performance testing README with setup and interpretation guides

- **Database Driver Enhancements**
  - Added `db_driver_postgres.py` for PostgreSQL-specific pipeline integration
  - Enhanced `db_driver.py` for SQLite pipelines
  - Support for multiple database modes: `serve`, `stop`, `serve-and-ingest`
  - Backward compatible with existing SQLite workflows

#### Changed
- **DB Module Reorganization**
  - Consolidated duplicate code across SQLite/PostgreSQL implementations
  - Moved specialized logic to backend-specific submodules
  - Updated `db_readme.md` to document unified interface and backend selection
  - Updated `cli_readme.md` with driver and configuration details
  - Removed redundant backend-specific CLI logic; now handled transparently

#### Verified
- ✅ SQLite backend fully tested and working
- ✅ PostgreSQL backend fully tested and working  
- ✅ Database performance testing suite completed with benchmarks
- ✅ Multi-stage pipeline support maintains compatibility
- ✅ State management and recovery mechanisms functional across backends

#### Pending Testing
- ⏳ CLI integration testing with both backends
- ⏳ Full backward compatibility verification (single-stage, multi-stage, existing DBs)
- ⏳ Main README and QUICKSTART updates for v1.0

#### Notes
- Backward compatibility maintained; existing SQLite databases are expected towork unchanged with unified interface
- PostgreSQL option available for teams needing multinode support and a much more robust database solution at scale
- Zero breaking changes to public APIs(function definitions).

---

## [0.4.4] - 2026-02-24

- **FSx Lustre compatibility — SQLite multi-node initialization fix**
  - Switched journal mode from WAL → PERSIST in `db_queries.py`
    - WAL uses mmap()-based shared memory (`-shm` file) which breaks on network filesystems
    - PERSIST mode uses only fcntl() file locks, which Lustre's LDLM supports across nodes
    - PERSIST keeps the journal file (zeros header on commit) instead of DELETE which
      creates/deletes it every transaction — avoids expensive metadata ops on network FS
    - Trade-off: writers block readers (no concurrent read+write), but correct cross-node locking
  - Upgraded `synchronous` NORMAL → FULL (DELETE mode needs FULL for crash safety on network FS)
  - Removed `wal_autocheckpoint` pragma (not applicable in DELETE mode)
  - Commented out `checkpoint_wal()` method (no WAL to checkpoint)
  - Increased `busy_timeout` 10s → 60s for network FS lock latency
  - Increased `sqlite3.connect(timeout=)` 60s → 120s
  - Fixed `isolation_level = None` ordering: now set **before** `_init_databases()` instead of after
    - Python's default `isolation_level=""` auto-begins implicit transactions, causing
      `"cannot start transaction within transaction"` errors with explicit `BEGIN IMMEDIATE`
  - Retry logic improvements in `_init_databases()`:
    - Increased attempts 3 → 5
    - Exponential backoff with jitter (`2^attempt + random(0,1)s`) instead of fixed 1s sleep
    - Close and reconnect on retry instead of reusing potentially broken connections
  - All old code commented out (not deleted) with explanations for traceability

## [0.4.3] - 2026-02-24
- MAJOR BUG in `prediction.py`- it caused the prompt of the first batch to be sent for entire batch, which is a disaster.
- Please check and discard results of any runs done with v0.3.9, since they would have been affected by this bug.

## [0.4.2] - 2026-02-19
- changed quickstart.md and examples
- but havent tested changed examples

## [0.4.1] - 2026-02-16

- **Multi-node database initialization**
  - Split PRAGMAs into `init_database_pragmas()` (persistent, one-time) and `init_connection_pragmas()` (per-connection)
  - Connection pragmas now set before `isolation_level = None` to prevent "PRAGMA inside transaction" errors
  - Added check-before-init pattern: lightweight table existence check before running DDL to reduce contention
  - `init_schema()` uses `BEGIN IMMEDIATE` to acquire write lock upfront and fail fast
  - Added ROLLBACK in retry loop to clean up stale transaction state between attempts
  - Added `main_logger` None guards to prevent secondary exceptions masking initialization failures
  - All initialization now occurs within retry protection before enabling autocommit mode

## [0.4.0] - 2026-02-14

- **LLM vs VLM Model Support**
  - Added `is_llm` parameter to `cfg.yaml` to distinguish between pure LLM models and Vision Language Models (VLMs). (Default is False)
  - For `is_llm: true` models, `AutoTokenizer` is used; for `is_llm: false` models, `AutoProcessor` is used (multimodal support)
  - Updated imports in `vllm_config.py` to include `AutoTokenizer` from transformers
  - Added conditional logic in `VLLMConfig.__init__()` to initialize the correct processor based on `is_llm` flag
  - Implemented message format handling: for LLM models, content lists are flattened to extract only text; for multimodal models, structured content with images is preserved
  - This ensures compatibility with both LLM-only and vision-enabled models

- **Thinking Mode Support**
  - Added `enable_thinking` parameter to `PromptAndValidation.__init__()` in `project_specific.py` (default: False)
  - Parameter is passed to `apply_chat_template()` in `vllm_config.py` to enable model thinking/reasoning mode
  - Allows project-specific control over model thinking behavior for improved reasoning quality

## [0.3.9] - 2026-02-12

- added a feature for `StructuredOutputsParams` in `prompt_validation_object` to specify the output format to vllm. This helps vllm generate better outputs.
- Added entire SamplingParams in `prompt_validation_object` to specify sampling params to vllm. This helps in better control over the generation.
- You can now pass the full model_path to `VLLMConfig` and it will directly look for that path in the config file.
- error in prepare prompts of one image should not fail the entire batch, added try catch around that and logging for the same. This is in `get_prompts` function in `prediction.py`
- changes in `def iter_paths()` in db.py.

## [0.3.8] - 2026-01-31

- next_stage_db_marked logging
- in `db.py`
  - added retry got init_databases
  - `get_state_pct(state)` feature added to db
- in `cli_utils.py` using the get_state_pct to decide if we should quit or wait
  - Check if we're done (state 0 is 0% means nothing left to fetch or for other process to be ready)

- ###### MAJOR ISSUE IN GET_LOGGER FIXED -> please verify
---

## [0.3.7] - 2026-01-28

### Added
- **Graceful shutdown mechanism** (see `figures/changelog_v0_3_7.png`)
  - `shutdown_event` (threading.Event) accepted in `cli()` and propagated through: `cli()` → `process_pipeline()` → `process_batch()` → `vlm_predict()` → `get_prompts()` / `send_to_vlm()` → `generate_with_validation()`
  - `GracefulShutdown` exception raised when shutdown_event is set, propagates up entire call stack to `cli()`
  
  - **In `process_pipeline()`**:
    - Checks shutdown_event before fetching each batch; raises immediately if set
    - Fetched paths added to `fetch_paths` set for cleanup in finally block
  
  - **In `process_batch()`**:
    - Passes shutdown_event to `vlm_predict()`
    - No shutdown check after `vlm_predict()` because remaining operations (`mark_failed`, `local_pool.submit`, `writer.enqueue`) are fast and handled in finally block
    - Next batch fetch in `process_pipeline()` rechecks shutdown_event before continuing
  
  - **In `vlm_predict()`**:
    - `get_prompts()`: Checks shutdown_event while waiting for futures from `get_cur_img_prompt_fn` and `process_one` in `vllm_config.py`
    - `send_to_vlm()` → `generate_with_validation()`: Submits requests to vlm_service, then polls shutdown_event every 0.25s while waiting for results
  
  - **Cleanup flow**: `cli()` catches `GracefulShutdown`, logs it, and executes finally block to reset all fetched paths to original state

### Changed
- **VLM service lifecycle management**
  - `VLMPrediction.close()` only closes `vlm_service` if initialized internally
  - External `vlm_service` instances (passed to constructor) are not closed; caller retains ownership

### Fixed
- Path recovery on shutdown: All paths in `fetch_paths` set reset to original state in `cli()` finally block
- Clean termination on Ctrl+C or SIGTERM with proper resource cleanup

---

## [0.3.6] - 2026-01-27

- added a warning in writer.py when marking in next stage db results in, incomplete rows changed.

---

## [0.3.5] - 2026-01-26

### Added
- **Process-local stuck path recovery**
  - CLI now tracks all fetched paths and resets them to original state on shutdown
  - Prevents paths from getting stuck in state 1 when process crashes or is interrupted
  - `DB.reset_paths_to_state()` for targeted path recovery (process-isolated, safe for multi-node)
  - Automatic cleanup in `cli.py` finally block resets only paths fetched by current process
- **Tested Finally block behavior**
  - Executes on Ctrl+C interrupts and normal process completion
  - Does not execute on `docker stop` (this is expected behavior as it prevents graceful shutdown)
- **DB insertion performance (12M image_paths)**
  - First ingestion: 301.04s; Second ingestion (duplicate paths): 106.97s
  - DB sizes:
    - `os.path.getsize()` values: Main DB: 2781.18 MB; Seen cache: 2641.16 MB
    - Actual sizes: Main DB: 346.7 MB; Seen cache: 260.6 MB
    - ⚠️ `os.path.getsize()` returns incorrect values (see TODO)
  - `mark_done()` latency remains in low millisecond range


### Changed
- `db_driver.py`: Updated `logs_dir` parameter in `ingest()` method
- Added empty image path list support in `vllm_config.py`
  - Enables text-only prompts by passing empty list in `project_specific.py`
- Main logger now emits checkpoint logs for all processed images crossing checkpoint intervals

---

## ⚠️ Backward Compatibility Notice

**v0.3.4 expects a different project_specific.py when compared to v0.2.0 and below**
- This change to `prediction.py` required change in `project_specific.py`, older version of `project_specific.py` will not work with this version of `prediction.py`. 
- Since there is no change in other db related code, so existing db will work perfectly fine.


## [0.3.4] - 2026-01-26

- **Multi-image prompt support**: Added ability to send multiple images in a single VLM request
  - Updated `vllm_config.py` to accept `List[Union[str, List[str]]]` for image paths
  - Supports both single-image prompts (backward compatible) and multi-image prompts
  - All images are loaded in parallel and attached to the same prompt
- **Custom image path support**: Changed `project_specific.py` and `prediction.py` to allow passing custom image paths for each prompt
  - Enables Set-of-Mark (SOM) or other preprocessed image variants
  - `get_image_specific_prompt()` can now return either a single path or list of paths
- **Full VLM object storage**: Added feature to store the complete vllm_output object for token counting and metrics
  - Placeholder function `post_process_full_vllm_object(pred: RequestOutput)` added in retry_validate.py (returns None currently) 

---
#### --------- All Readmes have been updated to reflect till v0.3.3 changes ---------
---

## ⚠️ Backward Compatibility Notice

**v0.3.3 introduces database schema changes when compared to v0.2.0 and below**
- Adds new processing state: `4 = Upstream-Ready`
- Adds `shard_path` column to the `images` table

Existing databases are **auto-migrated on startup** and remain backward compatible.

---

## [0.3.3] - 2026-01-25
- added shard_path as an argument to `mark_done` in JSONLWriter.
- added a `get_shard_paths` method to DB class to retrieve shard paths for given image paths.
- Readmes updated based on v0.3.3 changes.

---

## [0.3.2] – 2026-01-25

### Added
- **Shard Path Tracking**
  - Added `shard_path TEXT` column to `images` table to record JSONL shard locations
  - `mark_done(paths, shard_path=None)` accepts optional shard path
  - `mark_state_transition()` supports atomic state + shard updates
  - NULL values allowed for backward compatibility and minimal storage overhead

- **Multi-Stage Pipeline Support**
  - Added new state `4 = Upstream-Ready`
  - Enables cross-stage coordination for multi-stage pipelines

### Changed
- **Database Initialization & Migration**
  - `init_schema()` rewritten to be fully idempotent and concurrency-safe
  - Automatic schema migration for:
    - Missing `shard_path` column
    - Missing state `4` in `state_counts`
  - Uses `INSERT OR IGNORE` for safe concurrent initialization
  - Self-healing logic for partial or corrupted schemas
  - Defensive validation raises `RuntimeError` if required columns are missing
  - Supports fresh DBs, partially initialized DBs, and concurrent initializers

- **API Updates**
  - `mark_done(paths, shard_path=None)`
  - `mark_state_transition(cur, paths, from_state, to_state, shard_path=None) -> int`
  - `_execute_state_transition()` supports optional `shard_path`

### Fixed
- **State Count Drift**
  - Fixed `insert_paths_batch()` to count only rows actually inserted
  - Prevents `state_counts` drift when using `INSERT OR IGNORE`
  - Now relies on `changes()` for accurate insert counts

---

## [0.3.1] – 2026-01-25

- Added `next_stage_db_duration_ms` to `JSONLWriter` logging
- Ensured `cur_db_done` is called **after** marking `next_stage_db` ready
  - Guarantees retry safety if a crash occurs mid-transition
  - Next-stage transitions remain idempotent, preventing data loss

---

## [0.3.0] – 2026-01-25

### Added

* **Multi-Stage Pipeline Support**

  * Introduced **State 4 (Upstream-Ready)** for cross-stage coordination
    (`0=Pending, 1=In-Progress, 2=Done, 3=Failed, 4=Upstream-Ready`)
  * Each stage uses an **independent SQLite database**, enabling isolation and independent scaling
  * `DB.__init__(fetch_state)` and `fetch_and_lock_batch()` now support stage-aware fetching
  * Added `mark_previous_stage_done()` for downstream signaling
  * `JSONLWriter` can signal the next stage via `next_stage_db_path`
  * CLI support: `--fetch-state {0,4}`, `--next-stage-db-path`

* **Database & Pipeline Improvements**

  * Extended `state_counts` to include state `4`
  * Enabled `PRAGMA busy_timeout = 5000` to reduce write contention
  * `seen_cache_path` is now optional; workers no longer open seen-cache connections
  * Health checks and verification gracefully handle missing seen cache
  * Added SQLite design rationale to `db/readme.md`

* **Code & Initialization Simplification**

  * Unified all state transitions via `mark_state_transition()`
  * Centralized DB error handling with `_execute_state_transition()`
  * Removed duplicate logic across state update methods
  * Simplified worker initialization to connect only to the main DB

---
### Everything below till v0.2.0 is completely backwards compatible
---
## [0.2.0] - 2026-01-24

### Added
- **Package configuration**: Added `pyproject.toml` for proper Python package distribution
  - MIT license
  - Author: Srihari Bandarupalli
  - Package metadata and project URLs
  - Minimal configuration (dependencies managed separately via `init_env.sh`)

### Changed
- Restructured repository into a proper Python package layout
  - Core library moved to `src/vision_ingest/`
  - Example code moved to `example/example1/`
  - Documentation moved into the package directory
- Renamed package from `vision_ingestion_pipeline` → `vision_ingest`
- Converted all internal imports to absolute package imports (`vision_ingest.*`)
- Removed project-specific imports from library code; project logic is now passed explicitly
- Codebase is now ready for editable installs (`pip install -e .`)
- Updated documentation to reflect new `src/vision_ingest/` package structure
  - QUICKSTART.md now references correct paths for package files
  - Example readme.txt clarified to show imports from installed package
  - All internal documentation paths updated to reflect src/ structure
- Tested the new structure with examples and editable installs to ensure seamless usage after `pip install -e .`
- Added `MODULE_TO_RUN` variable in `run_pipeline.sh` to ensure clean up on error/interruption.

### Notes
- Dependencies still managed via `bash_scripts/init_env.sh` for specific installation order
- Package can now be imported anywhere in the environment after installation

---

## [0.1.0] - 2026-01-23

### Added
- Support for multiple pipelines sharing a single VLM instance
  - `cli()` accepts optional `vlm_config` and `vlm_service`
- Project-specific prompt/validation abstraction via `prompt_validation_object`
- Optional run-specific log directory control via `run_specific_log_path_in_args`

### Changed
- Updated `drivers/cli.py` to support multi-pipeline and project-specific configuration
- Documentation updated to describe resource sharing and customization hooks

### Notes
- `VLMPrediction` now reuses provided VLM resources instead of always initializing new ones
- Enables efficient multi-pipeline deployments with shared model state
