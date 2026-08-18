# Pipeline Orchestrator: End-to-End Lifecycle Management

**Author:** Srihari Bandarupalli  
**Entry Point:** `main.py`  
**Orchestrator Module:** `drivers/cli.py`  
**Design Philosophy:** From Cold Start to Graceful Shutdown at Billion-Image Scale

> **Note:** The pipeline is invoked via `main.py`, which parses arguments and calls `cli()` with:
> - `prompt_validation_object`: Project-specific prompt/validation logic (see `project_specific.py`)
> - `vlm_config`, `vlm_service`: Optional shared VLM resources for multi-pipeline setups
> - `run_specific_log_path_in_args`: Flag to use pre-existing log paths vs. creating timestamped directories
> - `shutdown_event`: Optional `threading.Event` for externally coordinated shutdown (e.g., multi-pipeline orchestrators)

---

## 🎯 The Central Orchestration Problem

At billion-image scale, the hardest problem isn't processing images—it's **coordinating 6 components with fundamentally different lifecycles, state transitions, and failure modes**. The orchestrator doesn't eliminate this complexity—it **organizes** it through three mechanisms: dependency-ordered initialization, `FSYNC_EVERY_LINES` alignment, and fail-fast error propagation.

### The Components and Their Lifecycles

The orchestrator manages 6 components, each with distinct characteristics:

| Component | Lifecycle | State Transitions | Init Failure | Post-Init Crash |
|-----------|-----------|-------------------|--------------|-----------------|
| **MainLogger** | First init, last close | N/A (stateless logging) | Abort (no logs) | Logs incomplete but recoverable |
| **Database** | Long-lived connection | pending(0) → processing(1) → done(2)/failed(3) | Abort (no state storage) | State inconsistent → recovery handles |
| **Recovery** | Runs once at startup | corrupted → validated → truncated | Abort (corruption undetected) | Already validated, safe |
| **Writer** | Background thread | queue → buffer → disk → fsync | Abort (no persistence) | Partial writes → recovery truncates |
| **VLMPrediction** | Per-batch invocation | paths → prompts → inference → results | Abort (GPU unavailable) | Model loaded, safe to restart |
| **PipelineMetrics** | Stateless accumulator | window → checkpoint → reset | Never fails (no I/O) | Never fails (no I/O) |

### The Coordination Challenge

These components must work together so that:

1. **Initialization respects dependencies** → MainLogger first, then orchestrator DB, then Writer (with its own DB instance)
2. **State transitions stay synchronized** → JSONL and DB align at checkpoint boundaries (local JSON is best-effort)
3. **Crashes leave system recoverable** → Deterministic state at `FSYNC_EVERY_LINES` boundaries
4. **Failures propagate clearly** → No tangled error handling, exit cleanly
5. **Operators diagnose bottlenecks** → Real-time metrics reveal limiting resources

### The Three Orchestration Mechanisms

**1. Dependency-Ordered Initialization**  
Fail-fast if anything is broken. No partial initialization states. Either all components are ready or pipeline aborts cleanly.

**2. FSYNC_EVERY_LINES Alignment**  
Synchronization primitive across all components. Writer fsync, DB commits, and checkpoint logging all occur at the same 250-image boundaries. Crashes resolve to the nearest boundary—at most 250 images replayed on restart.

**3. Fail-Fast Error Propagation**  
Detect errors early, log diagnostics, exit cleanly. Recovery handles inconsistent state on restart—no heroic mid-run recovery attempts.

### System Topology

```
┌─────────────────────────────────────────────────────────────┐
│              CLI ORCHESTRATOR SCOPE                         │
│                                                             │
│  ┌──────────┐    ┌─────────┐         ┌─────────────┐      │
│  │ Database │←───│ Recovery│         │ VLMPredict  │      │
│  │  (state) │    │ (startup)│        │  (inference)│      │
│  └────┬─────┘    └─────────┘         └──────┬──────┘      │
│       │                                      │              │
│       │          ┌──────────┐                │              │
│       └─────────→│  Writer  │←───────────────┘              │
│                  │ (persist)│                               │
│                  └────┬─────┘                               │
│                       │                                     │
│  ┌────────────┐       │        ┌────────────────┐          │
│  │MainLogger  │←──────┴───────→│PipelineMetrics │          │
│  │ (logging)  │                │   (tracking)   │          │
│  └────────────┘                └────────────────┘          │
│                                                             │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
                       [JSONL Shards]
                       [Local JSON Files]

External Process: [db_driver_postgres.py] → [Database] 
                  (image ingestion, runs separately)
```

**→ For complete architectural context and design philosophy, see main [README.md](../README.md) sections on Architectural Pillars and Design Philosophy.**

The rest of this document explains how the CLI orchestrator coordinates these components through initialization, processing, and shutdown.

---

## Lifecycle Phase 1: Initialization (Cold Start)

**Goal:** Establish the crash safety invariant—after Writer initialization, the system is recoverable.

### Database Backend Selection

The orchestrator now supports two database backends. The backend is selected automatically based on whether PostgreSQL connection arguments are present in `args`:

```python
# Postgres mode — args.pg_host is set
db_path_or_config = {
    'host': args.pg_host,
    'port': args.pg_port,
    'dbname': args.pg_dbname,
    'user': args.pg_user,
    'password': args.pg_password
}

# SQLite mode — fallback when pg_host is not set
db_path_or_config = os.path.join(args.db_path, "images.db")
```

The `DB` class accepts either form. All downstream components (Writer, next-stage handoff) receive the same `db_path_or_config` value and handle backend selection internally—the orchestrator does not need to branch on this after initialization.

### The Dependency Chain

Components initialize in strict dependency order to prevent cascading failures:

```
Run Metadata (run_id, json_prefix, log_dir)
    ↓
┌───────────────────────────────────────────┐
│ MainLogger (first - needed by all)        │ ← All components log here
└───────────────┬───────────────────────────┘
                ↓
┌───────────────────────────────────────────┐
│ Database (needs main_logger for errors)   │ ← State persistence (orchestrator DB)
│ SQLite path OR Postgres config dict       │   Backend auto-selected from args
└───────────────┬───────────────────────────┘
                ↓
┌───────────────────────────────────────────┐
│ Recovery (needs db to check states)       │ ← Validates last run
└───────────────┬───────────────────────────┘
                ↓
┌───────────────────────────────────────────┐
│ Writer (creates own DB instance + logger) │ ← Returns last_shard_path
│ Separate DB connection for concurrency    │   (Prevents transaction conflicts)
│ Next-stage DB also SQLite or Postgres     │
└───────────────┬───────────────────────────┘
                ↓
┌───────────────────────────────────────────┐
│ VLMPrediction (independent GPU init)      │ ← Model loading + prompt_validation_object
│ Uses shared vlm_config/vlm_service if set │   (Enables multi-pipeline resource sharing)
└───────────────┬───────────────────────────┘
                ↓
┌───────────────────────────────────────────┐
│ PipelineMetrics (stateless tracker)       │ ← No dependencies
└───────────────┬───────────────────────────┘
                ↓
         Start Processing
```

**Why This Order Matters:**

- **Logger first:** All components log to MainLogger, so it must exist before anything else initializes
- **Recovery before Writer:** Recovery truncates corrupted shards and returns `last_shard_path`. Writer needs this to know which shard to append to
- **VLMPrediction independent:** GPU initialization doesn't depend on DB/Writer state. We initialize it after Writer purely for logical flow (could be parallelized)

**Critical Boundary:** After Writer initialization, the system is in a **recoverable state**. If the process crashes, recovery will detect the last valid shard on next startup. This is the crash safety invariant that enables fail-fast error handling throughout the processing phase.

---

## Architecture: Component Contracts and State Machines

### Unified Component Reference

The orchestrator coordinates 6 components through narrow, well-defined contracts. This table shows the complete API surface and failure handling:

| Component | Lifecycle | Key Methods | Contract | Failure Handling | See Also |
|-----------|-----------|-------------|----------|------------------|----------|
| **MainLogger** | First init, last close | `log_info()`, `log_checkpoint()` | Centralized structured logging | Abort startup (no logs possible) | [`logger_readme.md`](../logger_readme.md) |
| **Database** | Long-lived connection | `fetch_batch(n)` | Atomically: state 0→1 (or 4→1 multi-stage), returns List[paths] | Log to assigned logger & raise | [`db/readme.md`](../db/readme.md) |
| | | `mark_done(paths)` | Atomically: state 1→2 (called by Writer's DB) | N/A (Writer handles) | |
| | | `mark_failed(paths)` | Atomically: state 1→3 | Log to assigned logger & raise | |
| | | `is_done(path)` | Read-only query (recovery validation) | N/A (recovery handles) | |
| | | `reset_paths_to_state(paths, target)` | Atomically: state 1→target (0 or 4) | Log to main_logger & raise | |
| | | `close()` | Commits pending transactions | N/A (cleanup phase) | |
| **Recovery** | Runs once at startup | `recover_last_shard()` | Validates/truncates JSONL, returns shard_path | Abort startup (corruption undetected) | [`modules/recovery_readme.md`](../modules/recovery_readme.md) |
| **Writer** | Background thread | `enqueue(obj)` | Non-blocking queue push | Raise immediately | [`modules/writer_readme.md`](../modules/writer_readme.md) |
| | | `close()` | Blocks until queue drained + final fsync | N/A (cleanup phase) | |
| **VLMPrediction** | Per-batch invocation | `vlm_predict(cur, next)` | Returns (results, next_prompts), parallel CPU/GPU | Raise immediately | [`modules/prediction_readme.md`](../modules/prediction_readme.md) |
| **PipelineMetrics** | Stateless accumulator | `update()`, `get_checkpoint()` | Accumulate stats between checkpoints | Never fails (no I/O) | (no separate readme) |

**→ For database state machine details, see [db/readme.md](../db/readme.md) and main [README.md](../README.md).**

---

## Lifecycle Phase 2: Continuous Processing (Hot Loop)

The main processing loop is implemented in `drivers/cli_utils.py` as `process_pipeline()`. The orchestrator in `cli.py` initializes all components and then delegates to `process_pipeline()`, passing:

- All initialized components (`db`, `writer`, `vlm_predictor`, etc.)
- `fetched_paths`: a `set` owned by the orchestrator for shutdown cleanup (see below)
- `shutdown_event`: the optional external `threading.Event` for cooperative shutdown

**The Batch Lifecycle:** Fetch → Process → Write → Checkpoint

#### Main Loop Structure

```
Initialize: Bootstrap first batch prompts (CPU-only)

while True:
    ┌─────────────────────────────────────┐
    │ 1. Fetch next_batch from DB         │ ← State transition: 0→1
    │    (atomically marks as processing) │   Paths added to fetched_paths set
    └─────────────────────────────────────┘
                  ↓
    ┌─────────────────────────────────────┐
    │ 2. Call vlm_predict(cur, next)      │
    │    ├─ GPU: Inference on cur_batch   │ ← Parallel execution
    │    └─ CPU: Prepare prompts for next │ ← (VLM handles internally)
    └─────────────────────────────────────┘
                  ↓
    ┌─────────────────────────────────────┐
    │ 3. Enqueue results to Writer        │ ← Non-blocking (queue.put)
    │    (async, returns immediately)      │   Completed paths removed from fetched_paths
    └─────────────────────────────────────┘
                  ↓
    ┌─────────────────────────────────────┐
    │ 4. Mark failures in DB              │ ← State transition: 1→3
    └─────────────────────────────────────┘
                  ↓
    ┌─────────────────────────────────────┐
    │ 5. Shift batches (next → cur)       │
    └─────────────────────────────────────┘
                  ↓
    ┌─────────────────────────────────────┐
    │ 6. Checkpoint boundary?             │
    │    If (count % FSYNC_EVERY_LINES):  │
    │    └─ Log metrics, DB state, Writer │
    └─────────────────────────────────────┘

Exit conditions:
  - 5 consecutive empty fetches (5-second timeout)
  - shutdown_event.is_set() checked at key points
  - GracefulShutdown exception raised internally

Flush: Process final prepared batch (GPU-only, no next batch)
```

#### Empty Batch Handling: The Five-Second Rule

When `db.fetch_batch()` returns an empty list (no pending images), the pipeline implements a retry mechanism with 1-second delays. This occurs when:

- All images are processed (completion scenario)
- Image ingestion (via `db_driver_postgres.py`) hasn't added new images yet
- Ingestion rate is slower than processing rate (pipeline waiting for data)

The pipeline sleeps for 1 second between retries to avoid hammering the database with high-frequency polling. After 5 consecutive empty fetches (`EMPTY_BATCH_TIMEOUT`), it logs a warning and exits gracefully.

**Why 5 Seconds?**

The timeout balances three competing concerns:

| Too Short (e.g., 1 second) | Current (5 seconds) | Too Long (e.g., 60 seconds) |
|----------------------------|---------------------|----------------------------|
| ❌ Pipeline exits before slow ingestion catches up | ✅ ~500 images in flight at 100 img/s | ❌ Pipeline waits minutes after completion |
| ❌ Frequent restarts = recovery overhead | ✅ Responsive to real completion | ❌ Slow reaction to genuine errors |
| ❌ Operators can't distinguish "done" vs "waiting" | ✅ Distinguishes temporary gaps from completion | ❌ Delayed feedback on ingestion failures |

At typical throughput (100 images/sec), 5 seconds represents ~500 images in flight—a reasonable buffer for ingestion irregularities. Metrics record these wait times to help identify ingestion bottlenecks.

**Future Enhancement:** The timeout should be made configurable for production deployments where images may arrive in irregular batches (e.g., hourly dumps vs. continuous streaming).

---

## Lifecycle Phase 3: Graceful Shutdown

**Design Principle:** Shutdown is deliberately simple—four entry points, one cleanup path.

The pipeline supports four shutdown scenarios, all converging to identical cleanup operations:

```
Four Shutdown Paths Converge:

Path A: Natural Completion
  └─ No more images in DB (5 consecutive empty fetches)
  └─ Process final prepared batch
  └─ → finally { cleanup() }

Path B: Cooperative Shutdown (shutdown_event)
  └─ External threading.Event set by orchestrator / multi-pipeline manager
  └─ Checked at key points in process_pipeline(); raises GracefulShutdown
  └─ → except GracefulShutdown { log } → finally { cleanup() }

Path C: Keyboard Interrupt (Ctrl+C)
  └─ User terminates pipeline
  └─ Current batch completes
  └─ → except KeyboardInterrupt { log } → finally { cleanup() }

Path D: Unhandled Exception
  └─ VLM crash / Disk full / GPU OOM
  └─ Exception propagates to cli()
  └─ → except Exception { log } → finally { cleanup() }

Cleanup Operations (always executed via finally block):
┌────────────────────────────────────────────┐
│ 1. Reset stuck paths (fetched_paths set)   │ ← state 1 → fetch_state (0 or 4)
│ 2. writer.close()                          │ ← Blocks until queue drained + fsync
│ 3. local_pool.shutdown(wait=True)          │ ← Waits for local JSON threads
│ 4. prompt_pool.shutdown(wait=True)         │ ← Waits for prompt preparation threads
│ 5. vlm_predictor.close(wait=False)         │ ← Non-blocking GPU teardown
│ 6. db.close()                              │ ← Closes DB connection
│ 7. Log final metrics (log_shutdown)        │ ← Runtime, throughput, failure counts
└────────────────────────────────────────────┘
```

### Stuck Path Reset on Shutdown

The orchestrator maintains a `fetched_paths` set that tracks every path transitioned to state 1 (in-progress) by this process. On any shutdown path, paths still in this set—meaning they were fetched but never completed—are atomically reset:

```python
db.reset_paths_to_state(fetched_paths, target_state=fetch_state)
# fetch_state is 0 (normal) or 4 (multi-stage upstream-ready)
```

This ensures no paths are permanently stranded in state 1 after a clean shutdown. The reset returns these paths to the correct pending state so other workers (or a restart of this process) can pick them up. The number of paths reset is logged as a warning so operators can correlate it with throughput anomalies.

> **Note:** This reset only occurs on **clean shutdown** (any of the four paths above reaching the `finally` block). A hard kill (`SIGKILL`) or power failure will leave paths in state 1. The separate `reset-stuck` maintenance command in `db_driver_postgres.py` handles this case.

### Cooperative Shutdown via `shutdown_event`

For multi-pipeline orchestration (e.g., running several `cli()` instances in parallel), pass a shared `threading.Event`:

```python
shutdown_event = threading.Event()
# ... in your orchestrator:
shutdown_event.set()  # signals all pipelines to stop cleanly
```

The event is checked at defined safe points in `process_pipeline()` before each DB fetch and before starting VLM inference. When set, `GracefulShutdown` is raised, which propagates to `cli()` for logging and cleanup. This enables coordinated teardown without relying on OS signals.

**Why This Simplicity Works**

Shutdown is tractable because there's **no complex state to reconcile**. The design choices throughout the pipeline enable this elegance:

1. **Stateless batches:** No partial work to cancel—each batch completes atomically
2. **Self-managing components:** Writer handles its queue, DB handles transactions, no orchestrator involvement
3. **Alignment synchronization:** At most 250 images in flight (will be replayed on restart)
4. **fetched_paths reset:** In-progress paths are returned to pending state, not abandoned
5. **Finally block guarantee:** Cleanup executes regardless of exit path (completion, interrupt, exception)

**Flush Phase on Natural Completion**

When the pipeline exits naturally (no more images), it processes the final prepared batch:
- `cur_batch` has prompts ready (CPU work already done in previous iteration)
- No `next_batch` (nothing to prepare)
- VLM runs GPU inference only
- Results written to JSONL
- Then cleanup executes

On interrupt or exception, the flush phase doesn't run—recovery will replay at most `FSYNC_EVERY_LINES` images on restart.

**What Shutdown Doesn't Do**

The orchestrator intentionally avoids complex shutdown logic:

- ❌ **No rollback transactions** (DB commits at checkpoint boundaries, not on shutdown)
- ❌ **No partial write reconciliation** (Writer's fsync alignment handles this)
- ❌ **No in-flight work cancellation** (batches complete atomically or not at all)
- ❌ **No retry-on-failure** (fail-fast philosophy, recovery handles restart)

This simplicity is a **design feature, not a limitation**. Complex shutdown coordination indicates poor separation of concerns—our architecture eliminates the need for it.

**→ For CPU/GPU parallelism implementation details, see [modules/prediction_readme.md](../modules/prediction_readme.md).**

---

## Error Handling Philosophy

### Error Propagation Flow

The pipeline implements hierarchical error handling with clear boundaries:

```
Error Propagation Hierarchy:

Component Error → Log to assigned logger → Raise Exception
  (Orchestrator DB logs to main_logger)          ↓
  (Writer's DB logs to writer_logger)   process_batch() catches
                                              ↓
                                    Propagates to process_pipeline()
                                              ↓
                                    Propagates to cli()
                                              ↓
                             ┌────────────────────────────┐
                             │ except GracefulShutdown    │ ← cooperative stop
                             │ except KeyboardInterrupt   │ ← Ctrl+C
                             │ except InterruptedError    │ ← OS interrupt
                             │ except Exception           │ ← all other failures
                             └────────────┬───────────────┘
                                          ↓
                                  finally: cleanup()
```

**Inside `process_batch`:**
- `get_prompts()` errors → raise immediately
- `send_to_vlm()` errors → raise immediately  
- `db.mark_failed()` errors → log to orchestrator's logger & raise
- `local_pool.submit()` (JSON write) → log & raise (writes individual VLM outputs to separate JSON files using the local threadpool)
- `writer.enqueue()` errors → raise immediately
- **Logging:** First & last batches logged explicitly; intermediate failures logged in VLM prediction logs

**Inside `process_pipeline`:**
- `shutdown_event` checked before fetch and before inference → raises `GracefulShutdown`
- `db.fetch_batch()` errors → log to orchestrator's logger & raise
- `process_batch()` errors → propagate upward (already logged at source)

**Inside `cli`:**
- Initialization errors → raise (abort before processing starts)
- `GracefulShutdown` → caught explicitly, logged with shutdown reason
- `KeyboardInterrupt` / `InterruptedError` → caught explicitly, logged
- All other exceptions → caught, logged, shutdown reason recorded
- All paths → `finally` block executes cleanup (including stuck path reset)

This hierarchical error handling ensures that:
1. Errors are logged at the point of detection (maximum context)
2. Critical errors propagate to abort processing
3. Cleanup always executes regardless of failure mode
4. Stuck paths are always returned to pending state on clean exit

**→ For architectural design philosophy and system-level guarantees, see main [README.md](../README.md) Design Philosophy section.**

---

## ⚠️ Local JSON Files: Best-Effort Guarantee

The pipeline writes results to two locations with **different consistency guarantees**:

### Consistency Hierarchy

| Output | Guarantee Level | Synchronization | What It Means |
|--------|----------------|-----------------|---------------|
| **JSONL Shards + Database** | ✅ **Atomic (Guaranteed)** | Write → fsync → DB.mark_done() | Perfectly synchronized, crash-safe, source of truth |
| **Local JSON Files** | ⚠️ **Best-Effort Only** | Thread pool submission, no verification | May be missing even if DB says `state=done` |

### How Local JSONs Are Written

```python
# In process_batch() - Two parallel write paths:

# Path 1: GUARANTEED (atomic protocol)
for obj in success_objs:
    writer.enqueue(obj)  # ← Queued for write → fsync → DB.mark_done()

# Path 2: BEST-EFFORT (fire-and-forget)
for obj in success_objs:
    local_pool.submit(save_local_json, obj)  # ← No error checking
```

**What We Do:**
- Submit JSON write tasks to a thread pool (`ThreadPoolExecutor`)
- Call `local_pool.shutdown(wait=True)` during cleanup to let threads finish

**What We Don't Do:**
- Track individual task futures to check for exceptions
- Retry failed writes
- Verify files were actually created
- Mark a separate "local_json_written" flag in the database

### Why This Limitation Exists

Making local JSONs fully reliable requires tracking futures, retry logic for transient failures, and error handling when local writes fail but JSONL+DB succeeded. This adds 10-15% throughput overhead—at billion-scale, that's days of compute time.

**Design Decision:** We prioritized **pipeline throughput** over **local JSON reliability** because:
1. JSONL shards are the authoritative source of truth
2. Local JSONs are a convenience layer for per-image access
3. Missing local JSONs can be regenerated from JSONL shards if needed

### When Local JSONs Might Be Missing

| Scenario | JSONL+DB State | Local JSON State |
|----------|----------------|------------------|
| **Disk full during write** | ✅ Present, marked done | ❌ Missing |
| **Permission denied** | ✅ Present, marked done | ❌ Missing |
| **Network FS timeout** | ✅ Present, marked done | ❌ Partial or missing |
| **Python exception in thread** | ✅ Present, marked done | ❌ Missing |
| **System crash** | ✅ Present (fsync'd) | ❌ May be missing |

**Recovery:** Extract missing data from JSONL shards using `jq` or parse shards directly in your analysis pipeline.

**📖 See:** Main [README.md](../README.md) section "Operational Realities & Limitations" for complete context.

---

## Appendix A: Configuration Constants

These constants control pipeline behavior and synchronization:

| Constant | Default Value | Purpose | Configured In | Tuning Trade-Offs |
|----------|---------------|---------|---------------|-------------------|
| `FSYNC_EVERY_LINES` | 250 | Checkpoint alignment boundary for crash recovery | `cli.py` | See below |
| `BATCH_SIZE` | 32 | Images per batch (balances GPU utilization and memory) | `cli.py` | Higher = better GPU utilization but more memory. Typical: 16-64 |
| `EMPTY_BATCH_TIMEOUT` | 5 seconds | Wait time before graceful exit when no images available | `cli.py` | Lower = faster completion detection, higher = tolerates slow ingestion |
| `SHARD_SIZE_LIMIT` | 1 GB | JSONL file rotation threshold | `writer.py` | Larger = fewer files but slower recovery scans |

### Multi-Stage Pipeline Parameters

**New in v0.3.0:** Enable cross-stage coordination for independent pipeline stages.

| Parameter | Type | Purpose | Usage |
|-----------|------|---------|-------|
| `--fetch-state` | int (0 or 4) | Which state to fetch from DB | Default=0 (pending). Use 4 for downstream stages fetching upstream-ready images |
| `--next-stage-db-path` | str | Next stage SQLite DB path for handoff | SQLite only. When set, Writer atomically marks current stage done (state=4) and next stage ready (state=0) |
| `--next-stage-pg-host` etc. | str/int | Next stage Postgres connection args | Postgres mode. Takes precedence over `--next-stage-db-path` when set |

### Database Backend Parameters

| Mode | Required Args | Notes |
|------|---------------|-------|
| **SQLite** | `--db-path` | Path to directory; `images.db` created inside |
| **PostgreSQL** | `--pg-host`, `--pg-port`, `--pg-dbname`, `--pg-user` | `--pg-password` optional (trust auth) |

Backend selection is automatic: if `--pg-host` is present in args, Postgres config dict is built; otherwise SQLite path string is used. The same logic applies to the next-stage DB.

### FSYNC_EVERY_LINES Tuning Guide

The value of 250 balances three competing concerns:

**Fsync Overhead:**
- Each fsync blocks for ~5-50ms depending on disk (SSD vs. HDD, RAID configuration)
- Lower values = more frequent fsyncs = reduced throughput
- Higher values = fewer fsyncs = higher throughput (diminishing returns beyond ~500)

**Recovery Granularity:**
- At most `FSYNC_EVERY_LINES` images replayed on restart
- 250 images ≈ 30 seconds at 100 img/s typical throughput
- Lower values = faster recovery but lower throughput
- Higher values = slower recovery but higher throughput

**Temporal Resolution:**
- Checkpoint logs appear every `FSYNC_EVERY_LINES` images
- Lower values = finer-grained bottleneck detection
- Higher values = coarser metrics (harder to pinpoint issues)

**Recommended Values by Workload:**

| Workload | FSYNC_EVERY_LINES | Rationale |
|----------|-------------------|-----------|
| Development/testing | 50-100 | Faster iteration, fine-grained logging |
| Production (fast disks) | 250 (default) | Balanced trade-off |
| Production (slow disks) | 500-1000 | Reduce fsync overhead |
| High-churn environments | 100-200 | Minimize replay time on frequent restarts |

**Critical:** Whatever value you choose, ensure it's consistent across the `--fsync-every-lines` argument passed to `main.py` (which propagates to both `cli.py` and `writer.py`). Mismatched values break the alignment guarantee and corrupt recovery state.