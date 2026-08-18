# JSONL Writer

**Author:** Srihari Bandarupalli

**Module:** `modules/writer.py`

---

## 📋 Purpose

The JSONL Writer provides **non-blocking, crash-safe persistence** of VLM inference results to disk. It runs in a background thread, accepting results through a queue and writing them to JSONL shard files with atomic fsync-DB synchronization guarantees.

### Key Responsibilities

- **Async writes:** Main pipeline never blocks on disk I/O
- **Crash safety:** Ensures JSONL files and DB state stay perfectly synchronized
- **Shard management:** Automatically rotates files when size limits are reached
- **Performance tracking:** Logs detailed write/fsync/DB metrics

---

## 🏗️ Architecture

### Thread-Based Async Design with Atomic Flush Protocol

The writer uses a **queue-buffer-fsync** pattern that separates the main pipeline from disk I/O while ensuring perfect consistency between JSONL files and database state:

```
Main Thread (Pipeline)                Background Thread (Writer)
─────────────────────                ──────────────────────────
                                     
VLM Results                          
     ↓                               
writer.enqueue(obj) ────────────────→ Queue.put(obj)
     ↓ (non-blocking)                      ↓
Continue processing                   Buffer.append(obj)
     ↓                                     ↓
More results                         Count lines in buffer
     ↓                                     ↓
writer.enqueue(obj) ────────────────→ Buffer full? (250 lines)
     ↓                                     ↓
Continue...                           ┌──────────────────────────────┐
                                      │   FLUSH GROUP OPERATION      │
                                      ├──────────────────────────────┤
                                      │                              │
                                      │ 1. Write buffer → JSONL      │
                                      │    Lines in memory buffer    │
                                      │           ↓                  │
                                      │ 2. Fsync to disk             │
                                      │    Lines persisted on disk   │
                                      │           ↓                  │
                                      │ 3. DB.mark_done(paths,       │
                                      │    shard_path=current_shard) │
                                      │    DB state = done (2)       │
                                      │    shard_path recorded       │
                                      │           ↓                  │
                                      │ 4. Next-stage handoff        │
                                      │    (if next_stage_db set)    │
                                      │    mark_previous_stage_done()│
                                      │           ↓                  │
                                      │ 5. Clear buffers             │
                                      │    Ready for next batch      │
                                      │           ↓                  │
                                      │ 6. Check rotation            │
                                      │    New shard if >1GB         │
                                      │                              │
                                      └──────────────────────────────┘
```

**This protocol ensures atomic consistency:** JSONL files and database state remain perfectly synchronized even during crashes, power failures, or unexpected shutdowns.

---

## 🛡️ Crash Safety Analysis

Understanding what happens when things go wrong is critical. Here's how the write-fsync-DB protocol handles every failure scenario:

**Scenario 1: Crash Before Fsync**
```
State: Lines in memory buffer, NOT on disk, DB state = in-progress (1)
Result: Lines lost, DB still shows in-progress
Recovery: reset_stuck_paths() in drivers/db_driver.py will fix paths stuck in state=1
Guarantee: ✅ No corruption, safe to retry
```

**Scenario 2: Crash After Fsync, Before DB Update**
```
State: Lines on disk, DB state still = in-progress (1)
Result: JSONL has records, DB doesn't acknowledge them
Recovery: recovery.py detects mismatch between JSONL and DB state, removes lines from JSONL
Guarantee: ✅ Consistency restored, no duplicate processing
```

**Scenario 3: Crash After DB Update**
```
State: Lines on disk, DB state = done (2)
Result: Perfect consistency
Recovery: No action needed, validation passes
Guarantee: ✅ Perfect state, processing continues
```

**Scenario 4: Background Thread Exception**
```
State: Exception in _run() during write/fsync/DB operations
Result: Writer immediately marks itself as closed, logs error details
Recovery: Pipeline fails fast on next enqueue(), preventing data loss
Guarantee: ✅ No silent failures, clear error visibility, prevents queue buildup
```

---

## 🔄 Core Operations

### Public Interface

The pipeline interacts only through `writer.enqueue(obj)`.  
This call is fully non-blocking: objects are pushed into the queue and the pipeline continues without waiting.

### Background Workflow

<div align="center">

**Queue** → **Buffer** → **Flush (250 lines)** → **Fsync** → **DB Update**

</div>

The `_run()` loop continuously consumes items from the queue. Each item is appended to the in-memory buffers. Once the buffer size reaches `fsync_every_lines` (250), `_flush()` is triggered to:

- **Write** the batch to the shard file  
- **Update** the database with the corresponding metadata

> **💡 Note:** The writer has its own DB instance and logger to ensure transaction isolation. Each concurrent DB operation (orchestrator's `fetch_batch()` vs. writer's `mark_done()`) runs on a separate connection, preventing "cannot start a new transaction" errors that would occur if sharing a single DB connection.

---

## 🛠️ Thread Safety & Error Handling

### Thread Safety Mechanisms

The writer uses **lock-free closed state** with `threading.Event`:
- `is_set()` has zero lock contention—critical for high-frequency `enqueue()` calls
- Proper memory barriers ensure instant cross-thread visibility
- Perfect for producer-consumer signaling patterns

### Error Handling Strategy

**Background thread error handling:**  
The `_run()` loop catches any exception (JSON serialization, file I/O, fsync, DB operations), then:
1. Sets closed event (thread-safe)
2. Logs full error details
3. Safely drains queue to prevent deadlock
4. Exits gracefully

**Pipeline protection:**  
Once closed, `enqueue()` immediately raises `RuntimeError`, preventing silent data loss, queue buildup, and resource exhaustion. **Fail-fast with clear error visibility.**

---

## 🔗 Integration Points

### Database Integration

After a successful fsync, the writer atomically marks all flushed paths as complete through `db.mark_done(paths, shard_path=current_shard)`. This single transaction updates:
- `images` table: state = 2 (done)
- `images` table: shard_path = current shard location (for lineage tracking)
- `state_counts` metadata: increment done count

**New in v0.3.2:** The `shard_path` column enables result lineage tracking—you can query the DB to find which JSONL shard contains a specific image's results.

> **⚡ Performance:** Processing all 250 paths in one SQL transaction is orders of magnitude faster than 250 individual UPDATEs, while guaranteeing atomicity—either all paths are marked done with shard paths, or none are.

**Error resilience:** The writer's DB instance handles errors internally and logs to the writer's logger, keeping writer code clean. Any JSONL-DB mismatches that occur during crashes are detected and corrected by the recovery module (see Crash Safety Analysis above).

### Multi-Stage Pipeline Support

**New in v0.3.0:** When `next_stage_db` is configured, the writer performs atomic cross-stage transitions:

```python
# After marking current stage done:
self.db.mark_done(paths, shard_path=self.current_shard)  # Current stage: state=2

# Atomically hand off to next stage:
if self.next_stage_db:
    self.next_stage_db.mark_previous_stage_done(paths)  # Next stage: state=0
    # Current stage: transition from state=2 → state=4 (upstream-ready)
```

**Crash Safety:** The order matters for retry safety:
1. Mark current stage done first (state 1→2, with shard_path)
2. Then insert into next stage (state=0)
3. Finally mark current stage as upstream-ready (state 2→4)

If crash happens mid-transition:
- ✅ Current stage shows state=2 (done) - work is preserved
- ⚠️ Next stage might not have the paths yet - re-run handoff on retry
- ✅ Idempotent operations prevent duplicate processing

**Use Case:** Multi-stage pipelines (e.g., layout extraction → table extraction) where each stage uses independent DBs for isolation and scaling.

### Shard Rotation

The writer checks for rotation **after every successful flush cycle**. If the current shard exceeds 1GB, a new shard is created with a zero-padded index (e.g., `results_00001.jsonl`).

**The critical design choice:** Rotation happens **only after** the complete write → fsync → DB update sequence finishes. This ensures:

- **Atomic batches never split across shards** — All 250 lines stay together, eliminating partial-batch complexity
- **Old shards are fully verified** — Data is on disk (fsync), acknowledged by DB, and internally consistent
- **Recovery operates on finalized files** — No partial writes or ambiguous state in closed shards
- **Simplified validation** — Each shard is either complete or still being written to

> **💡 Design Insight:** Shards can exceed 1GB by up to 250 lines (~5-15MB overshoot). This intentional tolerance keeps atomic batches intact and dramatically simplifies recovery logic.

---

### ⚠️ Critical: JSONL Shard Immutability Requirement

**Once a shard is rotated (closed), it MUST remain read-only forever.**

This is not a recommendation—it's a **fundamental correctness assumption** that the recovery system depends on.

**Why This Matters:**

The recovery system validates only the **last 250 lines** of the **active shard** (the one still being written to). This bounded validation is what makes recovery fast—at billion-scale, scanning all shards would take hours.

**The assumption:** Old shards (already rotated and closed) are immutable because:
1. They were fully fsync'd before rotation
2. All paths in them were marked `state=done` in the database
3. Recovery validated them when they were the active shard

**What Happens If You Edit Old Shards:**

```
Scenario: You manually edit line 5000 in results_00042.jsonl

Current State:
  ✅ Shard was rotated weeks ago (closed, immutable)
  ✅ DB says all paths in that shard have state=done
  ✅ Recovery only checks the active shard (e.g., results_00089.jsonl)

You Edit the Shard:
  ❌ Line 5000 is now corrupted or inconsistent
  ❌ DB still says that path is state=done (no way to detect change)
  ❌ Recovery won't catch it (doesn't re-validate old shards)
  ❌ System consistency is silently broken

Result:
  - Downstream analysis reads corrupted data
  - No warnings, no errors, no indication anything is wrong
  - Database thinks work is done, but JSONL is invalid
  - Manual recovery requires matching DB state to corrupted files
```

**Allowed Operations on Old Shards:**

✅ **DO:**
- Read files for analysis (`jq`, `pandas`, custom scripts)
- Copy files for backup or archival
- Parse files with read-only tools
- Stream files to downstream systems
- Compress files for long-term storage

❌ **DON'T:**
- Edit any lines in closed shards (even fixing typos)
- Append new data to closed shards
- Delete lines from closed shards
- Truncate closed shards
- Reorder lines in closed shards
- Run any process that opens shards in write mode

**Why We Accept This Limitation:**

Fully validating all shards after every crash would require reading billions of lines (hours at scale), comparing every line against DB state (expensive joins), and handling partial corruption with complex rollback logic. Instead, we guarantee **tail consistency** (last 250 lines of active shard) and require **operational discipline** (don't edit old files). This is the same trade-off Kafka, Git, and Elasticsearch make—committed data is immutable.

**📖 See:** Main [README.md](../README.md) section "Operational Realities & Limitations" and [`modules/recovery_readme.md`](recovery_readme.md) for recovery validation logic.

---

## ⚙️ Lifecycle

### Initialization

The writer is set up in the orchestration flow (`main.py` → `drivers/cli.py`) with its own DB instance and logger (separate from the orchestrator's DB instance). It computes the current shard index (from recovery or by scanning existing shards), opens the shard file in append mode, creates thread-safe queues and buffers, then launches a background daemon thread.

**Why separate DB instance?** The writer's `mark_done()` operations run concurrently with the orchestrator's `fetch_batch()` operations. Using separate DB connections prevents "cannot start a new transaction while existing transaction is going on" errors.

> **⚡ Non-blocking:** Setup completes instantly—the writer runs independently once started.

### Shutdown

When the pipeline finishes, it calls `writer.close(timeout=60)`. The writer places a stop sentinel in the queue, flushes any remaining buffered data, performs a final fsync, and closes file handles. If the background thread doesn't exit within the timeout, an error is logged but shutdown continues gracefully. Recovery handles any partial writes on next startup.

---

## 📊 Logging

Writer events are logged to **`jsonl_writer.log`** in structured JSON format for detailed performance monitoring and debugging.

### Key Events Logged

| Event | Description |
|-------|-------------|
| `fsync_complete` | Write, fsync, and DB timing per flush |
| `shard_rotated` | Shard size and new shard info |
| `background_thread_error` | Critical errors in the background writer thread |
| **Performance warnings** | Slow fsync (>1s), slow write (>5s), large lines (>10KB) |

> **📖 For detailed log format information, see:** [`logger_readme.md`](../logger_readme.md)

---

## 📜 Operational Contract (Assumptions & Guarantees)

<details open>
<summary><strong>1. Assumptions (Required for Correctness)</strong></summary>

<br/>

* **Single pipeline per node** ⚠️ — avoids shard-index collisions and interleaved writes.
* **Writer faster than pipeline** 🚀 — prevents queue buildup and RAM growth.
* **Fsync treated as durable** 💾 — required for atomic write → fsync → DB ordering.
* **VLM output small** ✉️ — keeps shard size predictable and batches stable.
* **DB layer stays healthy** 🗄️ — no long external locks that stall `mark_done()`.
* **Reprocessing allowed** 🔁 — state=1 items can be safely reset, making unflushed items recoverable.

</details>

<details open>
<summary><strong>2. Guarantees (What the Writer Provides)</strong></summary>

<br/>

* **Atomic flush groups** 🔒 — strict write → fsync → DB.mark_done sequence ensures JSONL + DB consistency.
* **Crash-safe for all flushed data** 🛡️ — fsynced records never duplicate or disappear after recovery.
* **Fail-fast on background thread errors** ⚡ — writer closes immediately and enqueue() raises.
* **Clean shutdown** ✔️ — final buffer is flushed; tail inconsistencies handled by recovery.

</details>

<details open>
<summary><strong>3. Non-Negotiable Rules (Must Follow)</strong></summary>

<br/>

* **Always call `writer.close()` before exit** 🧯 — ensures final batch is persisted.
* **Never enqueue after close** 🚫 — prevents silent data loss.
* **Avoid long DB locks** 🧱 — no long SELECT, VACUUM, or external operations on the DB.
* **Use local disk only** 📂 — network FS breaks fsync guarantees and atomicity.

</details>

<details open>
<summary><strong>4. Out of Scope (Intentionally Not Guaranteed)</strong></summary>

<br/>

* **Unflushed queue items not durable** — they re-enter via state reset.
* **Logs not atomic** — log file may have partial entries after a crash.
* **Multiple writers unsupported** — concurrent pipelines cause undefined behavior.

</details>

---

## 🎯 Final Summary: The Art of Robust Persistence

The JSONL Writer is the result of **careful engineering**, shaped by examining each failure mode, concurrency edge case, and performance constraint that shows up in large-scale VLM pipelines.

Its strength comes from how every design choice works together:

* **🧵 Queue-based async writes** keep the pipeline fully non-blocking.
* **🔒 Precise write → fsync → DB-update ordering** guarantees consistency across crashes and power failures.
* **�️ Thread-safe error handling** with `threading.Event` ensures fail-fast behavior without sacrificing performance.
* **�📦 Batched flush groups** stabilize I/O costs and reduce overhead.
* **📁 Post-flush shard rotation** ensures every shard is internally consistent, enabling simple, bounded recovery.
* **🗃️ Idempotent DB state transitions** eliminate duplicate or lost work and make retries safe.
* **🛠️ Robust shutdown logic** preserves data integrity even during abrupt stops.

Every part of this module—from batching strategy to rotation timing—was shaped through deliberate decision-making, validation, and iteration. The end result is a writer that is **fast, crash-safe, predictable, and easy to operate**, even under heavy load or unexpected failures.
