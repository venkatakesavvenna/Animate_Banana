# Database Layer — A Scalable Concurrent PostgreSQL Architecture

**Author:** Srihari Bandarupalli  
**Module:** `db/`

---

## 🎯 The Challenge

Building a vision pipeline that processes hundreds of millions (100M–1B+) of images presents an operational constraint that most database designs ignore: **you cannot afford to re-process the same image twice**. At scale, even a 0.1% duplication rate means wasting compute on 1M redundant operations. Crashes and network failures are inevitable—your database must guarantee exactly-once processing semantics without sacrificing throughput.

The system must support:
- **Continuous ingestion** - New images arrive constantly, must be deduped and queued without blocking processing
- **Concurrent processing** - Multiple workers fetch and process batches simultaneously, with true row-level locking
- **Crash recovery** - Detect and reset stuck work after unexpected shutdowns
- **Progress tracking** - Instant visibility into pipeline health (pending, in-progress, done, failed)
- **Zero redundancy** - Never insert the same path twice, never process the same image twice
- **Multi-node safety** - Multiple machines sharing a single database without filesystem-level lock quirks

This is the problem the database layer solves.

---

## 🎨 Design Philosophy

> **Why PostgreSQL?** The original system used SQLite, which worked well for single-node deployments but hit fundamental limitations at multi-node scale: WAL mode breaks on network filesystems (FSx Lustre), `fcntl()` file locks have unpredictable latency across nodes, and writers serialize globally. PostgreSQL provides true MVCC concurrency, `FOR UPDATE SKIP LOCKED` for zero-contention batch fetching, and `COPY` for million-row/sec bulk loading — all without filesystem-level lock coordination.

### 1. **Single Database, Atomic Deduplication**

The SQLite version used **two coordinated databases** — a main DB for state management and a separate seen cache for deduplication. This was necessary because SQLite's coarse-grained locking meant ingestion writes would block worker fetches.

PostgreSQL eliminates this complexity entirely. Deduplication is handled atomically inside `INSERT ... ON CONFLICT (path) DO NOTHING` — a single statement that inserts new paths and silently skips duplicates with no separate cache needed. This removes an entire database, its schema, its triggers, and the 3-phase commit ordering that was required to keep them consistent.

**What was removed:**
- `seen_cache.db` — the entire database
- `SeenCacheQueries` class — all seen-cache SQL operations
- `seen_conn` parameter and connection management in `DB.__init__`
- The 3-phase seen-then-insert ingestion ordering and its crash-safety reasoning
- Trigger-based count management in the seen cache

### 2. **Short-Lived Cursors, Persistent Connection**

Each `DB` instance maintains a persistent connection but creates and closes cursors per operation. PostgreSQL's MVCC allows concurrent readers and writers without blocking. Short-lived cursors release server-side resources immediately after each transaction, maximizing concurrency.

**Future scaling path:** The current single-connection design can be replaced with `psycopg.pool.ConnectionPool` (or `AsyncConnectionPool` for async workloads) by swapping `self.conn` for a pool. The pool's context manager pattern (`with pool.connection() as conn:`) is a drop-in replacement — the pool handles connection return automatically.

### 3. **O(1) State Tracking via Cached Counts**

Instead of expensive `COUNT(*)` queries, the system maintains a `state_counts` table with running totals updated atomically on every state transition. At 100M+ rows, this turns health monitoring from seconds into milliseconds.

### 4. **Idempotent State Transitions**

All state-changing operations (`mark_done`, `mark_failed`) only update paths **currently in the expected source state**:

```sql
UPDATE images SET state = 2
WHERE path = ANY($1) AND state = 1  -- Only update if currently in-progress
```

**Why idempotency matters:** If a worker crashes after marking paths done but before acknowledging, retrying the operation won't double-count completions. The `state_counts` update uses `cursor.rowcount` (actual modified rows) not `len(paths)` (requested changes), ensuring counts stay accurate even if some paths were already transitioned.

### 5. **Self-Healing Verification**

Despite careful design, count drift can occur from manual DB modifications, Postgres crashes, or logic bugs. Instead of requiring manual intervention, the system detects and repairs drift automatically during initialization and maintenance windows.

### 6. **Idempotent Schema Initialization**

Database initialization is fully idempotent and concurrency-safe:

- **Fresh databases:** Creates complete schema (tables, indexes, initial `state_counts` rows)
- **Existing databases:** `CREATE TABLE IF NOT EXISTS` and `INSERT ... ON CONFLICT DO NOTHING` are safe to run repeatedly
- Postgres handles DDL race conditions internally via its lock manager — no retry/backoff loops needed (unlike the SQLite version which retried 5× with exponential backoff)

**Why This Matters:** Multiple workers can call `init_schema()` concurrently without corruption. No migration scripts, no version checking, no `PRAGMA table_info()` introspection.

### 7. **Dual Ingestion Modes**

`add_paths_from_source()` supports two modes via the `use_copy` parameter:

- **`use_copy=False` (default)** — Uses `INSERT ... ON CONFLICT DO NOTHING` via `executemany()`. Safe for incremental ingestion with duplicates. Returns accurate count of actually-inserted rows.
- **`use_copy=True`** — Uses PostgreSQL `COPY FROM STDIN` for maximum throughput (~1M+ rows/sec). No duplicate detection — use only for first-load on an empty table. If any duplicate exists, the entire batch rolls back with `UniqueViolation`.

---

## 🔄 State Machine & Lifecycle

### Five-State Model

Every image path transitions through a strict lifecycle:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Ingestion    Fetch       Process      Mark                 │
│  ─────────►  ──────►  ────────►  ─────────►                 │
│   state=0     state=1    [work]      state=2/3              │
│   PENDING   IN-PROGRESS           DONE/FAILED               │
│                                        │                    │
│     ◄──────────────                    │ (multi-stage)      │
│      Crash Recovery                    ↓                    │
│     (reset state 1→0)             state=4                   │
│                                  UPSTREAM-READY             │
└─────────────────────────────────────────────────────────────┘
```

| State | Name | Meaning | Operations That Set This State |
|-------|------|---------|-------------------------------|
| **0** | Pending | Ingested, ready for processing | `add_paths_from_source()` inserts new paths |
| **1** | In-Progress | Locked by a worker, currently processing | `fetch_batch()` atomically transitions 0→1 (single/first stage) or 4→1 (intermediate stages) |
| **2** | Done | Successfully processed and saved | `mark_done()` transitions 1→2 after writing results |
| **3** | Failed | Processing failed (corrupted file, inference error, etc.) | `mark_failed()` transitions 1→3 for error handling |
| **4** | Upstream-Ready | (Multi-stage only) Processed by previous stage, ready for current stage | `mark_previous_stage_done()` in **current stage's database** called by **previous stage** after its completion (transitions 0→4) |

### Multi-Stage Pipeline Support

**State 4 enables independent pipeline stages with cross-stage coordination:**

- Each stage uses its own isolated database (separate Postgres databases or schemas)
- **Previous stage** marks images as **state=4** in the **current stage's database** after completing its own processing
- A path enters state=4 **only after** its previous stage has finished processing it

**Critical `fetch_batch()` behavior:**
- **First stage or single-stage pipeline**: `fetch_batch()` selects from **state=0** (pending) → transitions to state=1
- **Intermediate/downstream stages**: `fetch_batch()` selects from **state=4** (upstream-ready) → transitions to state=1
- This ensures each stage only processes images that have been successfully handled by the previous stage

**Example multi-stage workflow:**
```
Stage 1 (Layout Extraction):
  - Fetches from state=0 → processes → marks state=2 in stage1 DB
  - Cross-stage handoff: marks state=4 in stage2 DB

Stage 2 (Table Extraction):
  - Fetches from state=4 → processes → marks state=2 in stage2 DB
  - Cross-stage handoff: marks state=4 in stage3 DB (if exists)
```

**State transitions in multi-stage context:**
```
stage1 DB: 0 → 1 → 2             (Stage 1 processes its own work)
stage2 DB: 0 → 4 → 1 → 2        (Stage 1 marks 4, Stage 2 fetches 4)
stage3 DB: 0 → 4 → 1 → 2        (Stage 2 marks 4, Stage 3 fetches 4)
           ↑
  Initial state after path inserted by previous stage
```

### Shard Path Tracking

The `images` table includes a `shard_path TEXT` column to maintain output lineage:

**Purpose**: Records the **output JSONL shard file path** where this image's processing results are stored

**When it's set**: Updated by `mark_done()` when transitioning state 1→2, storing the JSONL shard file path (e.g., `/output/shard_0001.jsonl`)

**Why it matters**:
- **Result lineage**: Given an image path, instantly find which JSONL shard contains its output
- **Recovery & debugging**: Reconstruct processing history and locate specific results
- **Audit trail**: Track which batch/shard each image was written to

**Schema details**:
- Column type: `TEXT` (NULL allowed for images not yet processed)
- Updated atomically with state transitions to maintain consistency
- Not indexed by default (add if frequent shard-based queries needed)

### Transition Guarantees

- **Atomic batching**: `fetch_batch()` uses `FOR UPDATE SKIP LOCKED` in a CTE — no two workers ever receive the same row, even under high concurrency
- **Idempotent marking**: `mark_done()`/`mark_failed()` only update paths currently in state-1
- **Count consistency**: Every transition updates `state_counts` table atomically with actual `cursor.rowcount`

### Crash Recovery

**Failure scenario**: A worker crashes mid-processing (killed, OOM, network failure)

**Symptoms**: Paths stuck in state-1 (in-progress) indefinitely

**Recovery**: Run `reset_stuck_in_progress()` to atomically transition all state-1 paths back to state-0. See `drivers/db_driver_readme.md` for crash recovery CLI commands.

---

## ⚙️ Core Workflows — The Operational Rationale

### Workflow 1: Continuous Ingestion (`add_paths_from_source`)

**Problem**: How do you add 10M new images without blocking workers fetching batches from the DB?

**Solution**: Use PostgreSQL's `INSERT ... ON CONFLICT DO NOTHING` for atomic deduplication in a single statement. No separate seen cache, no pre-filtering step, no multi-database coordination.

#### Dataflow

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────────────────┐
│   Folder    │────►│ Walk & Batch     │────►│ INSERT ... ON CONFLICT        │
│   Scan      │     │ (32K images)     │     │ DO NOTHING                    │
│   or File   │     │                  │     │ (dedup + insert in one step)  │
└─────────────┘     └──────────────────┘     └───────────────────────────────┘
                                                          │
                                                    Short cursor
                                                   (releases locks)
```

#### Why This Design Works

1. **Atomic dedup** — `ON CONFLICT DO NOTHING` handles duplicates inside the database engine, no application-level filtering needed
2. **No seen cache** — Eliminates an entire database and its associated complexity (triggers, counts, 3-phase commit ordering)
3. **Short-lived cursors minimize lock duration** — Each batch operation releases locks in milliseconds
4. **COPY mode for bulk loading** — `use_copy=True` enables ~1M+ rows/sec for initial data loads on empty tables

For detailed logging metrics, see `logger_readme.md`.

---

### Workflow 2: Atomic Batch Fetching (`fetch_batch`)

**Problem**: How do you ensure two workers never process the same image?

**Solution**: Use a CTE with `FOR UPDATE SKIP LOCKED` to atomically select and lock rows. This is a fundamental improvement over the SQLite version:

```sql
WITH picked AS (
    SELECT path FROM images
    WHERE state = %s
    LIMIT %s
    FOR UPDATE SKIP LOCKED
)
UPDATE images
SET state = 1
WHERE path IN (SELECT path FROM picked)
RETURNING path;
```

**Why `FOR UPDATE SKIP LOCKED`?**
- `FOR UPDATE` — Locks selected rows so no other transaction can modify them
- `SKIP LOCKED` — Instead of waiting for locked rows, instantly skips them and grabs unlocked ones
- Combined with `LIMIT` — Each worker gets a non-overlapping batch in a single atomic statement
- No serialization — Multiple workers can fetch simultaneously without waiting for each other

**Comparison with SQLite version**: SQLite used `BEGIN IMMEDIATE` which acquires a database-level exclusive write lock — only one worker could fetch at a time, and all others waited. PostgreSQL locks only the specific rows being fetched, allowing truly concurrent batch fetching.

**State selection logic**:
- **Single-stage or first stage**: Fetches from `state=0` (pending) → transitions to `state=1`
- **Intermediate stages**: Fetches from `state=4` (upstream-ready) → transitions to `state=1`
- The caller specifies `fetch_state` parameter (defaults to 0 for backward compatibility)

---

### Workflow 3: State Marking (`mark_done`, `mark_failed`)

**Problem**: After processing, how do you mark paths complete without introducing count drift?

**Solution**: Use `WHERE state = 1` to only update paths currently in-progress, then update counts by actual rows changed (not requested count). This idempotent design ensures safe retries—if operations are repeated, already-transitioned paths won't be double-counted.

PostgreSQL's `path = ANY(%s)` with a Python list maps directly to a PostgreSQL array parameter — no dynamic placeholder building needed (unlike SQLite where `IN (?,?,?...)` placeholders had to be constructed per-batch).

---

### Workflow 4: Health Monitoring (`get_db_health`)

**Problem**: How do you monitor pipeline progress without expensive table scans?

**Solution**: Use cached `state_counts` table for O(1) lookups. At 100M rows, this provides millisecond-level health checks versus seconds with direct `COUNT(*)` queries. Enables real-time dashboard updates, crash recovery assessment, and capacity planning.

Database size is reported via `pg_database_size(current_database())` — an O(1) system catalog lookup, unlike `os.path.getsize()` which had known accuracy issues with the SQLite version.

---

### Workflow 5: Shard Path Retrieval (`get_shard_paths`)

**Problem**: Given a batch of image paths, how do you find which JSONL shard files contain their processing results?

**Solution**: Batch query the `shard_path` column using `path = ANY(%s)` to retrieve output lineage for multiple paths in one database round-trip. **Returns results in the same order as input** to guarantee 1-to-1 correspondence.

**Usage**:
```python
paths = ['/img/01.jpg', '/img/02.jpg', '/img/03.jpg']
shard_paths = db.get_shard_paths(paths)
# Returns: ['/output/shard_0001.jsonl', None, '/output/shard_0003.jsonl']
#          ↑ matches paths[0]         ↑ paths[1]  ↑ paths[2]

for path, shard in zip(paths, shard_paths):
    if shard:
        print(f"{path} → {shard}")
    else:
        print(f"{path} → Not yet processed")
```

**Use cases**:
- **Result reconstruction** — Find all JSONL shards needed to reconstruct results for specific images
- **Debugging** — Trace which batch/shard a particular image was written to
- **Recovery validation** — Verify shard assignments match expected output distribution
- **Audit trail** — Generate lineage reports mapping images to their output locations

**NULL handling**: Paths with `shard_path = NULL` (not yet processed) return `None` at the corresponding index. Paths not found in the database also return `None`. **Guarantees**: `len(output) == len(input)` with positional correspondence.

---

### Workflow 6: Self-Healing Verification (`verify_and_fix_counts`)

**Problem**: What if counts drift due to crashes, manual edits, or bugs?

**Solution**: Periodically compare cached counts against actual `COUNT(*)`, auto-fix discrepancies atomically.

**Understanding Count Drift at Scale:**

Count drift is an **expected operational reality** at billion-scale, not a system failure. It emerges from the fundamental trade-off between consistency and throughput.

**Why Drift Happens:**
- **Concurrent state transitions** — Workers mark paths done simultaneously; cached counts may lag by microseconds
- **Crash timing windows** — System crashes between updating row state and updating cached counts
- **Manual operations** — Direct SQL queries or external tools bypass count update logic

**The Design Trade-Off:**

Perfect consistency requires distributed transactions (10-20% throughput penalty), synchronous replication (2-3x latency), and coordination overhead that scales poorly. Instead, we accept <0.01% drift and fix it periodically—the same trade-off Kafka, Cassandra, and DynamoDB make.

**📖 See:** [`drivers/db_driver_readme.md`](../drivers/db_driver_readme.md) for CLI commands and main [README.md](../README.md) section "Operational Realities & Limitations" for complete context.

---

## 🏗️ Architecture & Concurrency Model

### Multi-Process Safety Guarantees

The system supports **multiple concurrent processes** safely:

| Process Type | Operation | Lock Behavior | Concurrency Level |
|-------------|-----------|---------------|-------------------|
| **Ingestion** | `add_paths_from_source()` | Row-level locks during INSERT (milliseconds) | Multiple concurrent writers |
| **Workers** | `fetch_batch()` | Row-level `FOR UPDATE SKIP LOCKED` (microseconds) | Truly concurrent fetches (no waiting) |
| **Workers** | `mark_done()`/`mark_failed()` | Row-level locks per batch | Multiple concurrent writers |
| **Monitoring** | `get_db_health()` | Read-only (MVCC snapshot) | Unlimited concurrent readers |

**PostgreSQL MVCC concurrency model**:
```
Readers: ∞ concurrent (MVCC snapshots, never blocked by writers)
Writers: Concurrent at row level (no global write lock)

Timeline example:
  Worker A: fetch_batch() [W] ────────■ (locks rows 1-32, 2ms)
  Worker B: fetch_batch() [W] ────────■ (skips locked, grabs rows 33-64, 2ms — simultaneous!)
  Ingestion: add_paths() [W]  ──────────■ (inserts new rows, no conflict with fetches)
  Monitor: get_health() [R]   ────────────────────── (MVCC snapshot, always instant)
```

**Key improvement over SQLite**: Workers no longer serialize on `fetch_batch()`. `FOR UPDATE SKIP LOCKED` means Worker B doesn't wait for Worker A — it instantly grabs different rows. This removes the single biggest bottleneck in the SQLite architecture.

### Schema Design

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              SINGLE DATABASE ARCHITECTURE                            │
│                                                                                      │
│  1. MAIN DATA TABLE                                                                  │
│  ┌──────────────┬──────────────┬──────────────┬────┐                                 │
│  │ path (PK)    │ state (INT)  │ shard_path   │ IDX│                                 │
│  ├──────────────┼──────────────┼──────────────┼────┤                                 │
│  │ /img/01.jpg  │ 2 (Done)     │ /out/01.json │ Y  │                                 │
│  │ /img/02.jpg  │ 0 (Pending)  │ NULL         │ Y  │                                 │
│  │ /img/03.jpg  │ 4 (UpReady)  │ /out/01.json │ Y  │                                 │
│  └──────┬───────┴──────┬───────┴──────┬───────┴─┬──┘                                 │
│         │              │              │         │                                    │
│   [Unique Path]  [State Machine] [Lineage]  [Batch Fetch]                            │
│   [ON CONFLICT]                              [SKIP LOCKED]                           │
│                                                                                      │
│  ...........................................................................         │
│                                                                                      │
│  2. HEALTH METRICS (Application Managed)                                             │
│  ┌───────────────┬───────────────┐                                                   │
│  │ state (PK)    │ count (INT)   │                                                   │
│  ├───────────────┼───────────────┤                                                   │
│  │ 0 (Pending)   │ 500,000       │                                                   │
│  │ 1 (Progress)  │ 10,000        │                                                   │
│  │ 2 (Done)      │ 9,400,000     │                                                   │
│  │ 3 (Failed)    │ 90,000        │                                                   │
│  │ 4 (UpStream)  │ 2,000,000     │                                                   │
│  └───────┬───────┴───────┬───────┘                                                   │
│          │               │                                                           │
│    [5 Rows Total]  [O(1) Lookup]                                                     │
│                                                                                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  >> PURPOSE                                                                          │
│  State management, deduplication, lineage & tracking — all in one database.          │
│                                                                                      │
│  >> DEDUPLICATION                                                                    │
│  INSERT ... ON CONFLICT (path) DO NOTHING — atomic, no separate cache needed.        │
│                                                                                      │
│  >> WORKERS                                                                          │
│  fetch_batch(), mark_done(), mark_failed(), mark_previous_stage_done()               │
│                                                                                      │
│  >> OPTIMIZATION                                                                     │
│  Row-level MVCC, FOR UPDATE SKIP LOCKED, COPY bulk loading.                          │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Design Rationale**:

- `path TEXT PRIMARY KEY` → Automatic uniqueness via `ON CONFLICT DO NOTHING`, eliminates need for a separate seen cache
- `state INTEGER` with index → Fast `WHERE state = ?` filtering for `fetch_batch()`
- `state_counts` table → O(1) health monitoring, no expensive table scans
- `pg_database_size()` → Accurate database size reporting (unlike `os.path.getsize()` which had known issues with SQLite)

---

## 🛡️ Failure Modes & Mitigation

### Crash Recovery Protocol

For complete crash recovery procedures, see `drivers/db_driver_readme.md`. The recovery process involves stopping all processes, resetting stuck paths from in-progress to pending, verifying and fixing count drift, and resuming normal operations.

**Data loss guarantees**:
- Paths marked done/failed before crash: **Preserved** (committed transactions)
- Paths in-progress during crash: **Reset to pending** (will be retried)
- Paths being ingested during crash: **May be re-inserted** (idempotent via `ON CONFLICT DO NOTHING`)

---

## 📊 Module Structure & Integration

### File Organization

```
db/
 ├── db.py                      # DB class: unified entry point (SQLite + Postgres)
 ├── db_metrics_log.py          # DBIngestLogger: structured JSONL logging
 ├── db_verification.py         # verify_counts(), fix_count_drift(), reset_stuck/failed
 ├── db_readme.md               # This documentation
 ├── db_postgres/
 │   ├── db.py                  # PostgresBackend: Postgres-specific implementation
 │   └── db_queries.py          # DBQueries: all SQL operations
 └── db_sql/
     ├── db.py                  # SQLiteBackend: SQLite-specific implementation
     └── db_queries.py          # DBQueries (SQLite dialect)

drivers/
 ├── db_driver_postgres.py      # CLI tool for Postgres ingestion and maintenance
 ├── db_driver_sql.py           # CLI tool for SQLite ingestion and maintenance
 └── db_driver_readme.md        # CLI tool documentation
```

---

### Integration with Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   VISION INGEST SYSTEM INTEGRATION                                  │
├──────────────────────────────────────────────────┬──────────────────────────────────────────────────┤
│             RUNTIME PIPELINE (HOT)               │             SUPPORT MODULES (Shared)             │
├──────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│                                                  │                                                  │
│  1. ORCHESTRATOR (main.py→cli.py)                │  3. OUTPUT HANDLER (modules/writer.py)           │
│  ┌───────────────────────────────────────────┐   │  ┌───────────────────────────────────────────┐   │
│  │ >> fetch_batch()                          │   │  │ Class: JSONLWriter                        │   │
│  │    └─ Retrieves pending images            │   │  │                                           │   │
│  │                                           │   │  │ >> _flush_group()                         │   │
│  │ >> mark_failed()                          │   │  │    └─ Updates DB state (mark_done)        │   │
│  │    └─ Handles VLM inference failures      │   │  │       after writing to JSONL              │   │
│  └───────────────────────────────────────────┘   │  └───────────────────────────────────────────┘   │
│                                                  │                                                  │
│  2. CRASH RECOVERY (modules/recovery.py)         │  4. OBSERVABILITY (utils/main_logger.py)         │
│  ┌───────────────────────────────────────────┐   │  ┌───────────────────────────────────────────┐   │
│  │ >> validate_outputs()                     │   │  │ Class: MainLogger                         │   │
│  │    └─ Uses is_done() to check JSONL vs DB │   │  │                                           │   │
│  │                                           │   │  │ >> Receives DB errors from:               │   │
│  │ >> monitor_health()                       │   │  │    - fetch_batch()                        │   │
│  │    └─ Uses get_db_health() for stuck paths│   │  │    - mark_done() / mark_failed()          │   │
│  │                                           │   │  │    - get_shard_paths()                    │   │
│  │ >> get_shard_paths()                      │   │  │                                           │   │
│  │    └─ Batch lookup of output lineage      │   │  │                                           │   │
│  └───────────────────────────────────────────┘   │  └───────────────────────────────────────────┘   │
│                                                  │                                                  │
├──────────────────────────────────────────────────┴──────────────────────────────────────────────────┤
│                         MAINTENANCE & INGESTION (COLD / OFFLINE)                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  5. ADMIN UTILITY (drivers/db_driver.py)                                                            │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ >> MODE: STANDALONE (Does NOT run with Main Pipeline)                                         │  │
│  │                                                                                               │  │
│  │ >> Actions:                                                                                   │  │
│  │    1. Ingestion      -> Bulk insert new images (INSERT or COPY mode)                          │  │
│  │    2. Verification   -> Ensure count integrity                                                │  │
│  │    3. Reset          -> Manual crash recovery / state resets                                  │  │
│  └───────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**DB instance pattern**:
```python
# In drivers/cli.py (called from main.py)
pg_config = {"host": "10.0.0.1", "port": 5432, "dbname": "visiondb",
             "user": "pipeline", "password": "..."}
main_logger = MainLogger(...)

# Main DB instance for orchestrator operations
db = DB(pg_config=pg_config, main_logger=main_logger, fetch_state=0)

# Writer gets its own DB instance to avoid transaction conflicts
writer_db = DB(pg_config=pg_config, main_logger=writer_logger)
writer = JSONLWriter(..., db=writer_db)

# Recovery shares the main DB instance
recovery = RecoverySystem(db=db)
```

**Why separate DB instances?**
- **Transaction isolation:** While one operation like `fetch_batch()` is in a transaction, another module trying `mark_done()` on the same connection would conflict. Each concurrent DB operation gets its own connection.
- **Future connection pooling:** When migrating to `ConnectionPool`, each `DB` instance can draw from a shared pool instead — the separation pattern remains the same.

**Error propagation flow**:
```
DB operation fails (e.g., fetch_batch connection error)
  → Exception raised
  → DB class logs to its assigned MainLogger
  → ROLLBACK issued (if in transaction)
  → Exception re-raised to caller
  → Pipeline handles error (retry, alert, etc.)
```

---

## ✅ Design Synthesis — Why This Architecture Wins

### Strategic Trade-offs

* **Single Database with ON CONFLICT** (vs. *Dual Database with Seen Cache*)
    * ↳ **Rationale:** PostgreSQL's `ON CONFLICT DO NOTHING` provides atomic deduplication without a separate cache. Eliminates an entire database, its schema, triggers, and the 3-phase commit ordering required to keep two databases consistent.

* **Cached Counts** (vs. *`COUNT(*)` Queries*)
    * ↳ **Rationale:** Maintenance overhead is justified for O(1) speed. Direct counting is prohibitively slow (30s+) at 100M+ scale.

* **Idempotent Transitions** (vs. *Direct Updates*)
    * ↳ **Rationale:** Conditional SQL is justified for crash safety. Simpler updates would lead to double-counting statistics during retries.

* **Short-Lived Cursors** (vs. *Long-Lived Cursors*)
    * ↳ **Rationale:** Object creation overhead is justified to maximize concurrency. Long-lived cursors invite lock escalation and system-wide stalls.

* **FOR UPDATE SKIP LOCKED** (vs. *Serialized BEGIN IMMEDIATE*)
    * ↳ **Rationale:** Row-level locking allows truly concurrent batch fetches. SQLite's database-level write lock serialized all workers — the single biggest throughput bottleneck eliminated.

### What Changed from SQLite

| Aspect | SQLite | PostgreSQL |
|--------|--------|------------|
| Deduplication | Separate `seen_cache.db` with triggers | `ON CONFLICT DO NOTHING` in single table |
| Batch fetch locking | `BEGIN IMMEDIATE` (database-level) | `FOR UPDATE SKIP LOCKED` (row-level) |
| Concurrent writers | Serialized (1 writer at a time) | Concurrent (row-level MVCC) |
| Bulk loading | `executemany` only | `executemany` + `COPY FROM STDIN` |
| Filesystem concerns | WAL/PERSIST mode, `fcntl()` locks, mmap issues on network FS | None — Postgres manages its own storage |
| Init retry logic | 5× with exponential backoff + jitter | None needed — Postgres handles DDL races |
| Placeholder syntax | `?` with dynamic `IN (?,?,?)` construction | `%s` with native `ANY(%s)` array binding |
| DB size reporting | `os.path.getsize()` (had accuracy issues) | `pg_database_size()` (always accurate) |
| Connection config | File paths, `PRAGMA` statements, journal modes | `pg_config` dict (host, port, dbname, user, password) |

### What Makes It Elegant

This architecture achieves **operational simplicity through database-level capabilities**. PostgreSQL's MVCC, row-level locking, and `ON CONFLICT` eliminate the need for application-level workarounds (dual databases, seen caches, 3-phase commits, filesystem lock gymnastics). The system essentially runs itself: ingestion flows without choking workers, dashboards load instantly, crashes are recovered automatically, and multiple nodes share a single database without filesystem-level coordination.

---

## 📖 Related Documentation

- **CLI Tool**: `drivers/db_driver_readme.md` — Command-line interface for ingestion and maintenance
- **Logging**: `logger_readme.md` — Structured JSON logging format and error tracking
- **Pipeline**: `drivers/cli_readme.md` — Main processing pipeline integration
- **Recovery**: `modules/recovery_readme.md` — Crash recovery and checkpoint restoration

---

**This database layer provides the foundation for massively scalable, crash-resilient image processing pipelines. The migration from SQLite to PostgreSQL eliminates the dual-database complexity, enables true concurrent batch fetching via row-level locking, and removes all filesystem-level lock coordination — while preserving the same public API, state machine, and operational guarantees.**