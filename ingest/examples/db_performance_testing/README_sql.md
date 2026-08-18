# SQLite DB Performance Testing

Tests ingest throughput (INSERT with dedup via seen-cache) and operation
performance (`fetch_batch`, `mark_done`, `mark_failed`) at scale using SQLite.

**No server required.** The `.db` files live inside
`examples/db_performance_testing/bench_sql_data/` — a folder on shared
network storage — so any node that mounts the same path can participate
without additional setup.

> **SQLite vs PostgreSQL key differences:**
> - No `COPY FROM STDIN` — SQLite uses `INSERT OR IGNORE` exclusively (with a seen-cache for dedup)
> - Default batch size is **32,000** (SQLite's `VARIABLE_NUMBER` limit is 32,766)
> - **PERSIST journal mode** — writers block readers for the duration of each transaction (no WAL-style concurrent read+write)
> - No separate server process — the `.db` file IS the server

---

## Step 1 — Any Node: Setup

```bash
cd examples/db_performance_testing
pip install -e /path/to/Vision-Ingestion-Engine
```

---

## Step 2 — Any Node: Generate 12M paths

```bash
# Reuse the existing generator — output goes to a shared path both nodes can see
python gen_paths.py --n 12000000 --out image_paths.txt
```

Takes ~30–60 s and produces a ~700 MB text file.

---

## Step 3 — Node A (or any single node): Ingest via INSERT OR IGNORE

SQLite does not support `COPY FROM STDIN`, so all inserts go through the
seen-cache deduplication path. Run ingest from **one node at a time**
(SQLite write lock is held per batch — concurrent ingestion from multiple
nodes will serialize and is not recommended).

```bash
python -m vision_ingest.drivers.db_driver_sql \
  --mode ingest \
  --db-path ./bench_sql_data \
  --folder image_paths.txt \
  --logs-path ./logs_sql \
  --batch-size 250000
```

The `bench_sql_data/` directory (with `images.db` + `seen_cache.db`) is
created automatically if it does not exist. Because it sits on shared
network storage, any other node can immediately open it read/write once
ingest finishes.

Elapsed time and paths/s are printed at completion.

---

## Step 4 — Any Node: Benchmark fetch_batch / mark_done / mark_failed

```bash
python bench_ops_sql.py \
  --db-dir ./bench_sql_data \
  --batch-size 1000 \
  --n-batches 200
```

Prints `avg ms/batch` and `paths/s` for each of the three operations,
followed by a final DB health snapshot.

Adjust `--batch-size` and `--n-batches` to match your pipeline's actual
working batch sizes. The SQLite driver caps inserts at 32,000 paths per
batch internally, but `fetch_batch` / `mark_done` / `mark_failed` work
at any size.

**Running from a second node** (read the same DB file over the network path):

```bash
# Node B — no extra setup needed, just point at the same directory
python bench_ops_sql.py \
  --db-dir /shared/path/to/examples/db_performance_testing/bench_sql_data \
  --batch-size 1000 --n-batches 200
```

---

## Step 5 — Reset and re-run (fresh benchmark)

SQLite does not have a `DROP DATABASE` command. Delete and recreate the
directory instead:

```bash
rm -rf ./bench_sql_data
mkdir  ./bench_sql_data
```

Then re-run Step 3 (ingest) followed by Step 4 (benchmark).

Alternatively, reset only the in-progress or failed paths without touching
the ingested data:

```bash
# Reset state-1 → state-0 (stuck in-progress paths)
python -m vision_ingest.drivers.db_driver_sql \
  --mode reset-stuck \
  --db-path ./bench_sql_data

# Reset state-3 → state-0 (failed paths)
python -m vision_ingest.drivers.db_driver_sql \
  --mode reset-failed \
  --db-path ./bench_sql_data
```

---

## Quick DB Inspection

```bash
sqlite3 bench_sql_data/images.db
```

### Useful queries

**Row count by state**
```sql
SELECT state, COUNT(*) FROM images GROUP BY state ORDER BY state;
```

**Fast O(1) count (from cached counters)**
```sql
SELECT state, count FROM state_counts ORDER BY state;
```

**Peek at data**
```sql
SELECT * FROM images LIMIT 10;
```

**DB file size on disk**
```bash
du -sh bench_sql_data/images.db bench_sql_data/seen_cache.db
```

---

## Notes on Multi-Node Access

- **PERSIST journal mode (not WAL)** — the driver was explicitly switched
  away from WAL because WAL relies on `mmap()`-based shared memory (`-shm`
  file) which does not work correctly on network filesystems (FSx Lustre,
  NFS). PERSIST uses only `fcntl()` byte-range locks, which Lustre's LDLM
  supports across nodes.
- **Writers block readers** — unlike WAL, PERSIST mode does not allow
  concurrent read+write. A write transaction (`BEGIN IMMEDIATE` in
  `fetch_batch`, `mark_done`, `mark_failed`) holds an exclusive lock for
  its duration. Any other node attempting a read or write during that window
  will wait up to the 60 s `busy_timeout` before failing. This is the
  deliberate trade-off: correct cross-node locking at the cost of
  read/write concurrency.
- **Writes from multiple nodes serialize** — concurrent `mark_done` /
  `mark_failed` calls from different nodes will queue (not corrupt). Expect
  higher latency under contention compared to the Postgres benchmark.
- For true concurrent multi-writer throughput testing, use the PostgreSQL
  benchmark (`bench_ops.py` + `README.md`) instead — Postgres MVCC has no
  such limitation.