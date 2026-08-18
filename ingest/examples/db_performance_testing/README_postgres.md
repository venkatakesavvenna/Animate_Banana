# DB Performance Testing

Tests ingest speed (COPY vs INSERT) and operation throughput (fetch_batch, mark_done, mark_failed) at 12M paths.

---

## Step 1 — Node A: Start Server

```bash
python -m vision_ingest.drivers.db_driver_postgres serve \
  --data-dir /opt/dlami/nvme/srihari.bandarupalli/db_test \
  --pg-host 0.0.0.0 \
  --pg-dbname visiondb \
  --pg-user pipeline
```

---

## Step 2 — Node B: Setup + Generate 12M paths + Ingest via COPY

```bash
cd examples/db_performance_testing
pip install -e /path/to/Vision-Ingestion-Engine
export PG_HOST=<NODE_A_IP>

# Generate 12M paths (~30–60s, ~700 MB output)
python gen_paths.py --n 12000000 --out image_paths.txt

# Ingest via COPY FROM STDIN (fast, no dedup)
python -m vision_ingest.drivers.db_driver_postgres ingest \
  --pg-host $PG_HOST --pg-dbname visiondb --pg-user pipeline \
  --source image_paths.txt --logs-path ./logs_postgres --batch-size 1000000 --use-copy
```

Elapsed time and paths/s printed at completion.

---

## Step 3 — Node B: Benchmark fetch_batch / mark_done / mark_failed

```bash
python bench_ops.py \
  --pg-host $PG_HOST --pg-dbname visiondb --pg-user pipeline \
  --batch-size 1000 --n-batches 200
```

Prints avg ms/batch and paths/s for each operation.
Adjust `--batch-size` and `--n-batches` to match your pipeline's actual batch sizes.

---

## Step 4 — Node A: Drop old DB, create fresh one

```bash
psql -U pipeline -d postgres \
  -c "DROP DATABASE visiondb; CREATE DATABASE visiondb;"
```

---

## Step 5 — Node A: Serve the fresh DB

The PostgreSQL server is still running — just restart the heartbeat monitor pointing at the fresh DB:

```bash
# Ctrl+C the old serve process, then re-run:
python -m vision_ingest.drivers.db_driver_postgres serve \
  --data-dir /opt/dlami/nvme/srihari.bandarupalli/db_test \
  --pg-host 0.0.0.0 \
  --pg-dbname visiondb \
  --pg-user pipeline
```

> Alternatively, skip the restart — the server is already running and `visiondb` is fresh. The `serve` command just manages startup and heartbeat; the DB itself is live.

---

## Step 6 — Node B: Ingest via INSERT ON CONFLICT (with dedup)

```bash
python -m vision_ingest.drivers.db_driver_postgres ingest \
  --pg-host $PG_HOST --pg-dbname visiondb --pg-user pipeline \
  --source image_paths.txt --logs-path ./logs_postgres --batch-size 1000000
```

Elapsed time and paths/s printed at completion. Compare against Step 2.

---

## Quick DB Checks

Connect to the database and run quick queries to inspect state:

```bash
psql -h 127.0.0.1 -p 5432 -U pipeline -d visiondb
```

### Common Commands

**List databases**
```sql
\l
```

**Connect to your DB** (if not already)
```sql
\c visiondb
```

**List tables**
```sql
\dt
```

**Describe a table**
```sql
\d your_table_name
```

**Peek at data**
```sql
SELECT * FROM your_table_name LIMIT 10;
```

**Check state counts** (if your table has a `state` column)
```sql
SELECT state, COUNT(*)
FROM your_table_name
GROUP BY state
ORDER BY COUNT(*) DESC;
```

**Check DB activity**
```sql
SELECT state, COUNT(*)
FROM pg_stat_activity
GROUP BY state;
```


