# db_driver

PostgreSQL management driver for image ingestion pipelines. All Postgres operations (`initdb`, `pg_ctl start/stop`) are executed via `subprocess.run` — no daemon, no service manager required. The script manages the full cluster lifecycle from a single entry point.

---

## Modes

| Mode | What it does |
|---|---|
| `serve` | Init cluster (if needed), start server, run heartbeat loop |
| `stop` | Stop server (`pg_ctl stop -m fast`) |
| `serve-and-ingest` | Start server + ingest (heartbeat runs in daemon thread) |
| `ingest` | Bulk-insert image paths into an already-running server |
| `verify` | Detect and fix `state_counts` drift |
| `reset-stuck` | Reset state `1 → 0` (in-progress → pending) |
| `reset-failed` | Reset state `3 → 0` (failed → pending) |
| `full-maintenance` | `reset-stuck` + optional `reset-failed` + `verify` |

---

## Flags

### Connection flags (all modes except `stop`)

| Flag | Required | Default | Description |
|---|---|---|---|
| `--pg-host` | ✅ | — | Server IP or hostname |
| `--pg-port` | | `5432` | Server port |
| `--pg-dbname` | ✅ | — | Database name |
| `--pg-user` | ✅ | — | Postgres user |
| `--pg-password` | | `None` | Password (omit for trust auth) |

### Serve / serve-and-ingest flags

| Flag | Required | Description |
|---|---|---|
| `--data-dir` | ✅ | Path to the Postgres cluster data directory (`PGDATA`) |

### Ingest flags (`ingest`, `serve-and-ingest`)

| Flag | Required | Default | Description |
|---|---|---|---|
| `--source` | ✅ | — | Folder to walk, or a file of paths (one per line) |
| `--logs-path` | | `logs` | Where to write ingest logs |
| `--batch-size` | | `100000` | Rows per insert batch |
| `--use-copy` | | off | Use `COPY FROM STDIN` (fast, no dedup). Default is `INSERT ON CONFLICT DO NOTHING` |
| `--fetch-state` | | `0` | Initial state assigned to inserted rows |

### Maintenance flags

| Flag | Mode | Description |
|---|---|---|
| `--reset-failed` | `full-maintenance` | Also reset failed paths before verify |

---

## Usage

```bash
# Start server (initializes cluster on first run)
python -m vision_ingest.drivers.db_driver serve \
  --data-dir /data/pgdata \
  --pg-host 10.0.0.1 --pg-dbname visiondb --pg-user pipeline

# Stop server
python -m vision_ingest.drivers.db_driver stop --data-dir /data/pgdata

# Start server + ingest in one shot
python -m vision_ingest.drivers.db_driver serve-and-ingest \
  --data-dir /data/pgdata \
  --pg-host 10.0.0.1 --pg-dbname visiondb --pg-user pipeline \
  --source /path/to/images

# Ingest from any node into an already-running server
python -m vision_ingest.drivers.db_driver ingest \
  --pg-host 10.0.0.1 --pg-dbname visiondb --pg-user pipeline \
  --source /path/to/images

# Maintenance
python -m vision_ingest.drivers.db_driver full-maintenance \
  --pg-host 10.0.0.1 --pg-dbname visiondb --pg-user pipeline --reset-failed
```

---

## Quick DB Checks

Connect directly:

```bash
psql -h 127.0.0.1 -p 5432 -U pipeline -d visiondb
```

```sql
\l                          -- list databases
\c visiondb                 -- connect to db
\dt                         -- list tables
\d your_table_name          -- describe table

SELECT * FROM your_table_name LIMIT 10;

SELECT state, COUNT(*)
FROM your_table_name
GROUP BY state
ORDER BY COUNT(*) DESC;
```