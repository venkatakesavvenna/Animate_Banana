# Database Management CLI Tool

**Author:** Srihari Bandarupalli  
**Module:** `drivers/db_driver_sql.py`

---

## Purpose

The `db_driver_sql.py` provides a command-line interface for database management operations including image ingestion, verification, and crash recovery for the PatramEDA database system.

---

## Usage

```bash
# Ingest images from a folder
python -m drivers.db_driver --mode ingest --folder /path/to/images

# Verify and fix database counts
python -m drivers.db_driver --mode verify

# Reset stuck in-progress paths (after crashes)
python -m drivers.db_driver --mode reset-stuck

# Reset failed paths (retry failed images)
python -m drivers.db_driver --mode reset-failed

# Run full maintenance (reset stuck + verify)
python -m drivers.db_driver --mode full-maintenance

# Run full maintenance including failed path reset
python -m drivers.db_driver --mode full-maintenance --reset-failed
```

---

## Operations

### 1. **Image Ingestion** (`--mode ingest`)

Recursively walks folder for `.jpg`, `.jpeg`, `.png` files, deduplicates against `seen_cache.db`, and inserts new paths into `images.db` with state=0 (pending).

**Arguments:**
- `--folder` (required): Source directory path
- `--batch-size` (optional): Images per batch (default: 32,000, max: 32,766)

**Metrics logged:**
- Discovery: walk time, files found
- Deduplication: query time, duplicate percentage
- Insert performance: DB/cache insert times, throughput
- Database health: state distribution, total images, DB sizes

**Logs:** `{LOGS_PATH}/{hostname}_ingest/{run_id}/db.log`

```bash
python -m drivers.db_driver --mode ingest --folder /data/images --batch-size 5000
```

---

### 2. **Database Verification** (`--mode verify`)

Validates cached counts (`state_counts`, `seen_counts`) against actual `COUNT(*)` queries. Automatically repairs detected drift.

**⚠️ Run with no concurrent DB access.**

**What it does:**
- Compares `state_counts` table against actual `COUNT(*)` from `images` table
- Compares `seen_counts` table against actual `COUNT(*)` from `seen` table
- Automatically repairs count drift if found
- Reports verification results (✅ pass, ⚠️ drift detected, 🔧 fixing, ✅ fixed)

**Understanding State Drift at Scale:**

At billion-image scale, minor state drift is **expected and acceptable**—it's a deliberate trade-off for throughput:

| Scale | Acceptable Drift | Example | Impact |
|-------|-----------------|---------|---------|
| 1M images | <10 paths | 0.001% | Negligible |
| 100M images | <1,000 paths | 0.001% | Acceptable |
| 1B images | <10,000 paths | 0.001% | Expected |

**The Trade-Off:** We prioritize **throughput over perfect consistency**. Achieving zero drift at all times requires:
- Distributed locks (10-20% throughput penalty)
- Synchronous replication (2-3x latency increase)
- Two-phase commits (complexity explosion)

Instead, we accept <0.01% drift and fix it periodically with maintenance operations.

**📖 See:** Main [README.md](../README.md) section "Operational Realities & Limitations" for complete context.
---

### 3. **Reset Stuck Paths** (`--mode reset-stuck`)

Resets state=1 (in-progress) → state=0 (pending). Essential after crashes where workers died mid-processing.

**⚠️ Run with no concurrent DB access.**

**Operations:**
- Queries all state=1 paths
- Atomically updates to state=0
- Updates `state_counts` table

**Run after:**
- System crashes/unexpected shutdowns (mandatory)
- Killing processing workers

**Expected stuck paths:** `batch_size × crashed_workers` (e.g., 32 × 4 = 128 paths = 0.00013% at 100M scale)

---

### 4. **Reset Failed Paths** (`--mode reset-failed`)

Resets state=3 (failed) → state=0 (pending). Allows retry of failed images.

**⚠️ Run with no concurrent DB access.**

**Operations:**
- Queries all state=3 paths
- Atomically updates to state=0
- Updates `state_counts` table

**Run after:**
- Fixing bugs that caused failures
- Model/pipeline upgrades
- Confirming failures were transient (GPU OOM, network timeouts)


**Best practice:** Test sample before bulk reset to avoid retry loops.

```bash
python -m drivers.db_driver --mode reset-failed
```

---

### 5. **Full Maintenance** (`--mode full-maintenance`)

Executes reset-stuck + verify in sequence. Optional: add `--reset-failed` flag to include failed path reset.

**⚠️ Run with no concurrent DB access.**

**Default behavior:**
```bash
python -m drivers.db_driver --mode full-maintenance
```
1. Reset state=1 → state=0
2. Verify and fix count drift

**With failed reset:**
```bash
python -m drivers.db_driver --mode full-maintenance --reset-failed
```
1. Reset state=1 → state=0
2. Reset state=3 → state=0
3. Verify and fix count drift

**Design:** Failed path reset is opt-in to prevent infinite retry loops from persistent failures.

---

## Safety Considerations

**Concurrent Access:**  
All maintenance operations require **exclusive access**. Stop all workers, ingestion processes, and DB readers/writers before running.

**Crash Recovery Workflow:**
```bash
# 1. Stop all processing workers
# 2. Run maintenance
python -m drivers.db_driver --mode full-maintenance

# 3. Optional: Reset failed paths if transient issues
python -m drivers.db_driver --mode reset-failed  # or use --reset-failed flag above

# 4. Resume processing
```

**Database Locking:**  
The maintenance operations acquire write locks on the database. During these operations:
- Read operations will block until maintenance completes
- Write operations will fail with timeout errors
- WAL mode allows reads during normal operation, but not during maintenance

---

## Configuration

Database paths are defined at the top of `db_driver.py`:

```python
BASE_PATH = "/projects/data/vision-team/srihari_bandarupalli/PatramEDA/db"
DB_PATH = os.path.join(BASE_PATH, "images.db")
SEEN_CACHE_PATH = os.path.join(BASE_PATH, "seen_cache.db")
LOGS_PATH = "/projects/data/vision-team/srihari_bandarupalli/PatramEDA/logs"
BATCH_SIZE = 32_000  # Keep under SQLite's parameter limit (32766)
```

Modify these paths if you need to work with different database locations.

---


---

## Integration with Main Pipeline

The `db_driver.py` tool is independent from the main processing pipeline (`main.py` → `drivers/cli.py`) and should be used for:
- Initial data ingestion
- Database maintenance
- Crash recovery
- Health monitoring

The main processing pipeline uses the `DB` class directly for:
- Fetching batches of pending images
- Marking images as done or failed
- Real-time state tracking

---

## Related Documentation

- **Database Architecture**: See `db/readme.md` for complete database design documentation
- **Main Logger**: See `logger_readme.md` for error tracking and structured logging
- **Processing Pipeline**: See `drivers/cli_readme.md` for the main processing workflow

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **Database is locked** | Concurrent process accessing DB | Stop all workers, wait for operations to complete, retry |
| **Count drift persists** | DB corruption or concurrent modifications | Stop all processes, run `full-maintenance`, check with `PRAGMA integrity_check` |
| **Ingestion extremely slow** | High deduplication ratio (>90%) | Check logs for duplicate %, verify folder contains new data |
| **Out of memory during ingestion** | Batch size too large | Reduce `--batch-size` (try 16K or 8K; max limit: 32,766) |

---
