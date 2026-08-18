# Logging Infrastructure

**Author:** Srihari Bandarupalli

---

## Purpose

PatramEDA uses **structured JSON logging** across all components to provide comprehensive observability, performance monitoring, and debugging capabilities. Each component maintains its own logger, writing to separate log files with consistent formatting and event-driven architecture.

---

## Logging Philosophy

### Hybrid Logging Approach

Logs use Python's standard logging framework with structured JSON payloads:

```
2024-11-13 14:23:45,123 - writer - INFO - {"event": "fsync_complete", "timing": {"write_duration_ms": 15.2, "fsync_duration_ms": 23.8}, "progress": {"lines_written": 250, "cumulative_images": 5000}}
```

Each log line consists of:
- **Timestamp:** `%(asctime)s` - Added by logging formatter (e.g., `2024-11-13 14:23:45,123`)
- **Component:** `%(name)s` - Logger name (e.g., `writer`, `db_ingest`, `main`)
- **Level:** `%(levelname)s` - Log level (e.g., `INFO`, `ERROR`)
- **Payload:** `%(message)s` - Either plain text (errors) or JSON string

**Benefits:**
- Standard logging format for easy filtering by timestamp, component, or level
- JSON portions can be parsed line-by-line by stripping the prefix
- Plain text errors intermixed with JSON for flexible error reporting
- Easy to parse with tools like `jq` and `grep`

### Event-Driven Architecture

When using JSON logging, each entry represents a discrete event with:
- **Event type:** Identifies what happened (e.g., `fsync_complete`, `corruption_detected`)
- **Metrics:** Quantitative measurements (timing, counts, sizes)
- **Metadata:** Additional debugging context

---

## Log Files Overview

### Directory Structure

```
logs/
└── <node_name>/          # Hostname of the machine
    └── <run_id>/         # Timestamp: YYYYMMDD-HHMMSS
        ├── recovery.log
        ├── jsonl_writer.log
        ├── db.log
        ├── main.log
        └── vlm_prediction.log
```

**Run Isolation:** Each pipeline run gets its own timestamped directory.

---

## 1. recovery.log

**Component:** Recovery System (`modules/recovery.py`)

**Purpose:** Tracks crash recovery operations performed at program startup, validating JSONL shard tails for corruption.

**Frequency:** Only runs at startup; logs recovery events once per program execution.

### What Gets Logged

**Startup:** When recovery begins, logs the shard being validated, file size, and number of tail lines to check (default: 250).

**Corruption Detection:** For each invalid line found, logs the line index, error type (JSON parse error, missing path, DB state mismatch), and the raw line content (truncated if long).

**Truncation Planning:** After validation completes, logs first invalid index, number of lines to keep vs remove, total corruption count, and error type distribution.

**Truncation Execution:** Logs original file size, new file size after truncation, and bytes removed.

**Completion:** Logs total recovery duration, whether corruption was found, and final file metrics.

### Key Events

- `recovery_started` - Shard identification and validation initiation
- `corruption_detected` - Per-line validation failures
- `truncation_planned` - Summary of what will be removed
- `truncation_executed` - File modification details
- `recovery_completed` - Final state and duration

### Typical Entry

**Raw Log Line:**
```
2024-11-13 14:23:45,123 - recovery - INFO - {"event": "recovery_completed", "shard_file": "results_00003.jsonl", "shard_index": 3, "duration_ms": 287.45, "corruption_found": true, "lines_truncated": 23, "total_corruptions": 23, "error_counts": {"db_state_mismatch": 23}}
```

**Extracted JSON (after stripping prefix):**
```json
{
  "event": "recovery_completed",
  "shard_file": "results_00003.jsonl",
  "shard_index": 3,
  "duration_ms": 287.45,
  "corruption_found": true,
  "lines_truncated": 23,
  "total_corruptions": 23,
  "error_counts": {
    "db_state_mismatch": 23
  }
}
```

---

## 2. jsonl_writer.log

**Component:** JSONL Writer (`modules/writer.py`)

**Purpose:** Monitors asynchronous JSONL writing, fsync operations, shard rotation, and write performance.

**Frequency:** Logs on every fsync operation (every `fsync_every_lines` interval, default 250).

### What Gets Logged

**Fsync Operations:** Every time the writer flushes a buffer, logs the write duration, fsync duration, DB mark_done duration, and total group duration. Also logs throughput (images/sec), cumulative images written, average bytes per line, current shard size, and shard index.

**Performance Warnings:** Automatically flags slow operations with warning level: slow fsync (>1s), slow write (>5s), or large line sizes (>10KB).

**Shard Rotation:** When a shard reaches 1GB and rotates to a new file, logs the previous shard name and size, new shard name and index, and cumulative images processed.

**Shutdown:** Logs if background thread fails to exit within timeout period.

### Key Events

- `fsync_complete` - Write/fsync/DB timing and throughput metrics
- `shard_rotated` - Shard file rotation details

### Typical Entry

**Raw Log Line:**
```
2024-11-13 14:23:45,123 - writer - INFO - {"event": "fsync_complete", "timing": {"write_duration_ms": 12.34, "fsync_duration_ms": 45.67, "total_group_duration_ms": 78.9, "db_mark_done_duration_ms": 20.88}, "progress": {"lines_written": 250, "cumulative_images": 25000, "current_shard_index": 0, "current_shard_size_mb": 245.32, "throughput_imgs_per_sec": 3164.56, "avg_bytes_per_line": 10234}}
```

**Extracted JSON (after stripping prefix):**
```json
{
  "event": "fsync_complete",
  "timing": {
    "write_duration_ms": 12.34,
    "fsync_duration_ms": 45.67,
    "total_group_duration_ms": 78.9,
    "db_mark_done_duration_ms": 20.88
  },
  "progress": {
    "lines_written": 250,
    "cumulative_images": 25000,
    "current_shard_index": 0,
    "current_shard_size_mb": 245.32,
    "throughput_imgs_per_sec": 3164.56,
    "avg_bytes_per_line": 10234
  }
}
```

---

## 3. db.log

**Component:** Database Driver (`drivers/db_driver.py`)

**Purpose:** Tracks folder ingestion pipeline with discovery, deduplication, insert performance, and database health monitoring.

**Frequency:** Logs per batch (every BATCH_SIZE ~32k images).

### What Gets Logged

**Discovery:** For each batch of paths discovered via `os.walk()`, logs the walk time (how long it took to traverse the filesystem) and images found in that batch.

**Deduplication:** Logs query time to check seen_cache.db, count and percentage of already-seen paths, and count of new paths identified for insertion.

**Insert Performance:** Logs main DB insert time, seen cache insert time, total insert time across both databases, and insertion rate (images/sec).

**Cumulative Stats:** Running totals of images inserted, total runtime, and average insertion rate since ingestion started.

**Database Health:** Periodically logs database file sizes (MB) for both images.db and seen_cache.db, state distribution counts and percentages for all 4 states (pending, in-progress, done, failed), and total seen cache count.

**Edge Cases:** Logs when an entire batch consists of duplicates (no new inserts needed).

**Final Summary:** At ingestion completion, logs total images discovered, total inserted, total skipped (duplicates), and overall throughput.

### Key Events

- `ingestion_started` - Folder ingestion begins
- `batch_processed` - Per-batch metrics (discovery, dedup, insert)
- `ingestion_complete` - Final summary and totals

### Typical Entry

**Raw Log Line:**
```
2024-11-13 14:23:45,123 - db_driver - INFO - {"component": "db_ingest", "event": "batch_processed", "batch_number": 5, "discovery": {"walk_time_sec": 2.34, "images_found": 32000}, "deduplication": {"query_time_sec": 0.87, "already_seen_count": 1500, "already_seen_pct": 4.69, "new_paths_count": 30500}, "insert_performance": {"main_db_insert_time_sec": 1.23, "seen_cache_insert_time_sec": 0.45, "total_insert_time_sec": 1.68, "insert_rate_per_sec": 18154.76}, "db_health": {"main_db_size_mb": 1234.56, "seen_cache_size_mb": 567.89, "state_0_pending": 125000, "state_2_done": 35000}}
```

**Extracted JSON (after stripping prefix):**
```json
{
  "component": "db_ingest",
  "event": "batch_processed",
  "batch_number": 5,
  "discovery": {
    "walk_time_sec": 2.34,
    "images_found": 32000
  },
  "deduplication": {
    "query_time_sec": 0.87,
    "already_seen_count": 1500,
    "already_seen_pct": 4.69,
    "new_paths_count": 30500
  },
  "insert_performance": {
    "main_db_insert_time_sec": 1.23,
    "seen_cache_insert_time_sec": 0.45,
    "total_insert_time_sec": 1.68,
    "insert_rate_per_sec": 18154.76
  },
  "db_health": {
    "main_db_size_mb": 1234.56,
    "seen_cache_size_mb": 567.89,
    "state_0_pending": 125000,
    "state_2_done": 35000
  }
  // Note: db_health data is passed to main_logger from components
  // main_logger does NOT have its own DB connection
}
```

---

## 4. main.log

**Component:** Main Pipeline (`main.py` → `drivers/cli.py`, `utils/main_logger.py`)

**Purpose:** End-to-end pipeline orchestration monitoring including fetch, VLM inference, writing, and system health.

**Frequency:** Logs at processing checkpoints (every `fsync_every_lines` interval, typically 250 images).

### What Gets Logged

**Startup:** Logs full configuration (batch sizes, paths, threads), initial DB state (counts for all states, total images), and recovery information if crash recovery was performed.

**Processing Checkpoints:** Every 250 images (aligned with writer fsync), logs progress metrics (images processed in window, cumulative count, window rate, overall rate, ETA), fetch metrics (total/avg/max fetch time, empty fetch count, wait time), VLM metrics (inference time total/avg per batch/avg per image/max, throughput, success/failed counts and rates, cumulative failed), pipeline timing (end-to-end batch time, bottleneck identification), DB state (pending, in-progress, complete, failed counts and percentages, completion %, ETA hours), and writer state (current shard index, size, cumulative written).

**Batch Operations:** Logs first batch (bootstrap phase - prompt preparation only) and last batch (flush phase - VLM inference only) with special annotations.

**Shutdown:** Logs shutdown reason (keyboard interrupt, complete, or error), total runtime, total images processed, total failed, final DB state distribution, and final writer shard info.

**Errors:** Any errors from DB operations (fetch_batch, mark_done, mark_failed) are logged with full traceback.

### Key Events

- `pipeline_started` - Configuration, initial state, recovery info
- `processing_checkpoint` - Comprehensive metrics every 250 images
- `batch_processing` - First/last batch special logging
- `pipeline_stopped` - Final summary and reason

### Typical Entry

**Raw Log Line:**
```
2024-11-13 14:30:00,456 - main - INFO - {"component": "main", "event": "processing_checkpoint", "timestamp": "2024-11-13T14:30:00", "progress": {"images_processed_window": 250, "cumulative_images": 5000, "window_rate_imgs_per_sec": 125.3, "overall_rate_imgs_per_sec": 102.4}, "vlm_metrics": {"inference_time_total_ms": 3456.78, "inference_time_avg_per_batch_ms": 432.1, "throughput_imgs_per_sec": 115.2, "success_count": 245, "failed_count": 5, "success_rate_pct": 98.0}, "db_state": {"state_0_pending": 95000, "state_1_processing": 32, "state_2_complete": 5000, "state_3_failed": 25, "completion_pct": 5.0, "eta_hours": 12.3}, "writer_state": {"current_shard": "results_00000.jsonl", "current_shard_size_mb": 49.2}}
```

**Extracted JSON (after stripping prefix):**
```json
{
  "component": "main",
  "event": "processing_checkpoint",
  "timestamp": "2024-11-13T14:30:00",
  "progress": {
    "images_processed_window": 250,
    "cumulative_images": 5000,
    "window_rate_imgs_per_sec": 125.3,
    "overall_rate_imgs_per_sec": 102.4
  },
  "vlm_metrics": {
    "inference_time_total_ms": 3456.78,
    "inference_time_avg_per_batch_ms": 432.1,
    "throughput_imgs_per_sec": 115.2,
    "success_count": 245,
    "failed_count": 5,
    "success_rate_pct": 98.0
  },
  "db_state": {
    "state_0_pending": 95000,
    "state_1_processing": 32,
    "state_2_complete": 5000,
    "state_3_failed": 25,
    "completion_pct": 5.0,
    "eta_hours": 12.3
  },
  "writer_state": {
    "current_shard": "results_00000.jsonl",
    "current_shard_size_mb": 49.2
  }
}
```

---

## 5. vlm_prediction.log

**Component:** VLM Prediction (`modules/prediction.py`)

**Purpose:** Tracks errors during VLM initialization, image loading, prompt generation, and inference operations.

**Frequency:** Logs errors in every batch (not structured JSON - plain text error messages).

### What Gets Logged

**Initialization Errors:** Logs failures in VLLMConfig initialization, VLLMService initialization, or prompt file reading with error details.

**Image Loading Failures:** Logs errors when images fail to load during prompt generation (file not found, corrupted, unsupported format) with the path and error type.

**Metadata Extraction Failures:** Logs failures to extract metadata or captions from images.

**Prompt Processing Failures:** Logs when prompt generation fails for specific images (template filling errors, image processing errors).

**VLM Inference Errors:** Logs vLLM service errors, out-of-memory errors, or other inference failures.

**Validation Failures:** Logs when VLM outputs fail Pydantic validation after multiple retry attempts.

### Key Error Messages

- `"Failed to initialize VLLMConfig: ..."`
- `"Failed to initialize VLLMService: ..."`
- `"Failed to read prompt file from ...: ..."`
- `"Error in get_prompts: ..."`
- `"Error in send_to_vlm: ..."`
- `"VLM Validation failed after multiple retries for path: ..."`

### Typical Entry

```
2024-11-13 14:30:15,789 - vlm_prediction - ERROR - Failed to initialize VLLMConfig: CUDA out of memory
```

```
2024-11-13 14:30:20,456 - vlm_prediction - ERROR - Error in get_prompts: PIL.UnidentifiedImageError: cannot identify image file /path/to/image.jpg
```

---

## Checkpoint Alignment

Different components log at different intervals:

```
Time →

[Program Start]
→ recovery.log (logs once at startup)

[Processing Every 250 images]
→ jsonl_writer.log (fsync_every_lines)
→ main.log (processing_checkpoint)

[Processing Every ~32k images]
→ db.log (batch_processed)

[Any Error in Batch]
→ vlm_prediction.log (error messages only)

[Program End]
→ main.log (pipeline_stopped)
→ recovery.log (startup on next run)
```

**Note:** `vlm_prediction.log` only contains errors, not JSON events. Verify and fix operations print to terminal only.

---

## Log Analysis Examples

### Extract JSON from Logs

To parse JSON from logs, strip the prefix (timestamp - component - level -):

```bash
cat logs/*/*/jsonl_writer.log | sed 's/^[^-]*-[^-]*-[^-]*-//' | jq '.'
```

### Find Slow Fsyncs

```bash
cat logs/*/*/jsonl_writer.log | \
  sed 's/^[^-]*-[^-]*-[^-]*-//' | \
  jq 'select(.event == "fsync_complete" and .timing.fsync_duration_ms > 1000)'
```

### Calculate Average Throughput

```bash
cat logs/*/*/main.log | \
  sed 's/^[^-]*-[^-]*-[^-]*-//' | \
  jq 'select(.event == "processing_checkpoint") | .progress.window_rate_imgs_per_sec' | \
  awk '{sum+=$1; count++} END {print sum/count}'
```

### Find All VLM Errors

```bash
cat logs/*/*/vlm_prediction.log | grep "ERROR"
```

### Check Recovery Impact

```bash
cat logs/*/*/recovery.log | \
  sed 's/^[^-]*-[^-]*-[^-]*-//' | \
  jq 'select(.event == "recovery_completed") | {shard: .shard_file, truncated: .lines_truncated, duration_ms: .duration_ms}'
```

---

## Summary

PatramEDA's logging infrastructure provides:

- **Structured observability:** JSON format for machine-parseability and easy analysis
- **Component isolation:** Separate logs per module for focused debugging
- **Performance tracking:** Detailed timing and throughput metrics at every level
- **Hybrid logging:** Mix of JSON events and plain text error messages
- **Checkpoint alignment:** Coordinated logging intervals for cross-component correlation

All logs use Python's standard logging formatter and JSON payloads, enabling easy parsing with standard text processing tools like `jq`, `grep`, and `sed`.
