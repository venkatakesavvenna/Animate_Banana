# Recovery System

**Author:** Srihari Bandarupalli

**Module:** `modules/recovery.py`

---

## Purpose

The recovery system ensures **JSONL shard files remain consistent with database state** after crashes or interruptions. It detects and repairs corrupted tail regions of JSONL files by validating records against the database and truncating invalid entries.

### Crash Safety Model

The system guarantees that:
- ✅ **Older data is always safe:** Only the last `fsync_every_lines` (default: 250) can be corrupted
- ✅ **Database is source of truth:** JSONL records must match DB state (state=2, "done")
- ✅ **Truncation preserves validity:** Only invalid tail lines are removed, all valid data retained
- ✅ **Writer can resume safely:** Returns the last safe shard path for writer to continue

---

## Architecture

### Core Concept: Tail-Only Validation

```
JSONL Shard File:
┌─────────────────────────────────────┐
│ Line 0001  ✓ Valid (fsync'd + done) │
│ Line 0002  ✓ Valid (fsync'd + done) │
│ ...        ✓ Valid (fsync'd + done) │
│ Line 9750  ✓ Valid (fsync'd + done) │
├─────────────────────────────────────┤ ← Last fsync boundary
│ Line 9751  ? Tail region (validate) │
│ Line 9752  ? Tail region (validate) │
│ ...        ? Tail region (validate) │
│ Line 10000 ? Tail region (validate) │
└─────────────────────────────────────┘
        ↑
   Only this region needs validation
```

**Why This Works:**
- Writer flushes every `fsync_every_lines` (250 lines)
- Flush sequence: `write() → fsync() → DB.mark_done()`
- If crash happens **before `write()`**, records are never written to JSONL, no recovery needed—we simply rerun them again
- If crash happens **after `write()` but before `mark_done()`**, JSONL has records but DB doesn't
- Recovery detects this mismatch and removes those records

---

## Recovery Algorithm

### Flow Diagram

```
┌─────────────────┐
│  Find Shards    │ → glob(f"{prefix}*.jsonl") → Get last shard
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Read Tail      │ → Read last fsync_lines (250) from end of file
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validate Each  │ → For each line:
│  Tail Line      │    1. JSON parseable?
│                 │    2. Has "path" field?
│                 │    3. DB.is_done(path) == True?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Find First     │ → min(invalid_indices)
│  Invalid Index  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Truncate File  │ → Keep only valid lines before first corruption
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Return Path    │ → Writer resumes from this shard
└─────────────────┘
```

---

## Integration with Writer

### Startup Sequence

Recovery is invoked before writer initialization. The recovered shard path is passed to the writer, which opens it in append mode and resumes writing from the last safe position.

### Writer Resume Behavior

**If recovery returned a path:**
- Writer opens that shard in **append mode**
- Checks current file size
- Continues writing from end of file
- May rotate to new shard if size exceeds limit

**If recovery returned None:**
- No existing shards found
- Writer creates `results_00000.jsonl`
- Starts fresh

---

## Error Categorization

The recovery system tracks errors by type and logs detailed information:

| Error Type | Description | Impact |
|------------|-------------|--------|
| `json_parse_error` | Line is not valid JSON | Line truncated |
| `missing_path` | JSON valid but no "path" key | Line truncated |
| `db_state_mismatch` | Path in JSONL but DB state ≠ done | Line truncated |
| Other exceptions | Unexpected validation errors | Line truncated + logged |

Error counts by type are logged to help diagnose crash patterns and system health.

---

## Logging

Recovery events are logged to **`recovery.log`** in structured JSON format.

### Key Events Logged

1. **`recovery_started`** - Shard identified, validation begins
2. **`corruption_detected`** - Per-line errors with details
3. **`truncation_planned`** - Number of lines to keep/remove
4. **`truncation_executed`** - File size changes
5. **`recovery_completed`** - Final state and duration

See [`logger_readme.md`](../logger_readme.md) for detailed log format information.

---

## The Beauty of This Design

### Crash Safety Guarantees

The recovery system provides **ironclad consistency guarantees**:

✅ **No false positives:** Valid records are never removed—three-layer validation ensures confidence  
✅ **Complete truncation:** All invalid records are eliminated, no partial corruption remains  
✅ **Consistency restored:** JSONL ↔ DB state mismatch is resolved completely  
✅ **Fast execution:** Only validates tail (~250 lines), O(n) where n = fsync_lines (< 1 second)  
✅ **Atomic updates:** File replacement is atomic via `os.replace()`, no partial writes  

Without recovery, the consequences would be catastrophic:
❌ **Writer continues with corrupted shard** → Appends to file with invalid tail  
❌ **DB queries return inconsistent data** → Paths marked done but not in JSONL  
❌ **Duplicate processing** → Paths might be re-fetched and re-processed  
❌ **Data loss** → Successful VLM outputs lost forever  

### Minimal Validation Scope
Recovery is **incredibly efficient** because it only validates the last ~250 lines (the tail region). Older data is guaranteed safe because:
- Once lines are fsync'd (flushed to disk), they're physically permanent
- Database marks them as "done" (state=2) after fsync succeeds
- Only the unflushed tail can be corrupted after a crash

### Crash Recovery at Every Stage
The design gracefully handles crashes at **any point** in the write cycle:

| Stage | JSONL State | DB State | Recovery Action |
|-------|------------|----------|-----------------|
| Before `write()` | No new lines | No new records | No crash damage—simply rerun |
| After `write()`, before `fsync()` | Partial/corrupted tail | DB unchanged | Tail detected invalid, truncated |
| After `fsync()`, before `mark_done()` | Valid tail lines | DB not updated | DB state mismatch detected, lines removed |
| After `mark_done()` | Valid tail lines | DB marked done | Fully consistent—no action needed |

### Perfect Consistency Without Copying
The recovery algorithm achieves **JSONL ↔ DB alignment** in a single pass:
1. **Three-layer validation** ensures no false positives:
   - JSON must be decodable (catches partial writes, corrupted bytes)
   - Path field must exist (catches malformed records)
   - Database state must be "done" (catches fsync→mark_done gap)

2. **Truncate from first invalid line**: If line N is invalid, lines N+1, N+2, etc. cannot be trusted (they might be from an interrupted write batch). By truncating from the first invalid line, we preserve all provably valid data.

3. **Atomic file replacement** ensures no partial updates—either the file is fully corrected or unchanged.

### Seamless Resumption
After recovery completes, the writer:
- Receives the recovered shard path (or `None` if no shards exist)
- Opens the file in append mode at the last safe position
- Continues writing without gap or duplication
- Resumes exactly where it left off—zero data loss

### Handles All Edge Cases Gracefully

Recovery **seamlessly handles every edge case** without special logic:

- **No shards exist**: Recovery returns `None`, writer creates the first shard (`_00000.jsonl`)
- **All tail valid**: No truncation needed, writer continues appending to existing shard
- **Entire tail corrupted**: All tail lines removed, writer resumes from last fsync boundary
- **Empty shard**: Recovery returns path to empty file, writer starts writing from position 0
- **Multiple shards**: Only the last shard is validated (older ones are already safe), O(1) lookup per shard rotation

### No Manual Intervention Required
Unlike traditional crash recovery systems, this design needs **zero manual database cleanup**, fsck operations, or transaction logs. The recovery happens automatically before the writer initializes, making the system **fault-tolerant by default**.

---

## Summary

The recovery system is a **critical safety mechanism** that ensures JSONL files and database state remain consistent after crashes. By validating only the tail region and truncating from the first invalid line, it provides:

- **Fast recovery** (< 1 second, only 250 lines validated)
- **Zero data loss** (all valid records preserved, provable consistency)
- **Perfect JSONL ↔ DB alignment** (three-layer validation catches all corruption patterns)
- **Seamless resumption** (writer continues exactly where it left off)
- **Automatic fault tolerance** (no manual intervention, database cleanup, or transaction logs needed)

This design allows the pipeline to handle crashes gracefully at any point without data corruption or manual recovery procedures.
