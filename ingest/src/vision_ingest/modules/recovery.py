# recovery.py
import os
import json
import glob
import time
import traceback
from vision_ingest.db.db import DB
from vision_ingest.utils.utils import get_logger

def read_last_k_lines(path, k=500):
    """Read last k lines from a file efficiently."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block = 4096
        data = b""
        while size > 0 and data.count(b"\n") <= k:
            delta = min(block, size)
            f.seek(-delta, os.SEEK_CUR)
            chunk = f.read(delta)
            data = chunk + data
            f.seek(-delta, os.SEEK_CUR)
            size -= delta
    return data.decode("utf-8", errors="ignore").splitlines()[-k:]


def log_json(logger, level, data):
    """Log structured JSON data."""
    log_func = getattr(logger, level)
    log_func(json.dumps(data, ensure_ascii=False))


def recover_last_shard(db: DB, prefix: str, fsync_lines=250, log_dir="./logs"):
    """
    Recover the last shard file by detecting and truncating corrupted/incomplete records.
    
    Args:
        db: Database instance to check record states
        prefix: Prefix for result files (e.g., "/path/to/PatramEDA/jsonl_outputs/results_")
        fsync_lines: Number of lines to check from end of file
        log_dir: Directory for recovery logs
        
    Returns:
        str: Path to the last shard file, or None if no shards found
    """
    fsync_lines=fsync_lines+10 # checking few more lines to ensure we dont miss any boundary errors
    logger = get_logger(log_dir, "recovery")
    start_time = time.time()
    
    # 1. Find last shard
    existing = sorted(glob.glob(f"{prefix}*.jsonl"))
    if not existing:
        log_json(logger, "info", {
            "event": "no_shards_found",
            "prefix": prefix
        })
        return None
    
    last = existing[-1]
    shard_filename = os.path.basename(last)
    
    # Extract shard index from filename using consistent logic with writer
    try:
        idx_str = last.replace(prefix, "").replace(".jsonl", "")
        shard_index = int(idx_str)
    except Exception as e:
        shard_index = None
        log_json(logger, "error", {
            "event": "shard_index_extraction_failed",
            "shard_file": shard_filename,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        })
        raise
    
    original_size = os.path.getsize(last)
    
    # 2. Log recovery start
    log_json(logger, "info", {
        "event": "recovery_started",
        "shard_file": shard_filename,
        "shard_index": shard_index,
        "original_size_bytes": original_size,
        "fsync_lines": fsync_lines,
        "tail_lines_checked": fsync_lines
    })
    
    # 3. Read and validate tail
    tail = read_last_k_lines(last, fsync_lines)
    invalid_idx = []
    error_counts = {}
    
    for idx, line in enumerate(tail):
        try:
            obj = json.loads(line)
            path = obj.get("path")
            
            if not path:
                invalid_idx.append(idx)
                error_counts["missing_path"] = error_counts.get("missing_path", 0) + 1
                log_json(logger, "error", {
                    "event": "corruption_detected",
                    "line_index": idx,
                    "error_type": "missing_path",
                    "raw_line": line[:200] if len(line) > 200 else line
                })
                continue
            
            if not db.is_done(path):
                invalid_idx.append(idx)
                error_counts["db_state_mismatch"] = error_counts.get("db_state_mismatch", 0) + 1
                log_json(logger, "error", {
                    "event": "corruption_detected",
                    "line_index": idx,
                    "error_type": "db_state_mismatch",
                    "path": path
                })
                
        except json.JSONDecodeError as e:
            invalid_idx.append(idx)
            error_counts["json_parse_error"] = error_counts.get("json_parse_error", 0) + 1
            log_json(logger, "error", {
                "event": "corruption_detected",
                "line_index": idx,
                "error_type": "json_parse_error",
                "error_message": str(e),
                "raw_line": line[:200] if len(line) > 200 else line
            })
            
        except Exception as e:
            invalid_idx.append(idx)
            error_type = type(e).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            log_json(logger, "error", {
                "event": "corruption_detected",
                "line_index": idx,
                "error_type": error_type,
                "error_message": str(e),
                "raw_line": line[:200] if len(line) > 200 else line
            })
    
    # 4. If no corruption found
    if not invalid_idx:
        duration_ms = (time.time() - start_time) * 1000
        log_json(logger, "info", {
            "event": "recovery_completed",
            "shard_file": shard_filename,
            "shard_index": shard_index,
            "duration_ms": round(duration_ms, 2),
            "corruption_found": False,
            "lines_checked": len(tail),
            "all_valid": True
        })
        return last
    
    # 5. Plan truncation
    cut_idx = min(invalid_idx)
    keep_tail = tail[:cut_idx]
    lines_to_remove = len(tail) - cut_idx
    
    log_json(logger, "warning", {
        "event": "truncation_planned",
        "first_invalid_index": cut_idx,
        "lines_to_keep": len(keep_tail),
        "lines_to_remove": lines_to_remove,
        "total_corruptions": len(invalid_idx),
        "error_counts": error_counts
    })
    
# 6. Execute truncation
    tmp = last + ".tmp"
    try:
        with open(last, "rb") as src, open(tmp, "wb") as dst:
            if cut_idx == 0:
                pass  # write nothing → full truncate
            else:
                tail_bytes = ("\n".join(tail)).encode()
                total_size = src.seek(0, os.SEEK_END)
                cut_point = max(total_size - len(tail_bytes), 0)
                src.seek(0)
                dst.write(src.read(cut_point))
                dst.write(b"\n".join(line.encode() for line in keep_tail))
                dst.write(b"\n")
        
        new_size = os.path.getsize(tmp)
        os.replace(tmp, last)
        
        log_json(logger, "info", {
            "event": "truncation_executed",
            "shard_file": shard_filename,
            "shard_index": shard_index,
            "original_size_bytes": original_size,
            "new_size_bytes": new_size,
            "bytes_removed": original_size - new_size,
            "lines_kept": len(keep_tail),
            "lines_removed": lines_to_remove
        })
        
    except Exception as e:
        log_json(logger, "error", {
            "event": "truncation_failed",
            "shard_file": shard_filename,
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        # Clean up temp file if it exists
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    
    # 7. Log completion
    duration_ms = (time.time() - start_time) * 1000
    final_size = os.path.getsize(last)
    
    log_json(logger, "info", {
        "event": "recovery_completed",
        "shard_file": shard_filename,
        "shard_index": shard_index,
        "duration_ms": round(duration_ms, 2),
        "corruption_found": True,
        "lines_truncated": lines_to_remove,
        "total_corruptions": len(invalid_idx),
        "error_counts": error_counts,
        "final_size_bytes": final_size,
        "final_line_count": len(keep_tail)
    })
    
    return last
