"""
Unified structured JSONL logger for database ingestion metrics.

Writes to:
    <logs_dir>/ingest_batches.jsonl  — one line per batch / error
    (summary line appended to same file at end of run)

Used by both SQLiteBackend and PostgresBackend via the same public API.
query_time and seen_cache_insert_time are 0.0 when the backend has no seen cache.
"""
import json
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from vision_ingest.utils.utils import _clean


class DBIngestLogger:
    """Structured JSONL logger for database ingestion metrics."""

    def __init__(self, logs_dir: str) -> None:
        os.makedirs(logs_dir, exist_ok=True)
        self._batches_path = os.path.join(logs_dir, "ingest_batches.jsonl")
        self._write({
            "event": "ingestion_started",
            "timestamp": self._now(),
            "message": "DB INGESTION STARTED",
        })

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def _write(self, obj: dict) -> None:
        with open(self._batches_path, "a") as f:
            f.write(json.dumps(_clean(obj), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def log_all_duplicates(
        self,
        batch_num: int,
        walk_time: float,
        images_found: int,
        query_time: float,
    ) -> None:
        self._write({
            "event": "all_duplicates",
            "timestamp": self._now(),
            "batch_num": batch_num,
            "walk_time_sec": round(walk_time, 3),
            "images_found": images_found,
            "query_time_sec": round(query_time, 3),
        })

    def log_batch(
        self,
        batch_num: int,
        walk_time: float,
        images_found: int,
        query_time: float,
        already_seen: int,
        new_paths: int,
        main_db_insert_time: float,
        seen_cache_insert_time: float,
        total_inserted: int,
        total_runtime: float,
        db_health: dict,
    ) -> None:
        total_insert_time = main_db_insert_time + seen_cache_insert_time
        insert_rate = new_paths / total_insert_time if total_insert_time > 0 else 0.0
        seen_pct = (already_seen / images_found * 100) if images_found > 0 else 0.0
        avg_rate = total_inserted / total_runtime if total_runtime > 0 else 0.0

        self._write({
            "event": "batch_ingested",
            "timestamp": self._now(),
            "batch_num": batch_num,
            "discovery": {
                "walk_time_sec": round(walk_time, 3),
                "images_found": images_found,
            },
            "deduplication": {
                "query_time_sec": round(query_time, 3),
                "already_seen": already_seen,
                "already_seen_pct": round(seen_pct, 1),
                "new_paths": new_paths,
            },
            "insert_performance": {
                "main_db_insert_sec": round(main_db_insert_time, 3),
                "seen_cache_insert_sec": round(seen_cache_insert_time, 3),
                "total_insert_sec": round(total_insert_time, 3),
                "insert_rate_per_sec": int(insert_rate),
            },
            "cumulative": {
                "total_inserted": total_inserted,
                "total_runtime_sec": round(total_runtime, 1),
                "avg_rate_per_sec": int(avg_rate),
            },
            "db_health": db_health,
        })

    def log_summary(
        self,
        total_runtime: float,
        total_inserted: int,
        total_discovered: int,
        total_seen: int,
    ) -> None:
        avg_rate = total_inserted / total_runtime if total_runtime > 0 else 0.0
        seen_pct = (total_seen / total_discovered * 100) if total_discovered > 0 else 0.0
        new_pct = (total_inserted / total_discovered * 100) if total_discovered > 0 else 0.0

        self._write({
            "event": "ingestion_complete",
            "timestamp": self._now(),
            "total_runtime_sec": round(total_runtime, 1),
            "total_runtime_min": round(total_runtime / 60, 1),
            "total_discovered": total_discovered,
            "total_inserted": total_inserted,
            "total_inserted_pct": round(new_pct, 1),
            "total_seen_skipped": total_seen,
            "total_seen_skipped_pct": round(seen_pct, 1),
            "avg_insert_rate_per_sec": int(avg_rate),
        })

    def log_error(
        self,
        batch_num: int,
        operation: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "event": "ingestion_error",
            "timestamp": self._now(),
            "batch_num": batch_num,
            "operation": operation,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        if context:
            entry["context"] = context
        self._write(entry)
