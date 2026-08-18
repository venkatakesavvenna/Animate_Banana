from __future__ import annotations

import os
import json
import glob
import threading
import queue
import time
import traceback
from typing import TYPE_CHECKING, Optional, Any
from vision_ingest.db.db import DB
from vision_ingest.utils.utils import get_logger

if TYPE_CHECKING:
    from stageweaver.router_coordinator.dag_pipeline import RouterCoordinator


class JSONLWriter:
    """Async non-blocking JSONL writer with shard rotation and DB syncing."""

    def __init__(
        self,
        db_path_or_config,
        router_coordinator: Optional[RouterCoordinator],
        stage_name: str,
        prefix="/projects/data/vision-team/srihari_bandarupalli/PatramEDA/jsonl_outputs/results_",
        digits=5,
        shard_size_mb=1024,
        fsync_every_lines=250,
        last_shard_path: Optional[str] = None,
        queue_maxsize: int = 0,
        logs_dir: Optional[str] = "./logs",
    ):
        # self.db = db # was causing concurrency issues when same db is shared by main cli and writer
        self.prefix = prefix
        self.digits = digits
        self.shard_bytes_limit = shard_size_mb * 1024 * 1024
        self.fsync_every_lines = fsync_every_lines
        self.logger = get_logger(logs_dir, "jsonl_writer")
        self.db:DB = DB(db_path_or_config, main_logger=self.logger)

        # Router-driven downstream signaling. The coordinator's mark_done()
        # handles routing decisions, downstream stage-DB activation, and
        # router_items bookkeeping. The legacy next_stage_db.mark_previous_stage_done
        # path is no longer used.
        self.router_coordinator = router_coordinator
        self.stage_name = stage_name

        # Tracking variables for logging
        self._cumulative_images = 0  # Total images written in this run

        # pick shard
        self.shard_idx, self.current_path = self._init_shard(last_shard_path)
        os.makedirs(os.path.dirname(self.current_path) or ".", exist_ok=True)
        self.f = open(self.current_path, "a", encoding="utf-8", buffering=1024 * 1024)
        self.f.seek(0, os.SEEK_END)
        self.current_size = self.f.tell()

        # buffer and thread. _pending_paths mirrors _paths as a set purely so
        # the duplicate check is O(1) — it was a list scan per enqueued item,
        # i.e. O(fsync_every_lines) on every image.
        self._objs, self._paths = [], []
        self._pending_paths = set()
        self._q = queue.Queue(maxsize=queue_maxsize)
        self._stop = object()
        self._flush_signal = object()
        self._closed_event = threading.Event()  # More efficient than lock for this use case
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---------- internal helpers ----------
    def _init_shard(self, last_shard_path):
        if last_shard_path:
            return self._extract_idx(last_shard_path), last_shard_path
        existing = sorted(glob.glob(f"{self.prefix}*.jsonl"))
        if not existing:
            return 0, self._shard_name(0)
        return self._extract_idx(existing[-1]), existing[-1]

    def _extract_idx(self, path):
        try:
            return int(path.replace(self.prefix, "").replace(".jsonl", ""))
        except Exception:
            return 0

    def _shard_name(self, i):
        return f"{self.prefix}{i:0{self.digits}d}.jsonl"

    def _rotate(self):
        # Capture previous shard info before rotation
        previous_shard_path = self.current_path
        previous_shard_size_bytes = self.current_size
        
        self.f.close()
        self.shard_idx += 1
        self.current_path = self._shard_name(self.shard_idx)
        self.f = open(self.current_path, "a", encoding="utf-8", buffering=1024 * 1024) # TODO:verify what is buffering and how it effect us.
        self.current_size = 0
        
        # Log rotation (calculations happen inside)
        self._log_rotation(previous_shard_path, previous_shard_size_bytes)

    # ---------- Core Functionality - internal thread runner ----------
    def _flush_group(self):
        if not self._objs:
            return

        # Start timing
        group_start = time.time()
        lines_written = len(self._objs)

        # Fetch the route each image took through the DAG up to (but not including)
        # this stage, then stamp it onto every output object before serialization.
        # Skipped in standalone mode (router_coordinator is None).
        if self.router_coordinator is not None:
            routes = self.router_coordinator.get_current_path_taken(self._paths)
            for obj in self._objs:
                obj["route_taken"] = routes.get(obj.get("path", ""), [])

        # Build payload and measure bytes
        payload = "\n".join(json.dumps(obj, ensure_ascii=False) for obj in self._objs) + "\n"
        payload_bytes = len(payload.encode('utf-8'))

        # TIME: Write operation
        write_start = time.time()
        self.f.write(payload)
        write_duration_ms = (time.time() - write_start) * 1000

        # TIME: Fsync operation
        fsync_start = time.time()
        self.f.flush()
        os.fsync(self.f.fileno())
        fsync_duration_ms = (time.time() - fsync_start) * 1000

        self.current_size = self.f.tell()

        # TIME: Router-driven downstream signaling.
        # mark_done() runs the stage's routing_fn (if any), writes fetch_state=4
        # into downstream stage DBs, then ORs this stage's bit into done_mask.
        # First call for any item also UPSERTs it into router_items.
        router_start = time.time()
        if self.router_coordinator is not None:
            self.router_coordinator.mark_done(
                self._paths, self.stage_name, successful_objs=self._objs
            )
        router_mark_done_duration_ms = (time.time() - router_start) * 1000

        # TIME: Mark done in DB
        db_start = time.time()
        self.db.mark_done(self._paths, self.current_path)
        db_mark_done_duration_ms = (time.time() - db_start) * 1000

        # Update cumulative counter
        self._cumulative_images += lines_written

        # Log flush metrics (calculations happen inside)
        self._log_flush_complete(lines_written, payload_bytes, write_duration_ms,
                                 fsync_duration_ms, db_mark_done_duration_ms, router_mark_done_duration_ms, group_start)

        # Clear buffers
        self._objs.clear()
        self._paths.clear()
        self._pending_paths.clear()

        # Check for rotation
        if self.current_size >= self.shard_bytes_limit:
            self._rotate()

    def _run(self):
        try:
            while True:
                item = self._q.get()
                if item is self._stop:
                    self._flush_group()
                    break
                if item is self._flush_signal:
                    self._flush_group()
                    self._q.task_done()
                    continue
                path = item.get("path", "")
                if path in self._pending_paths:
                    # Same path twice inside one flush group. mark_done() is
                    # keyed on path, so writing it twice would produce two JSONL
                    # records for one DB row. Dropping it is correct, but it
                    # should never happen (a path is fetched once) — so say so
                    # rather than discarding output in silence.
                    self._log_json("warning", {
                        "component": "writer",
                        "event": "duplicate_path_in_flush_group",
                        "path": path,
                    })
                else:
                    self._objs.append(item)
                    self._paths.append(path)
                    self._pending_paths.add(path)
                if len(self._objs) >= self.fsync_every_lines:
                    self._flush_group()

                self._q.task_done()
        except Exception as e:
            self._closed_event.set()  # Signal that writer is closed
            self._log_json("error", {
                "component": "writer",
                "event": "error in _run or _flush_group() of JSONLWriter",
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            })
            # Mark any remaining items in the queue as done to prevent deadlock
            try:
                self._q.task_done()
            except ValueError:
                pass  # task_done() called too many times or queue already empty

    # ---------- public API ----------
    def enqueue(self, obj: dict[str, Any]):
        """
        the objs here are successful_obj returned by `send_to_vlm()` function of prediction.py
        successful_objs.append({
            "path":         batch_paths[original_index],
            "prompt_str":   final_prompts[original_index]["prompt"],
            "vlm_output":   validated_output,
            "full_vlm_object": full_vlm_obj,
                    })
        """
        if self._closed_event.is_set():
            raise RuntimeError("Writer is closed")
        self._q.put(obj)

    def flush(self):
        """
        Force whatever's currently buffered to be written/fsynced/mark_done'd
        now, without waiting for fsync_every_lines or close(). A batch smaller
        than fsync_every_lines would otherwise sit unflushed indefinitely, and
        mark_done()'s downstream stage-DB activation only runs at flush — so a
        caller that's about to go idle (e.g. process_pipeline's empty-fetch
        wait) should call this instead of just sleeping.

        Async: enqueues a flush request on the writer thread's queue and
        returns immediately. No-op if nothing is buffered (_flush_group()
        checks that) or if the writer is already closed.
        """
        if self._closed_event.is_set():
            return
        self._q.put(self._flush_signal)

    def unflushed_paths(self):
        """
        Paths this writer accepted but never committed (buffered, or still on
        the queue) — i.e. rows sitting in DB state=1 that no fsync boundary will
        ever cover. Empty after a clean close(); non-empty only if the writer
        thread died or close() timed out mid-flush.

        cli()'s shutdown resets these back to the fetch state so they are
        replayed. That is safe precisely because the JSONL commit protocol is
        write -> fsync -> mark_done: anything not marked done was, by
        construction, not fsynced, so replaying it cannot duplicate a committed
        record.
        """
        paths = list(self._paths)
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is self._stop or item is self._flush_signal:
                continue
            path = item.get("path")
            if path:
                paths.append(path)
        return paths

    def stats(self):
        return {
            "queue_size": self._q.qsize(),
            "current_shard": self.current_path,
            "cumulative_images": self._cumulative_images,
            "current_shard_size_mb": round(self.current_size / (1024 * 1024), 2),
        }

    def close(self, timeout=60):
        if self._closed_event.is_set():
            return
        self._q.put(self._stop)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self._log_json("error", {"component": "writer", "event": "shutdown_timeout"})
        self._flush_group()
        try:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()
            self.db.close()
        finally:
            self._closed_event.set()


    # ---------- Logging helpers ----------
    def _log_json(self, level, data):
        """Log structured JSON data without indent."""
        log_func = getattr(self.logger, level)
        log_func(json.dumps(data, ensure_ascii=False))

    def _log_flush_complete(self, lines_written, payload_bytes, write_duration_ms, fsync_duration_ms,
                           db_mark_done_duration_ms, router_mark_done_duration_ms, group_start_time):
        """Log metrics after flushing a group of records."""
        # Calculate metrics here
        total_group_duration_ms = (time.time() - group_start_time) * 1000
        throughput_imgs_per_sec = (lines_written / total_group_duration_ms * 1000) if total_group_duration_ms > 0 else 0
        avg_bytes_per_line = payload_bytes // lines_written if lines_written > 0 else 0
        current_shard_size_mb = self.current_size / (1024 * 1024)

        log_data = {
            "component": "writer",
            "event": "fsync_complete",

            "timing": {
                "write_duration_ms": round(write_duration_ms, 2),
                "fsync_duration_ms": round(fsync_duration_ms, 2),
                "db_mark_done_duration_ms": round(db_mark_done_duration_ms, 2),
                "router_mark_done_duration_ms": round(router_mark_done_duration_ms, 2),
                "total_group_duration_ms": round(total_group_duration_ms, 2),
            },
            
            "progress": {
                "lines_written": lines_written,
                "cumulative_images": self._cumulative_images,
                "current_shard_index": self.shard_idx,
                "current_shard_size_mb": round(current_shard_size_mb, 2),
                "throughput_imgs_per_sec": round(throughput_imgs_per_sec, 2),
                "avg_bytes_per_line": avg_bytes_per_line
            },
            
            "shard": {
                "rotated": False,
                "path": os.path.basename(self.current_path)
            }
        }
        
        # Determine log level based on performance
        log_level = "info"
        if fsync_duration_ms > 1000:
            log_level = "warning"
            log_data["warning"] = "slow_fsync"
        elif write_duration_ms > 5000:
            log_level = "warning"
            log_data["warning"] = "slow_write"
        elif avg_bytes_per_line > 10240:
            log_level = "warning"
            log_data["warning"] = "large_line_size"
        
        self._log_json(log_level, log_data)

    def _log_rotation(self, previous_shard_path, previous_shard_size_bytes):
        """Log shard rotation event."""
        # Calculate metrics here
        previous_shard_size_mb = previous_shard_size_bytes / (1024 * 1024)
        
        log_data = {
            "component": "writer",
            "event": "shard_rotated",
            
            "shard": {
                "rotated": True,
                "previous_shard": os.path.basename(previous_shard_path),
                "previous_shard_size_mb": round(previous_shard_size_mb, 2),
                "new_shard": os.path.basename(self.current_path),
                "new_shard_index": self.shard_idx
            },
            
            "progress": {
                "cumulative_images": self._cumulative_images
            }
        }
        
        self._log_json("info", log_data)