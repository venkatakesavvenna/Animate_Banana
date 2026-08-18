# db/db.py  —  Unified DB class
"""
Single entry point for all database operations.

Selects the correct backend at construction time:
  - SQLiteBackend  when path_or_config is a str  (or path= kwarg is given)
  - PostgresBackend when path_or_config is a dict  (or pg_config= kwarg is given)

All call sites continue to use `from vision_ingest.db.db import DB` unchanged.
"""
import os
import time
from typing import Generator, Iterable, List, Optional, Tuple

from vision_ingest.utils.main_logger import MainLogger
from vision_ingest.db.db_verification import (
    reset_stuck_in_progress as _reset_stuck,
    reset_failed_paths as _reset_failed,
)


class DB:
    """
    Unified database handler for the image processing pipeline.

    Wraps either SQLiteBackend or PostgresBackend via self._b.
    All shared pipeline methods are implemented directly here.
    Backend-specific methods (init, close, add_paths_from_source, get_db_health,
    verify_and_fix_counts, is_done, get_shard_paths) are thin delegators to self._b.

    Supports all existing call patterns:
        DB(path_str, main_logger=..., fetch_state=...)         # SQLite, positional
        DB(config_dict, main_logger=..., fetch_state=...)      # Postgres, positional
        DB(path=..., seen_path=..., verify=False)              # SQLite, keyword
        DB(pg_config=..., fetch_state=...)                     # Postgres, keyword
    """

    def __init__(
        self,
        path_or_config=None,
        *,
        # SQLite-specific kwargs (backward-compat with db_driver_sql.py)
        path: Optional[str] = None,
        seen_path: Optional[str] = None,
        # Postgres-specific kwargs (backward-compat with db_driver_postgres.py)
        pg_config: Optional[dict] = None,
        # Common kwargs
        main_logger: Optional[MainLogger] = None,
        fetch_state: int = 0,
        verify: bool = False,
    ) -> None:
        if pg_config is not None or isinstance(path_or_config, dict):
            from vision_ingest.db.db_postgres.db import PostgresBackend
            self._b = PostgresBackend(
                pg_config or path_or_config,
                main_logger=main_logger,
                fetch_state=fetch_state,
                verify=verify,
            )
        else:
            from vision_ingest.db.db_sql.db import SQLiteBackend
            self._b = SQLiteBackend(
                path or path_or_config,
                seen_path=seen_path,
                main_logger=main_logger,
                fetch_state=fetch_state,
                verify=verify,
            )

    # ------------------------------------------------------------------
    # Properties — live references into the backend (never stale after reconnects)
    # ------------------------------------------------------------------

    @property
    def conn(self):
        return self._b.conn

    @property
    def main_logger(self) -> Optional[MainLogger]:
        return self._b.main_logger

    @property
    def fetch_state(self) -> int:
        return self._b.fetch_state

    @property
    def DBQueries(self):
        return self._b.DBQueries

    # ------------------------------------------------------------------
    # Backend-delegating methods
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._b.close()

    def add_paths_from_source(self, source: str, batch_size: int = 32_000,
                               logs_dir: Optional[str] = None, **kwargs) -> None:
        self._b.add_paths_from_source(source, batch_size, logs_dir, **kwargs)

    def get_db_health(self) -> dict:
        return self._b.get_db_health()

    def verify_and_fix_counts(self) -> dict:
        return self._b.verify_and_fix_counts()

    def is_done(self, path: str) -> bool:
        return self._b.is_done(path)

    def get_shard_paths(self, paths: Iterable[str]) -> List[Optional[str]]:
        return self._b.get_shard_paths(paths)

    def _do_rollback(self, cur) -> None:
        self._b._do_rollback(cur)

    # ------------------------------------------------------------------
    # Shared: path iterator
    # ------------------------------------------------------------------

    def _iter_paths_batched(
        self, source: str, batch_size: int
    ) -> Generator[Tuple[List[str], float], None, None]:
        """Yield (batch_paths, walk_time) from a folder or file of paths."""
        def iter_paths():
            if os.path.isfile(source):
                with open(source, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            yield line
            else:
                for root, _, files in os.walk(source):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png")):
                            yield os.path.join(root, f)

        batch: List[str] = []
        batch_start = time.time()
        for p in iter_paths():
            batch.append(os.path.abspath(p))
            if len(batch) >= batch_size:
                yield (batch, time.time() - batch_start)
                batch = []
                batch_start = time.time()
        if batch:
            yield (batch, time.time() - batch_start)

    # ------------------------------------------------------------------
    # Shared: internal state-transition helper
    # ------------------------------------------------------------------

    def _execute_state_transition(
        self,
        operation_name: str,
        paths: Iterable[str],
        from_state: int,
        to_state: int,
        shard_path: Optional[str] = None,
    ) -> int:
        paths = list(paths)
        if not paths:
            return 0

        cur = self.conn.cursor()
        try:
            rows_changed = self.DBQueries.mark_state_transition(
                cur, paths, from_state, to_state, shard_path
            )
            return rows_changed
        except Exception as e:
            self._do_rollback(cur)
            if self.main_logger:
                self.main_logger.log_error(
                    f"[DB] {operation_name}", e, {"paths_count": len(paths)}
                )
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # Shared: atomic fetch
    # ------------------------------------------------------------------

    def fetch_batch(self, batch_size: int) -> List[str]:
        """
        Atomically fetch and lock a batch of pending images.
        Transitions state from fetch_state (0 or 4) to 1 (in-progress).
        """
        cur = self.conn.cursor()
        try:
            paths = self.DBQueries.fetch_and_lock_batch(cur, batch_size, self.fetch_state)
            return paths
        except Exception as e:
            self._do_rollback(cur)
            if self.main_logger:
                self.main_logger.log_error("[DB] fetch_batch", e, {"batch_size": batch_size})
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # Shared: state transitions
    # ------------------------------------------------------------------

    def mark_done(self, paths: Iterable[str], shard_path: Optional[str] = None) -> None:
        """Mark paths as done (state 1 → 2). Optionally records shard_path."""
        self._execute_state_transition("mark_done", paths, from_state=1, to_state=2,
                                       shard_path=shard_path)

    def mark_failed(self, paths: Iterable[str]) -> None:
        """Mark paths as failed (state 1 → 3)."""
        self._execute_state_transition("mark_failed", paths, from_state=1, to_state=3)

    def mark_previous_stage_done(self, paths: Iterable[str]) -> int:
        """Mark paths as ready for next stage (state 0 → 4)."""
        return self._execute_state_transition(
            "mark_previous_stage_done", paths, from_state=0, to_state=4
        )

    def reset_paths_to_state(self, paths: Iterable[str], target_state: int) -> None:
        """
        Reset specific paths from state=1 back to target_state (0 or 4).
        Used during shutdown to re-queue paths that were fetched but not completed.
        """
        self._execute_state_transition(
            "reset_paths_to_state", paths, from_state=1, to_state=target_state
        )

    # ------------------------------------------------------------------
    # Shared: health monitoring
    # ------------------------------------------------------------------

    def get_state_count(self, state: int) -> int:
        """
        Exact number of rows in a state.

        Use this, not get_state_pct(), for any "is there work left?" decision.
        get_state_pct() rounds to one decimal place, so on a 10M-row database a
        state holding up to ~5000 rows still reports 0.0% — which made the
        pipeline's exit condition declare the dataset finished while thousands
        of images were still pending.
        """
        cur = self.conn.cursor()
        try:
            state_rows = self.DBQueries.get_state_distribution(cur)
        finally:
            cur.close()
        return next((count for s, count in state_rows if s == state), 0)

    def get_state_pct(self, state: int) -> float:
        """Percentage of rows in a state, for reporting only. Returns 0.0 if
        total is 0. Never branch on this — see get_state_count()."""
        cur = self.conn.cursor()
        try:
            state_rows = self.DBQueries.get_state_distribution(cur)
        finally:
            cur.close()

        total = sum(count for _, count in state_rows)
        if total == 0:
            return 0.0
        state_count = next((count for s, count in state_rows if s == state), 0)
        return round(state_count / total * 100, 1)

    # ------------------------------------------------------------------
    # Shared: reset via unified db_verification
    # ------------------------------------------------------------------

    def reset_stuck_in_progress(self) -> int:
        """Reset all state=1 (in-progress) paths back to state=0 (pending)."""
        return _reset_stuck(self.conn)

    def reset_failed_paths(self) -> int:
        """Reset all state=3 (failed) paths back to state=0 (pending)."""
        return _reset_failed(self.conn)
