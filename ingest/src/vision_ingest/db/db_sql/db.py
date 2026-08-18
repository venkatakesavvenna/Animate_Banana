# db_sql/db.py  —  SQLite-specific backend
#
# [given by claude(not verified)] ARCHITECTURE NOTES (FSx Lustre Compatibility):
# - Uses PERSIST journal mode (not WAL) — fcntl() locks work across network nodes
# - WAL mode (mmap-based shared memory) fails on network filesystems
# - PERSIST zeros journal header instead of create/delete, reducing network overhead

import os
import sqlite3
import time
import random
from typing import Generator, List, Optional, Tuple

from vision_ingest.db.db_metrics_log import DBIngestLogger
from vision_ingest.db.db_verification import verify_counts, fix_count_drift
from vision_ingest.db.db_sql.db_queries import DBQueries, SeenCacheQueries
from vision_ingest.utils.main_logger import MainLogger

# https://www.sqlite.org/whentouse.html

BASE_PATH = "/projects/data/vision-team/srihari_bandarupalli/PatramEDA/db"
DB_PATH = os.path.join(BASE_PATH, "images.db")
SEEN_CACHE_PATH = os.path.join(BASE_PATH, "seen_cache.db")

BATCH_SIZE = 32_000  # Keep under SQLite's default parameter limit (32766)


class SQLiteBackend:
    """
    SQLite-specific database backend for FSx Lustre multi-node pipelines.

    Handles: connection setup with retry/backoff, SQLite pragmas, seen-cache
    management, and SQLite-specific SQL dialect for is_done / get_shard_paths.

    Exposed to the unified DB class via self._b.
    All shared pipeline methods (fetch_batch, mark_done, etc.) live in DB.

    [given by claude(not verified)] Multi-Process Safety (FSx Lustre compatible):
    ================================================
    SWITCHED FROM WAL TO PERSIST JOURNAL MODE:
    - WAL mode requires mmap()-based shared memory (-wal, -shm files)
      which does NOT work on network filesystems (FSx Lustre uses LDLM)
    - PERSIST mode uses only fcntl() file locks (no shared memory files)
      and zeros the journal header instead of create/delete overhead
    
    Trade-off: Writers block readers (no concurrent read+write), but ensures
    correct locking behavior across distributed nodes.

    Retry Logic:
    - Exponential backoff with jitter prevents thundering herd when
      multiple processes start simultaneously on different nodes
    - Connection reconnection on retry recovers from broken pipes on network FS
    - Short-lived cursors per operation minimize lock contention

    State Machine:
        0 = Pending (ready to process)
        1 = In-Progress (locked by worker)
        2 = Done (successfully processed)
        3 = Failed (processing failed)
        4 = Upstream-ready (waiting for upstream completion)
    """

    # Class attribute — used by DB.fetch_batch, DB._execute_state_transition, etc.
    DBQueries = DBQueries

    def __init__(
        self,
        path: str = DB_PATH,
        seen_path: Optional[str] = None,
        main_logger: Optional[MainLogger] = None,
        fetch_state: int = 0,
        verify: bool = False,
    ) -> None:
        try:
            self.main_logger = main_logger
            self.fetch_state = fetch_state
            self.path = path
            self.seen_path = seen_path

            os.makedirs(os.path.dirname(path), exist_ok=True)

            self.conn = sqlite3.connect(path, timeout=120, check_same_thread=False)
            self.seen_conn: Optional[sqlite3.Connection] = None
            if seen_path is not None:
                self.seen_conn = sqlite3.connect(seen_path, timeout=120, check_same_thread=False)

            # [given by claude(not verified)] Set autocommit (isolation_level=None) BEFORE _init_databases.
            # This ensures init_schema's explicit BEGIN/COMMIT statements have full
            # transaction control. Default isolation_level="" auto-begins transactions
            # for DML, causing "cannot start a transaction within a transaction" errors.
            self.conn.isolation_level = None
            if self.seen_conn is not None:
                self.seen_conn.isolation_level = None

            self._init_databases()

            if verify:
                self.verify_and_fix_counts()

        except Exception as e:
            if self.main_logger:
                self.main_logger.log_error(
                    "[DB] SQLiteBackend init failed", e,
                    {"db_path": path, "seen_cache_path": seen_path},
                )
            else:
                print("SQLiteBackend init failed", e)
            raise

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_databases(self) -> None:
        """
        Initialize main DB and seen-cache with PERSIST journal mode pragmas + schema.

        [given by claude(not verified)] Retry Strategy (FSx Lustre optimized):
        - Increased from 3 to 5 attempts for network filesystem reliability
        - Exponential backoff with jitter (2^attempt + 0-1s random) prevents
          thundering herd when multiple nodes initialize simultaneously
        - Reconnects on retry (closes and reopens) to recover from broken pipes
          on network FS instead of reusing potentially corrupted connection objects
        - Rollback on failure cleans up stale transaction state

        Check-before-init optimization avoids unnecessary DDL contention when
        multiple processes start at the same time.
        """
        max_attempts = 5

        for attempt in range(max_attempts):
            try:
                DBQueries.init_connection_pragmas(self.conn)

                cur = self.conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images';")
                main_needs_init = cur.fetchone() is None
                cur.close()

                if main_needs_init:
                    DBQueries.init_database_pragmas(self.conn)
                    DBQueries.init_schema(self.conn)

                if self.seen_conn is not None:
                    DBQueries.init_connection_pragmas(self.seen_conn)

                    seen_cur = self.seen_conn.cursor()
                    seen_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='seen';")
                    seen_needs_init = seen_cur.fetchone() is None
                    seen_cur.close()

                    if seen_needs_init:
                        DBQueries.init_database_pragmas(self.seen_conn)
                        SeenCacheQueries.init_schema(self.seen_conn)

                break  # success

            except Exception as e:
                try:
                    self.conn.execute("ROLLBACK;")
                    if self.seen_conn is not None:
                        self.seen_conn.execute("ROLLBACK;")
                except Exception:
                    pass

                if attempt == max_attempts - 1:
                    raise

                # [given by claude(not verified)] Reconnect on retry to recover from broken pipes or stale connection state
                # on network filesystems. Reusing a bad connection will fail repeatedly.
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = sqlite3.connect(self.path, timeout=120, check_same_thread=False)
                self.conn.isolation_level = None

                if self.seen_conn is not None:
                    try:
                        self.seen_conn.close()
                    except Exception:
                        pass
                    self.seen_conn = sqlite3.connect(self.seen_path, timeout=120, check_same_thread=False)
                    self.seen_conn.isolation_level = None

                if self.main_logger:
                    self.main_logger.log_error("[DB] Init attempt failed", e)
                else:
                    print(f"[DB] Init attempt {attempt + 1} failed: {e}")

                # [given by claude(not verified)] Exponential backoff with jitter prevents thundering herd when multiple
                # nodes initialize simultaneously (2^attempt + random 0-1s jitter)
                backoff = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(backoff)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self.conn.close()
            if self.seen_conn is not None:
                self.seen_conn.close()
        except Exception:
            pass

    def _do_rollback(self, cur) -> None:
        cur.execute("ROLLBACK;")

    # ------------------------------------------------------------------
    # Path iterator (shared logic, duplicated here to avoid circular import)
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
        for path in iter_paths():
            batch.append(os.path.abspath(path))
            if len(batch) >= batch_size:
                yield (batch, time.time() - batch_start)
                batch = []
                batch_start = time.time()
        if batch:
            yield (batch, time.time() - batch_start)

    # ------------------------------------------------------------------
    # Ingestion (seen-cache path)
    # ------------------------------------------------------------------

    def add_paths_from_source(
        self,
        source: str,
        batch_size: int = BATCH_SIZE,
        logs_dir: Optional[str] = None,
    ) -> None:
        """
        Ingest images from source (folder or file of paths).
        Filters duplicates via the seen-cache DB before inserting into main DB.
        """
        if logs_dir is None:
            raise ValueError("logs_dir is required for add_paths_from_source")
        if self.seen_conn is None:
            raise ValueError(
                "seen_path is required for add_paths_from_source. "
                "Initialize SQLiteBackend with seen_path parameter."
            )

        logger = DBIngestLogger(logs_dir)
        batch_num = 0
        inserted_total = 0
        total_discovered = 0
        total_seen = 0
        start_time = time.time()

        try:
            for batch, walk_time in self._iter_paths_batched(source, batch_size):
                batch_num += 1
                total_discovered += len(batch)

                seen_cur = self.seen_conn.cursor()
                t0 = time.time()
                new_paths = SeenCacheQueries.filter_new_paths(seen_cur, batch)
                query_time = time.time() - t0
                seen_cur.close()

                already_seen_count = len(batch) - len(new_paths)
                total_seen += already_seen_count

                if not new_paths:
                    logger.log_all_duplicates(
                        batch_num=batch_num,
                        walk_time=walk_time,
                        images_found=len(batch),
                        query_time=query_time,
                    )
                    continue

                db_cur = self.conn.cursor()
                seen_cur = self.seen_conn.cursor()

                t0 = time.time()
                DBQueries.insert_paths_batch(db_cur, new_paths)
                main_db_insert_time = time.time() - t0

                t0 = time.time()
                SeenCacheQueries.insert_paths_batch(seen_cur, new_paths)
                seen_cache_insert_time = time.time() - t0

                db_cur.close()
                seen_cur.close()

                inserted_total += len(new_paths)
                db_health = self.get_db_health()
                logger.log_batch(
                    batch_num=batch_num,
                    walk_time=walk_time,
                    images_found=len(batch),
                    query_time=query_time,
                    already_seen=already_seen_count,
                    new_paths=len(new_paths),
                    main_db_insert_time=main_db_insert_time,
                    seen_cache_insert_time=seen_cache_insert_time,
                    total_inserted=inserted_total,
                    total_runtime=time.time() - start_time,
                    db_health=db_health,
                )

        except Exception as e:
            logger.log_error(
                batch_num=batch_num,
                operation="add_paths_from_source",
                error=e,
                context={
                    "source": source,
                    "batch_size": batch_size,
                    "total_discovered": total_discovered,
                    "total_inserted": inserted_total,
                    "total_seen": total_seen,
                },
            )
            print(f"Error in add_paths_from_source: {e}")
            raise

        total_runtime = time.time() - start_time
        logger.log_summary(total_runtime, inserted_total, total_discovered, total_seen)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_db_health(self) -> dict:
        cur = self.conn.cursor()
        state_rows = DBQueries.get_state_distribution(cur)
        total_main = sum(count for _, count in state_rows)
        cur.close()

        total_seen = None
        seen_cache_size_mb = None
        if self.seen_conn is not None:
            seen_cur = self.seen_conn.cursor()
            total_seen = SeenCacheQueries.get_total_count(seen_cur)
            seen_cur.close()
            seen_cache_size_mb = os.path.getsize(self.seen_path) / (1024 * 1024)

        main_db_size_mb = os.path.getsize(self.path) / (1024 * 1024)

        state_distribution = {}
        for state, count in state_rows:
            pct = (count / total_main * 100) if total_main else 0
            state_distribution[f"state_{state}"] = {
                "count": count,
                "pct": round(pct, 1),
            }

        result = {
            "main_db_size_mb": round(main_db_size_mb, 2),
            "total_images_main": total_main,
            "state_distribution": state_distribution,
        }
        if seen_cache_size_mb is not None:
            result["seen_cache_size_mb"] = round(seen_cache_size_mb, 2)
        if total_seen is not None:
            result["total_images_seen"] = total_seen
        return result

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_and_fix_counts(self) -> dict:
        if self.seen_conn is None:
            raise ValueError(
                "seen_path is required for verify_and_fix_counts. "
                "Initialize SQLiteBackend with seen_path parameter."
            )
        start = time.time()
        result = verify_counts(self.conn, self.seen_conn)
        print(f"Verification completed in {time.time() - start:.2f}s")

        if not result["valid"]:
            print(f"Count drift detected: {result['discrepancies']}")
            print("Fixing count drift...")
            fix_start = time.time()
            fix_result = fix_count_drift(self.conn, self.seen_conn)
            print(f"Fix completed in {time.time() - fix_start:.2f}s")
            if fix_result["valid"]:
                print("Count drift fixed successfully")
            else:
                print(f"Failed to fix count drift: {fix_result}")
            return fix_result
        else:
            print("Count verification passed — no drift detected")
            return result

    # ------------------------------------------------------------------
    # Single-row lookups (SQLite dialect: ? placeholders, IN(...))
    # ------------------------------------------------------------------

    def is_done(self, path: str) -> bool:
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT state FROM images WHERE path = ?;", (path,))
            row = cur.fetchone()
            return (row and row[0] == 2) is True
        except Exception as e:
            if self.main_logger:
                self.main_logger.log_error("[DB] is_done", e, {"path": path})
            raise
        finally:
            cur.close()

    def get_shard_paths(self, paths: list) -> List[Optional[str]]:
        paths = list(paths)
        if not paths:
            return []
        cur = self.conn.cursor()
        try:
            placeholders = ",".join("?" * len(paths))
            cur.execute(
                f"SELECT path, shard_path FROM images WHERE path IN ({placeholders});",
                paths,
            )
            path_to_shard = {row[0]: row[1] for row in cur.fetchall()}
            return [path_to_shard.get(p) for p in paths]
        except Exception as e:
            if self.main_logger:
                self.main_logger.log_error("[DB] get_shard_paths", e, {"paths_count": len(paths)})
            raise
        finally:
            cur.close()
