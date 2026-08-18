# db_postgres/db.py  —  PostgreSQL-specific backend
import os
import time
from typing import Generator, List, Optional, Tuple

import psycopg

from vision_ingest.db.db_metrics_log import DBIngestLogger
from vision_ingest.db.db_verification import verify_counts, fix_count_drift
from vision_ingest.db.db_postgres.db_queries import DBQueries
from vision_ingest.utils.main_logger import MainLogger

# For future scaling: replace self.conn with psycopg.pool.ConnectionPool (sync)
# or AsyncConnectionPool for async workloads. Both are in psycopg_pool package.


class PostgresBackend:
    """
    PostgreSQL-specific database backend.

    Handles: single psycopg3 connection, schema init (no retry — Postgres handles
    concurrent DDL safely), ON CONFLICT deduplication, and Postgres-specific SQL
    dialect for is_done / get_shard_paths.

    Exposed to the unified DB class via self._b.
    All shared pipeline methods (fetch_batch, mark_done, etc.) live in DB.

    No seen-cache — deduplication is atomic via INSERT ... ON CONFLICT DO NOTHING.
    No PRAGMA statements or isolation_level hacks — those are SQLite-specific.

    State Machine:
        0 = Pending   1 = In-Progress   2 = Done   3 = Failed   4 = Upstream-ready
    """

    # Class attribute — used by DB.fetch_batch, DB._execute_state_transition, etc.
    DBQueries = DBQueries

    def __init__(
        self,
        pg_config: dict,
        main_logger: Optional[MainLogger] = None,
        fetch_state: int = 0,
        verify: bool = False,
    ) -> None:
        """
        pg_config required keys: host, port, dbname, user, password
        Example: {"host": "10.0.0.1", "port": 5432, "dbname": "visiondb",
                  "user": "pipeline", "password": "..."}
        """
        self.pg_config = pg_config
        self.main_logger = main_logger
        self.fetch_state = fetch_state
        self.conn = psycopg.connect(**pg_config)
        self._init_database()
        if verify:
            self.verify_and_fix_counts()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_database(self) -> None:
        try:
            DBQueries.init_schema(self.conn)
        except Exception as e:
            if self.main_logger:
                self.main_logger.log_error("[DB] Schema initialization failed", e)
            raise

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _do_rollback(self, cur) -> None:
        self.conn.rollback()

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
    # Ingestion (ON CONFLICT path — no seen cache)
    # ------------------------------------------------------------------

    def add_paths_from_source(
        self,
        source: str,
        batch_size: int = 32_000,
        logs_dir: Optional[str] = None,
        use_copy: bool = False,
    ) -> None:
        """
        Ingest images from source (folder or file of paths).
        Deduplication is handled atomically via INSERT ... ON CONFLICT DO NOTHING.
        use_copy=True enables faster COPY FROM STDIN (no dedup — first load only).
        """
        if logs_dir is None:
            raise ValueError("logs_dir is required for add_paths_from_source")

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

                cur = self.conn.cursor()
                try:
                    if use_copy:
                        t0 = time.time()
                        rows_inserted = DBQueries.copy_paths_batch(
                            cur, batch, initial_state=self.fetch_state
                        )
                        main_db_insert_time = time.time() - t0
                        already_seen = 0
                    else:
                        t0 = time.time()
                        rows_inserted = DBQueries.insert_paths_batch(
                            cur, batch, initial_state=self.fetch_state
                        )
                        main_db_insert_time = time.time() - t0
                        already_seen = len(batch) - rows_inserted

                    total_seen += already_seen
                    inserted_total += rows_inserted

                    if rows_inserted == 0:
                        logger.log_all_duplicates(
                            batch_num=batch_num,
                            walk_time=walk_time,
                            images_found=len(batch),
                            query_time=0.0,
                        )
                        continue

                    db_health = self.get_db_health()
                    logger.log_batch(
                        batch_num=batch_num,
                        walk_time=walk_time,
                        images_found=len(batch),
                        query_time=0.0,
                        already_seen=already_seen,
                        new_paths=rows_inserted,
                        main_db_insert_time=main_db_insert_time,
                        seen_cache_insert_time=0.0,
                        total_inserted=inserted_total,
                        total_runtime=time.time() - start_time,
                        db_health=db_health,
                    )
                finally:
                    cur.close()

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
        try:
            cur.execute("SELECT pg_database_size(current_database()) / 1048576.0")
            main_db_size_mb = round(cur.fetchone()[0], 2)
            state_rows = DBQueries.get_state_distribution(cur)
        finally:
            cur.close()

        total = sum(count for _, count in state_rows)
        state_distribution = {}
        for s, count in state_rows:
            pct = round(count / total * 100, 1) if total > 0 else 0.0
            state_distribution[f"state_{s}"] = {"count": count, "pct": pct}

        return {
            "main_db_size_mb": main_db_size_mb,
            "total_images_main": total,
            "state_distribution": state_distribution,
        }

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_and_fix_counts(self) -> dict:
        start = time.time()
        result = verify_counts(self.conn)
        print(f"Verification completed in {time.time() - start:.2f}s")

        if not result["valid"]:
            print(f"Count drift detected: {result['discrepancies']}")
            print("Fixing count drift...")
            fix_start = time.time()
            fix_result = fix_count_drift(self.conn)
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
    # Single-row lookups (Postgres dialect: %s placeholders, ANY(...))
    # ------------------------------------------------------------------

    def is_done(self, path: str) -> bool:
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT state FROM images WHERE path = %s", (path,))
            row = cur.fetchone()
            return row is not None and row[0] == 2
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
            cur.execute(
                "SELECT path, shard_path FROM images WHERE path = ANY(%s)",
                (paths,),
            )
            path_to_shard = {row[0]: row[1] for row in cur.fetchall()}
            return [path_to_shard.get(p) for p in paths]
        except Exception as e:
            if self.main_logger:
                self.main_logger.log_error("[DB] get_shard_paths", e, {"paths_count": len(paths)})
            raise
        finally:
            cur.close()
