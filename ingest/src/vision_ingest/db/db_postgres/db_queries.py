import io
from typing import List, Tuple, Optional

# psycopg (v3) — installed via: pip install "psycopg[binary,pool]"
# NOTE: the package name changed from 'psycopg2' to 'psycopg' (no suffix)
# https://www.psycopg.org/psycopg3/docs/basic/index.html
import psycopg

# psycopg3 has no separate 'extras' module for execute_values.
# Bulk inserts are done via executemany() or copy() directly on the cursor.
# The old psycopg2.extras.execute_values() is no longer needed.


class DBQueries:

    # ------------------------------------------------------------------
    # init_schema
    # ------------------------------------------------------------------
    @staticmethod
    def init_schema(conn: psycopg.Connection) -> None:
        """
        Creates tables and index if they don't exist. Safe to call concurrently —
        Postgres handles DDL races correctly via its internal locking.
        No retry/backoff needed (unlike SQLite version).

        Creates:
            images      (path TEXT PRIMARY KEY, state INTEGER DEFAULT 0, shard_path TEXT)
            idx_images_state  ON images(state)
            state_counts (state INTEGER PRIMARY KEY, count INTEGER DEFAULT 0)
                rows: 0,1,2,3,4 — inserted with ON CONFLICT DO NOTHING (idempotent)

        Commits at end.
        """
        cur = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    path TEXT PRIMARY KEY,
                    state INTEGER DEFAULT 0,
                    shard_path TEXT
                );
            """)

            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_images_state ON images(state);"
            )

            cur.execute("""
                CREATE TABLE IF NOT EXISTS state_counts (
                    state INTEGER PRIMARY KEY,
                    count INTEGER DEFAULT 0
                );
            """)

            # Ensure all valid states exist (idempotent)
            cur.executemany(
                "INSERT INTO state_counts(state, count) VALUES (%s, 0) ON CONFLICT (state) DO NOTHING;",
                [(0,), (1,), (2,), (3,), (4,)],
            )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # insert_paths_batch
    # ------------------------------------------------------------------
    @staticmethod
    def insert_paths_batch(
        cur: psycopg.Cursor,
        paths: List[str],
        initial_state: int = 0,
    ) -> int:
        """
        Bulk-inserts paths using cursor.executemany().

        SQL:
            INSERT INTO images(path, state)
            VALUES (%s, %s)
            ON CONFLICT (path) DO NOTHING

        After insert:
            UPDATE state_counts SET count = count + <rows_inserted>
            WHERE state = initial_state

        Commits.
        Returns int: number of rows actually inserted (0 if all duplicates).
        """
        if not paths:
            return 0

        cur.executemany(
            "INSERT INTO images(path, state) VALUES (%s, %s) ON CONFLICT (path) DO NOTHING",
            [(p, initial_state) for p in paths],
        )
        rows_inserted = cur.rowcount

        if rows_inserted > 0:
            cur.execute(
                "UPDATE state_counts SET count = count + %s WHERE state = %s",
                (rows_inserted, initial_state),
            )

        cur.connection.commit()
        return rows_inserted

    # ------------------------------------------------------------------
    # copy_paths_batch
    # ------------------------------------------------------------------
    @staticmethod
    def copy_paths_batch(
        cur: psycopg.Cursor,
        paths: List[str],
        initial_state: int = 0,
    ) -> int:
        """
        Bulk-loads paths via COPY for maximum throughput (~1M+ rows/sec).

        psycopg3 pattern:
            with cur.copy("COPY images(path, state) FROM STDIN") as copy:
                for path in paths:
                    copy.write_row((path, initial_state))

        After copy:
            UPDATE state_counts SET count = count + len(paths) WHERE state = initial_state
        (COPY has no rowcount for conflicts — assumes all rows inserted.)

        Commits.
        Returns int: len(paths).

        !! WARNING !!
        COPY does not support ON CONFLICT. If any path already exists, Postgres
        raises UniqueViolation and the ENTIRE batch is rolled back.
        Only call when you are certain no duplicates exist.
        """
        if not paths:
            return 0

        with cur.copy("COPY images(path, state) FROM STDIN") as copy:
            for path in paths:
                copy.write_row((path, initial_state))

        count = len(paths)
        cur.execute(
            "UPDATE state_counts SET count = count + %s WHERE state = %s",
            (count, initial_state),
        )

        cur.connection.commit()
        return count

    # ------------------------------------------------------------------
    # fetch_and_lock_batch
    # ------------------------------------------------------------------
    @staticmethod
    def fetch_and_lock_batch(
        cur: psycopg.Cursor,
        batch_size: int,
        fetch_state: int = 0,
    ) -> List[str]:
        """
        Atomically fetches up to batch_size paths in state=fetch_state
        and transitions them to state=1 (in-progress).

        Uses FOR UPDATE SKIP LOCKED to guarantee no two workers get the same row.

        After update:
            UPDATE state_counts: decrement fetch_state, increment state 1

        Commits.
        Returns List[str] of locked paths. Empty list if nothing in fetch_state.
        """
        cur.execute(
            """
            WITH picked AS (
                SELECT path FROM images
                WHERE state = %s
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            )
            UPDATE images
            SET state = 1
            WHERE path IN (SELECT path FROM picked)
            RETURNING path;
            """,
            (fetch_state, batch_size),
        )
        rows = cur.fetchall()

        if not rows:
            cur.connection.commit()
            return []

        paths = [r[0] for r in rows]
        n = len(paths)

        cur.execute(
            "UPDATE state_counts SET count = count - %s WHERE state = %s",
            (n, fetch_state),
        )
        cur.execute(
            "UPDATE state_counts SET count = count + %s WHERE state = 1",
            (n,),
        )

        cur.connection.commit()
        return paths

    # ------------------------------------------------------------------
    # mark_state_transition
    # ------------------------------------------------------------------
    @staticmethod
    def mark_state_transition(
        cur: psycopg.Cursor,
        paths: List[str],
        from_state: int,
        to_state: int,
        shard_path: Optional[str] = None,
    ) -> int:
        """
        Transitions paths from from_state to to_state.
        Optionally sets shard_path in the same UPDATE.

        After update:
            UPDATE state_counts: decrement from_state, increment to_state

        Commits.
        Returns int: rows actually changed.

        Early return (no DB call) if paths is empty.
        """
        if not paths:
            return 0

        if shard_path is not None:
            cur.execute(
                "UPDATE images SET state = %s, shard_path = %s "
                "WHERE path = ANY(%s) AND state = %s",
                (to_state, shard_path, paths, from_state),
            )
        else:
            cur.execute(
                "UPDATE images SET state = %s "
                "WHERE path = ANY(%s) AND state = %s",
                (to_state, paths, from_state),
            )

        rows_changed = cur.rowcount

        if rows_changed > 0:
            cur.execute(
                "UPDATE state_counts SET count = count - %s WHERE state = %s",
                (rows_changed, from_state),
            )
            cur.execute(
                "UPDATE state_counts SET count = count + %s WHERE state = %s",
                (rows_changed, to_state),
            )

        cur.connection.commit()
        return rows_changed

    # ------------------------------------------------------------------
    # get_state_distribution
    # ------------------------------------------------------------------
    @staticmethod
    def get_state_distribution(
        cur: psycopg.Cursor,
    ) -> List[Tuple[int, int]]:
        """
        SELECT state, count FROM state_counts ORDER BY state

        Read-only. No commit needed.
        Returns List[Tuple[state_int, count_int]], always 5 rows (states 0–4).
        """
        cur.execute("SELECT state, count FROM state_counts ORDER BY state")
        return [(row[0], row[1]) for row in cur.fetchall()]