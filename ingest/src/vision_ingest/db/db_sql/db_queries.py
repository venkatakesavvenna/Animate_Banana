# db/db_queries.py
"""
Centralized SQL queries and database operations for the image database.
All SQL statements and common database operations are defined here for reusability.
"""

from typing import List, Tuple
import sqlite3


class DBQueries:
    """SQL queries and helper methods for the main images database."""
    
    # ========== Reusable Queries (used in multiple places) ==========
    
    GET_CHANGES_COUNT = "SELECT changes();"
    UPDATE_STATE = "UPDATE images SET state = ? WHERE path = ?;"
    
    # ========== Helper Methods ==========
    
    @staticmethod
    def init_database_pragmas(conn: sqlite3.Connection):
        """Initialize one-time database-level persistent pragmas.
        
        This should only be called during first initialization.
        journal_mode is persistent and stored in the database file.
        Safe to call multiple times (idempotent).
        
        CHANGED 2026-02-24: WAL → PERSIST journal mode for FSx Lustre compatibility.
        WAL mode uses mmap()-based shared memory (-shm file) which does not work
        correctly on network filesystems. PERSIST uses only fcntl() file locks
        which Lustre's LDLM (distributed lock manager) supports, and avoids the
        file create/delete overhead of DELETE mode on network FS.
        Trade-off: No concurrent readers during writes, but correct locking.
        """
        cur = conn.cursor()
        try:
            # --- OLD: WAL mode — broken on network/shared filesystems (FSx Lustre) ---
            # WAL relies on mmap() shared memory (-shm file) for cross-process coordination.
            # On network FS, mmap() does not provide coherent shared memory across nodes,
            # causing silent corruption, "disk I/O error", or "database is locked" errors.
            # cur.execute("PRAGMA journal_mode=WAL;")
            
            # --- OLD: DELETE journal mode — safe but slow on network FS ---
            # DELETE creates and deletes the journal file on every transaction.
            # On network FS, file create/unlink are expensive metadata operations
            # that go through Lustre's distributed lock manager on every commit.
            # cur.execute("PRAGMA journal_mode=DELETE;")
            
            # --- NEW: PERSIST journal mode — safe AND fast for FSx Lustre ---
            # Same locking model as DELETE (fcntl byte-range locks, no mmap/-shm).
            # But instead of deleting the journal file after each transaction,
            # it zeros out the journal header. This avoids the expensive file
            # create/unlink metadata round-trips on network FS.
            # Reference: https://www.sqlite.org/pragma.html#pragma_journal_mode
            cur.execute("PRAGMA journal_mode=PERSIST;")
        finally:
            cur.close()
    
    @staticmethod
    def init_connection_pragmas(conn: sqlite3.Connection):
        """Initialize per-connection pragmas.
        
        These must be set for EVERY connection as they are connection-scoped.
        
        CHANGED 2026-02-24: Adapted for DELETE journal mode on FSx Lustre.
        - synchronous upgraded NORMAL → FULL for crash safety on network FS
        - wal_autocheckpoint removed (not applicable in DELETE mode)
        - busy_timeout increased 10s → 60s for network FS lock latency
        """
        cur = conn.cursor()
        try:
            # --- OLD: NORMAL sync was fine for WAL (WAL has its own crash safety) ---
            # cur.execute("PRAGMA synchronous=NORMAL;")
            # --- NEW: FULL sync required for DELETE mode on network FS ---
            # In DELETE mode, NORMAL can lose data on OS crash. FULL ensures
            # journal is fsynced before writing to main DB file.
            cur.execute("PRAGMA synchronous=FULL;")
            
            cur.execute("PRAGMA temp_store=MEMORY;")
            
            # --- OLD: WAL-specific autocheckpoint (not applicable in DELETE mode) ---
            # cur.execute("PRAGMA wal_autocheckpoint=10000;")
            
            # --- OLD: 10s busy timeout — too short for network FS lock contention ---
            # cur.execute("PRAGMA busy_timeout=10000;")
            # --- NEW: 60s busy timeout — network FS lock acquisition is slower ---
            # Lustre LDLM lock round-trips can take 1-10ms vs 1μs on local FS.
            # Under multi-node contention, 10s was insufficient.
            cur.execute("PRAGMA busy_timeout=60000;")
        finally:
            cur.close()
    
    @staticmethod
    def init_schema(conn: sqlite3.Connection):
        """Initialize and migrate the main images database schema safely.
        
        Uses BEGIN IMMEDIATE to acquire write lock upfront and fail fast
        if another process is already initializing.
        """
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE;")

        # ------------------------------------------------------------------
        # 1. Create images table (safe if it already exists)
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                path TEXT PRIMARY KEY,
                state INTEGER DEFAULT 0,
                shard_path TEXT
            );
        """)

        # ------------------------------------------------------------------
        # 2. Backward compatibility: ensure shard_path column exists
        # ------------------------------------------------------------------
        cur.execute("PRAGMA table_info(images);")
        columns = {col[1] for col in cur.fetchall()}

        if "shard_path" not in columns:
            cur.execute("ALTER TABLE images ADD COLUMN shard_path TEXT;")

        # Optional but defensive: ensure required columns exist
        if "state" not in columns:
            raise RuntimeError("images table exists but is missing required 'state' column")

        # ------------------------------------------------------------------
        # 3. Index for fast state queries
        # ------------------------------------------------------------------
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_state ON images(state);"
        )

        # ------------------------------------------------------------------
        # 4. Create state_counts table
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS state_counts (
                state INTEGER PRIMARY KEY,
                count INTEGER DEFAULT 0
            );
        """)

        # ------------------------------------------------------------------
        # 5. Ensure all valid states exist (idempotent + race-safe)
        # ------------------------------------------------------------------
        cur.executemany(
            "INSERT OR IGNORE INTO state_counts(state, count) VALUES (?, 0);",
            [(0,), (1,), (2,), (3,), (4,)]
        )

        # ------------------------------------------------------------------
        # 7. Commit atomically
        # ------------------------------------------------------------------
        cur.execute("COMMIT;")
        cur.close()

    
    @staticmethod
    def insert_paths_batch(cur: sqlite3.Cursor, paths: List[str]) -> int:
        """
        Insert a batch of paths with state=0 and update state_0 count.
        Returns the number of paths actually inserted (excludes duplicates).
        """
        cur.execute("BEGIN;")
        
        # Track how many paths were actually inserted (vs ignored as duplicates)
        rows_inserted = 0
        for path_tuple in [(p,) for p in paths]:
            cur.execute("INSERT OR IGNORE INTO images(path, state) VALUES (?, 0);", path_tuple)
            rows_inserted += cur.execute(DBQueries.GET_CHANGES_COUNT).fetchone()[0]
        
        # Only increment count by actually inserted rows
        if rows_inserted > 0:
            cur.execute(
                "UPDATE state_counts SET count = count + ? WHERE state = 0;",
                (rows_inserted,)
            )
        
        cur.execute("COMMIT;")
        return rows_inserted
    
    @staticmethod
    def fetch_and_lock_batch(cur: sqlite3.Cursor, batch_size: int, fetch_state: int = 0) -> List[str]:
        """
        Atomically fetch a batch of pending images and mark them as in-progress.
        Returns list of image paths.
        
        Args:
            cur: Database cursor
            batch_size: Number of images to fetch
            fetch_state: State to fetch from (0 for first stage, 4 for subsequent stages)
        """
        cur.execute("BEGIN IMMEDIATE;")
        
        # Fetch pending images from specified state
        cur.execute(
            "SELECT path FROM images WHERE state = ? LIMIT ?;",
            (fetch_state, batch_size)
        )
        rows = cur.fetchall()
        
        if not rows:
            cur.execute("COMMIT;")
            return []
        
        paths = [r[0] for r in rows]
        count = len(paths)
        
        # Update state from 0 to 1
        cur.executemany(
            DBQueries.UPDATE_STATE,
            [(1, p) for p in paths]
        )
        
        # Update counts: decrement fetch_state, increment state_1
        cur.execute(
            "UPDATE state_counts SET count = count - ? WHERE state = ?;",
            (count, fetch_state)
        )
        cur.execute(
            "UPDATE state_counts SET count = count + ? WHERE state = 1;",
            (count,)
        )
        
        cur.execute("COMMIT;")
        return paths
    
    @staticmethod
    def mark_state_transition(cur: sqlite3.Cursor, paths: List[str], from_state: int, to_state: int, shard_path: str = None) -> int:
        """
        Mark paths as transitioning from one state to another.
        Only updates paths currently in from_state (idempotent).
        Optionally set shard_path for tracking output location.
        Returns the number of rows actually changed.
        """
        if not paths:
            return 0
        
        cur.execute("BEGIN;")
        
        # Build dynamic query with correct number of placeholders
        placeholders = ",".join(["?"] * len(paths))
        
        if shard_path is not None:
            # Update state and shard_path
            cur.execute(
                f"UPDATE images SET state = ?, shard_path = ? WHERE path IN ({placeholders}) AND state = ?;",
                [to_state, shard_path] + paths + [from_state]
            )
        else:
            # Update state only
            cur.execute(
                f"UPDATE images SET state = ? WHERE path IN ({placeholders}) AND state = ?;",
                [to_state] + paths + [from_state]
            )
        
        # Get actual number of rows changed
        rows_changed = cur.execute(DBQueries.GET_CHANGES_COUNT).fetchone()[0]
        
        if rows_changed > 0:
            # Update state counts: decrement from_state, increment to_state
            cur.execute(
                "UPDATE state_counts SET count = count - ? WHERE state = ?;",
                (rows_changed, from_state)
            )
            cur.execute(
                "UPDATE state_counts SET count = count + ? WHERE state = ?;",
                (rows_changed, to_state)
            )
        
        cur.execute("COMMIT;")
        return rows_changed
    
    @staticmethod
    def get_state_distribution(cur: sqlite3.Cursor) -> List[Tuple[int, int]]:
        """Get current state distribution from state_counts table."""
        cur.execute("SELECT state, count FROM state_counts;")
        return cur.fetchall()
    
    # --- OLD: WAL checkpoint — not applicable in DELETE journal mode ---
    # In DELETE mode, changes are written directly to the main DB file
    # (via the rollback journal), so there is no WAL to checkpoint.
    # Keeping as commented-out reference.
    # @staticmethod
    # def checkpoint_wal(cur: sqlite3.Cursor):
    #     """Perform WAL checkpoint to flush changes to main database."""
    #     cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    pass  # placeholder to keep class valid if this was the last method


class SeenCacheQueries:
    """SQL queries and helper methods for the seen cache database."""
    
    # ========== Helper Methods ==========
    
    @staticmethod
    def init_schema(conn: sqlite3.Connection):
        """Initialize the seen cache database schema.
        
        Uses BEGIN IMMEDIATE to acquire write lock upfront.
        """
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE;")
        
        # Create seen paths table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS seen (
                path TEXT PRIMARY KEY
            );
        """)
        
        # Create seen counts table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS seen_counts (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                count INTEGER DEFAULT 0
            );
        """)
        
        # Initialize count if empty
        cur.execute("SELECT COUNT(*) FROM seen_counts;")
        if cur.fetchone()[0] == 0:
            cur.execute("INSERT INTO seen_counts(id, count) VALUES (1, 0);")
        
        # Create trigger to auto-increment count on insert
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS increment_seen_count
            AFTER INSERT ON seen
            BEGIN
                UPDATE seen_counts SET count = count + 1 WHERE id = 1;
            END;
        """)
        
        # Create trigger to auto-decrement count on delete
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS decrement_seen_count
            AFTER DELETE ON seen
            BEGIN
                UPDATE seen_counts SET count = count - 1 WHERE id = 1;
            END;
        """)
        
        cur.execute("COMMIT;")
        cur.close()
    
    @staticmethod
    def filter_new_paths(cur: sqlite3.Cursor, paths: List[str]) -> List[str]:
        """
        Return only those paths not already in the seen cache.
        """
        if not paths:
            return []
        
        placeholders = ",".join(["?"] * len(paths))
        cur.execute(
            f"SELECT path FROM seen WHERE path IN ({placeholders});",
            paths
        )
        seen = {row[0] for row in cur.fetchall()}
        
        return [p for p in paths if p not in seen]
    
    @staticmethod
    def insert_paths_batch(cur: sqlite3.Cursor, paths: List[str]):
        """Insert a batch of paths into the seen cache."""
        cur.execute("BEGIN;")
        cur.executemany(
            "INSERT OR IGNORE INTO seen(path) VALUES (?);",
            [(p,) for p in paths]
        )
        cur.execute("COMMIT;")
    
    @staticmethod
    def get_total_count(cur: sqlite3.Cursor) -> int:
        """Get total number of seen paths."""
        cur.execute("SELECT count FROM seen_counts WHERE id = 1;")
        return cur.fetchone()[0]