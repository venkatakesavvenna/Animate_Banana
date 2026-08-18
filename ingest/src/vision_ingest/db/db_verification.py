"""
Unified database count verification and self-healing utilities.

Works with both SQLite (sqlite3.Connection) and PostgreSQL (psycopg.Connection).
Connection type is detected at runtime to select the correct SQL dialect and
transaction handling.
"""
import sqlite3
from typing import Optional


# ---------------------------------------------------------------------------
# Internal dialect helpers
# ---------------------------------------------------------------------------

def _is_sqlite(conn) -> bool:
    return isinstance(conn, sqlite3.Connection)


def _ph(conn) -> str:
    """Return the parameter placeholder for this connection type."""
    return "?" if _is_sqlite(conn) else "%s"


def _begin(conn, cur) -> None:
    """Start an explicit transaction (SQLite only; Postgres auto-starts)."""
    if _is_sqlite(conn):
        cur.execute("BEGIN;")


def _commit(conn, cur) -> None:
    if _is_sqlite(conn):
        cur.execute("COMMIT;")
    else:
        conn.commit()


def _rollback(conn, cur) -> None:
    if _is_sqlite(conn):
        cur.execute("ROLLBACK;")
    else:
        conn.rollback()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_counts(conn, seen_conn=None) -> dict:
    """
    Verify that cached counts in state_counts match actual row counts.

    Args:
        conn:       Main DB connection (SQLite or Postgres).
        seen_conn:  Seen-cache DB connection (SQLite only, optional).

    Returns:
        {"valid": bool, "discrepancies": [...]}
    """
    ph = _ph(conn)
    cur = conn.cursor()
    discrepancies = []

    try:
        cur.execute("SELECT state, count FROM state_counts ORDER BY state;")
        cached_counts = {state: count for state, count in cur.fetchall()}

        cur.execute("SELECT state, COUNT(*) FROM images GROUP BY state;")
        actual_counts = {state: count for state, count in cur.fetchall()}

        for state in range(5):  # states 0–4
            cached = cached_counts.get(state, 0)
            actual = actual_counts.get(state, 0)
            if cached != actual:
                discrepancies.append({
                    "state": state,
                    "cached": cached,
                    "actual": actual,
                    "drift": actual - cached,
                })
    finally:
        cur.close()

    # Seen-cache check (SQLite only)
    if seen_conn is not None:
        seen_cur = seen_conn.cursor()
        try:
            seen_cur.execute("SELECT count FROM seen_counts WHERE id = 1;")
            row = seen_cur.fetchone()
            cached_seen = row[0] if row else 0

            seen_cur.execute("SELECT COUNT(*) FROM seen;")
            actual_seen = seen_cur.fetchone()[0]

            if cached_seen != actual_seen:
                discrepancies.append({
                    "db": "seen",
                    "cached": cached_seen,
                    "actual": actual_seen,
                    "drift": actual_seen - cached_seen,
                })
        finally:
            seen_cur.close()

    return {"valid": len(discrepancies) == 0, "discrepancies": discrepancies}


def fix_count_drift(conn, seen_conn=None) -> dict:
    """
    Recompute all state counts from actual rows and overwrite state_counts.
    Also fixes seen_counts if seen_conn is provided.
    Calls verify_counts() after fix to confirm correction.

    Returns same shape as verify_counts() with 'fixed': True appended.
    """
    ph = _ph(conn)
    cur = conn.cursor()
    try:
        _begin(conn, cur)
        cur.execute("SELECT state, COUNT(*) FROM images GROUP BY state;")
        actual_counts = {state: count for state, count in cur.fetchall()}

        for state in range(5):
            count = actual_counts.get(state, 0)
            cur.execute(
                f"UPDATE state_counts SET count = {ph} WHERE state = {ph};",
                (count, state),
            )
        _commit(conn, cur)
    except Exception:
        _rollback(conn, cur)
        raise
    finally:
        cur.close()

    if seen_conn is not None:
        seen_ph = _ph(seen_conn)
        seen_cur = seen_conn.cursor()
        try:
            _begin(seen_conn, seen_cur)
            seen_cur.execute("SELECT COUNT(*) FROM seen;")
            actual_seen = seen_cur.fetchone()[0]
            seen_cur.execute(
                f"UPDATE seen_counts SET count = {seen_ph} WHERE id = 1;",
                (actual_seen,),
            )
            _commit(seen_conn, seen_cur)
        except Exception:
            _rollback(seen_conn, seen_cur)
            raise
        finally:
            seen_cur.close()

    result = verify_counts(conn, seen_conn)
    result["fixed"] = True
    return result


def _reset_state(conn, from_state: int, to_state: int) -> int:
    """
    Atomically move all rows from from_state to to_state and update state_counts.
    Returns number of paths reset.
    """
    ph = _ph(conn)
    cur = conn.cursor()
    try:
        _begin(conn, cur)

        cur.execute(f"SELECT COUNT(*) FROM images WHERE state = {ph};", (from_state,))
        count_to_reset = cur.fetchone()[0]

        if count_to_reset == 0:
            _commit(conn, cur)
            return 0

        cur.execute(
            f"UPDATE images SET state = {ph} WHERE state = {ph};",
            (to_state, from_state),
        )
        cur.execute(
            f"UPDATE state_counts SET count = count - {ph} WHERE state = {ph};",
            (count_to_reset, from_state),
        )
        cur.execute(
            f"UPDATE state_counts SET count = count + {ph} WHERE state = {ph};",
            (count_to_reset, to_state),
        )

        _commit(conn, cur)
        return count_to_reset
    except Exception:
        _rollback(conn, cur)
        raise
    finally:
        cur.close()


def reset_stuck_in_progress(conn) -> int:
    """
    Reset all state=1 (in-progress) paths back to state=0 (pending).
    WARNING: Only run when no workers are actively processing.
    Returns number of paths reset.
    """
    return _reset_state(conn, from_state=1, to_state=0)


def reset_failed_paths(conn) -> int:
    """
    Reset all state=3 (failed) paths back to state=0 (pending).
    WARNING: Only run when no workers are actively processing.
    Returns number of paths reset.
    """
    return _reset_state(conn, from_state=3, to_state=0)
