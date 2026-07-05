"""SQLite-backed annotation store.

One row per sample holding the current (possibly edited) TikZ source plus
review status, independent of the pristine .tex files on disk. Editing in
the viewer never touches the original dataset files -- annotations live
entirely in this DB until exported.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

USERS = ["user_1", "user_2", "user_3"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    sample_id TEXT PRIMARY KEY,
    tex TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- 'pending' | 'done'
    annotated_by TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass
class Annotation:
    sample_id: str
    tex: str
    status: str
    annotated_by: str | None
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "status": self.status,
            "annotated_by": self.annotated_by,
            "updated_at": self.updated_at,
        }


class AnnotationDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, sample_id: str) -> Annotation | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM annotations WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            return Annotation(**dict(row)) if row else None

    def all_statuses(self) -> dict[str, Annotation]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM annotations").fetchall()
            return {r["sample_id"]: Annotation(**dict(r)) for r in rows}

    def save_draft(self, sample_id: str, tex: str, user: str) -> Annotation:
        """Save edited tex without changing done/pending status (unless no row yet)."""
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT status FROM annotations WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            status = existing["status"] if existing else "pending"
            conn.execute(
                """
                INSERT INTO annotations (sample_id, tex, status, annotated_by, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(sample_id) DO UPDATE SET
                    tex = excluded.tex,
                    annotated_by = excluded.annotated_by,
                    updated_at = excluded.updated_at
                """,
                (sample_id, tex, status, user),
            )
        return self.get(sample_id)

    def set_status(self, sample_id: str, tex: str, user: str, status: str) -> Annotation:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO annotations (sample_id, tex, status, annotated_by, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(sample_id) DO UPDATE SET
                    tex = excluded.tex,
                    status = excluded.status,
                    annotated_by = excluded.annotated_by,
                    updated_at = excluded.updated_at
                """,
                (sample_id, tex, status, user),
            )
        return self.get(sample_id)

    def progress(self) -> dict:
        with self._connect() as conn:
            total_done = conn.execute(
                "SELECT COUNT(*) AS c FROM annotations WHERE status = 'done'"
            ).fetchone()["c"]
            by_user = conn.execute(
                """
                SELECT annotated_by, COUNT(*) AS c FROM annotations
                WHERE status = 'done' GROUP BY annotated_by
                """
            ).fetchall()
            return {
                "done": total_done,
                "by_user": {r["annotated_by"]: r["c"] for r in by_user},
            }
