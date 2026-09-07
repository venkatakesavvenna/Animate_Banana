"""SQLite store for the study.

Follows `viewer/db.py`: a module-level SCHEMA string and a `_connect()`
contextmanager that commits on success. Extended in three ways that matter for
a study rather than a review tool.

**Append-only below `trial`.** `response`, `event`, `qc_flag`,
`calibration_attempt` and `participant_annotation` are never updated and never
deleted. There is deliberately no `update_response` or `delete_response` method
-- the absence of the method IS the enforcement, and `tests/test_study_scheduler.py`
asserts it. An exclusion decision writes a `participant_annotation` row; the raw
responses are byte-identical before and after, so a result can always be
recomputed with and without a QC rule.

**A revision is an append.** Participants are expected to revise a score as
they grow familiar with a figure -- that is why the answered-questions strip is
clickable. Each revision inserts a new `response` row; analysis reads the latest
per (trial, question) and the log keeps the whole trail.

**PII lives in its own table.** Names and roll numbers go in `participant_pii`,
joined only by admin views. `participant_id` is an opaque token everywhere else,
so an export for analysis carries no identity.

One connection per request: sqlite3 connections are not thread-safe and the app
runs threaded.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1

PRAGMAS = (
    "PRAGMA journal_mode = WAL",       # readers never block the writer
    "PRAGMA synchronous  = NORMAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA busy_timeout = 15000",     # contend, don't error
)

SCHEMA = """
-- ---------------------------------------------------------- bundle mirror --
-- Insert-or-ignore only. A changed stimulus gets a new narrative_id in a new
-- bundle; an existing row is never rewritten, because trials already collected
-- point at it.
CREATE TABLE IF NOT EXISTS diagram (
    diagram_id          TEXT NOT NULL,
    bundle_id           TEXT NOT NULL,
    title               TEXT,
    figure_media_id     TEXT NOT NULL,
    figure_w            INTEGER, figure_h INTEGER,
    source_collection   TEXT,
    domain              TEXT,
    layout_type         TEXT,
    context_dependence  TEXT,
    element_count       INTEGER, edge_count INTEGER, node_count INTEGER,
    element_density     TEXT, connectivity REAL, connectivity_level TEXT,
    hierarchy_depth     INTEGER, has_raster INTEGER DEFAULT 0,
    study_day           INTEGER,
    complexity          TEXT,                   -- complex | easy (main study)
    PRIMARY KEY (bundle_id, diagram_id)
);

CREATE TABLE IF NOT EXISTS narrative (
    narrative_id        TEXT PRIMARY KEY,
    bundle_id           TEXT NOT NULL,
    diagram_id          TEXT NOT NULL,
    animation_style     TEXT NOT NULL,
    method              TEXT NOT NULL,          -- animatebanana | baseline
    context_condition   TEXT NOT NULL,          -- with_context|without_context|not_applicable
    verification_state  TEXT NOT NULL,          -- pre_verification|verified|not_applicable
    correction_type     TEXT,
    correction_magnitude TEXT,
    narrative_version   INTEGER NOT NULL DEFAULT 1,
    n_frames            INTEGER, n_steps INTEGER,
    duration            REAL,
    spoken_step_fraction REAL,
    timing_source       TEXT,
    is_attention_check  INTEGER NOT NULL DEFAULT 0,
    degradation         TEXT,
    payload_json        TEXT NOT NULL           -- frames + timeline, served as-is
);

-- ------------------------------------------------------------- session --
CREATE TABLE IF NOT EXISTS participant (
    participant_id      TEXT PRIMARY KEY,
    education_level     TEXT,                   -- student | faculty | researcher
    background_json     TEXT,
    stage               TEXT NOT NULL DEFAULT 'registered',
    consent_version     TEXT, consented_at TEXT,
    config_version      INTEGER NOT NULL,
    bundle_id           TEXT NOT NULL,
    calibration_passed  INTEGER NOT NULL DEFAULT 0,
    calibration_attempts INTEGER NOT NULL DEFAULT 0,
    is_expert           INTEGER NOT NULL DEFAULT 0,   -- seeds the prep examples
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at        TEXT,
    completed_at        TEXT
);

-- Identity, deliberately apart. Never joined into an analysis export.
CREATE TABLE IF NOT EXISTS participant_pii (
    participant_id      TEXT PRIMARY KEY REFERENCES participant(participant_id),
    display_name        TEXT,
    roll_no             TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------- append-only --
CREATE TABLE IF NOT EXISTS trial (
    trial_id            TEXT PRIMARY KEY,
    participant_id      TEXT NOT NULL REFERENCES participant(participant_id),
    trial_index         INTEGER NOT NULL,
    experiment          TEXT NOT NULL,          -- exp1..exp5 | attention
    experiment_index    INTEGER NOT NULL,       -- position within this experiment
    cell_key            TEXT NOT NULL,
    diagram_id          TEXT NOT NULL,
    animation_style     TEXT NOT NULL,
    bundle_id           TEXT NOT NULL,
    -- Condition columns are WRITE-ONCE. Analysis recovers preference from
    -- these, never from which side of the screen was clicked.
    presentation_a_id   TEXT NOT NULL REFERENCES narrative(narrative_id),
    presentation_a_condition TEXT NOT NULL,
    presentation_b_id   TEXT REFERENCES narrative(narrative_id),
    presentation_b_condition TEXT,
    -- All contenders in served order (JSON lists). A tournament has three;
    -- a_/b_ above stay populated for the first two so older readers work.
    presentation_ids    TEXT,
    presentation_conditions TEXT,
    position_seed       TEXT NOT NULL,
    show_captions       INTEGER NOT NULL DEFAULT 0,
    assignment_reason   TEXT NOT NULL,
    is_attention_check  INTEGER NOT NULL DEFAULT 0,
    counts_toward_target INTEGER NOT NULL DEFAULT 1,
    config_version      INTEGER NOT NULL,
    status              TEXT NOT NULL DEFAULT 'open',   -- open|submitted|abandoned
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    served_at           TEXT,
    submitted_at        TEXT
);

-- One row per (trial, question) ANSWER EVENT. A revision appends; it never
-- overwrites. Read the newest per question.
CREATE TABLE IF NOT EXISTS response (
    response_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id            TEXT NOT NULL REFERENCES trial(trial_id),
    participant_id      TEXT NOT NULL,
    question_id         TEXT NOT NULL,
    value               TEXT NOT NULL,
    revision            INTEGER NOT NULL DEFAULT 0,
    server_ms_since_open INTEGER,
    submitted_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS event (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id            TEXT NOT NULL REFERENCES trial(trial_id),
    participant_id      TEXT NOT NULL,
    type                TEXT NOT NULL,
    slot                TEXT,                   -- A | B | SINGLE
    client_seq          INTEGER,
    t_video             REAL,
    t_wall_client       TEXT,
    -- Millisecond precision, spelled the long way: the `subsec` modifier to
    -- datetime() only exists from sqlite 3.42 and this host runs 3.37, where
    -- it silently returns NULL and trips the NOT NULL constraint.
    server_ts           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now')),
    payload_json        TEXT
);

CREATE TABLE IF NOT EXISTS calibration_attempt (
    attempt_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id      TEXT NOT NULL REFERENCES participant(participant_id),
    attempt_index       INTEGER NOT NULL,
    answers_json        TEXT NOT NULL,
    score               REAL NOT NULL,
    threshold           REAL NOT NULL,
    passed              INTEGER NOT NULL,
    submitted_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS qc_flag (
    flag_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id      TEXT NOT NULL,
    trial_id            TEXT,
    kind                TEXT NOT NULL,
    severity            TEXT NOT NULL,
    detail_json         TEXT,
    detected_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS participant_annotation (
    annotation_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id      TEXT NOT NULL,
    kind                TEXT NOT NULL,          -- exclude | include | note
    reason              TEXT NOT NULL,
    author              TEXT NOT NULL,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS study_config (
    config_version      INTEGER PRIMARY KEY AUTOINCREMENT,
    params_json         TEXT NOT NULL,
    state               TEXT NOT NULL,          -- open | paused | closed
    bundle_id           TEXT NOT NULL,
    note                TEXT, author TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS ix_trial_pt    ON trial(participant_id, status);
CREATE INDEX IF NOT EXISTS ix_trial_pe    ON trial(participant_id, experiment, status);
CREATE INDEX IF NOT EXISTS ix_trial_cell  ON trial(experiment, cell_key, status);
CREATE INDEX IF NOT EXISTS ix_event_trial ON event(trial_id, server_ts);
CREATE INDEX IF NOT EXISTS ix_resp_trial  ON response(trial_id, question_id, revision);
CREATE INDEX IF NOT EXISTS ix_narr_cell   ON narrative(bundle_id, diagram_id, animation_style);
CREATE INDEX IF NOT EXISTS ix_qc_pt       ON qc_flag(participant_id, kind);
-- One open trial per participant, enforced by the engine rather than by
-- careful callers: a double-clicked "next" makes the second INSERT raise and
-- the handler re-reads the trial that already exists.
CREATE UNIQUE INDEX IF NOT EXISTS ux_one_open_trial
    ON trial(participant_id) WHERE status = 'open';
"""


def new_id() -> str:
    return uuid.uuid4().hex


class StudyDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for pragma in PRAGMAS:
                conn.execute(pragma)
            conn.executescript(SCHEMA)
            conn.executescript(INDEXES)
            # Additive migration for databases created before the tournament
            # columns existed. Never rewrite, only add.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(trial)")}
            for col in ("presentation_ids", "presentation_conditions"):
                if col not in cols:
                    conn.execute(f"ALTER TABLE trial ADD COLUMN {col} TEXT")
            dcols = {r[1] for r in conn.execute("PRAGMA table_info(diagram)")}
            for col, typ in (("study_day", "INTEGER"), ("complexity", "TEXT")):
                if col not in dcols:
                    conn.execute(f"ALTER TABLE diagram ADD COLUMN {col} {typ}")

    @contextmanager
    def _connect(self, immediate: bool = False):
        conn = sqlite3.connect(self.db_path, timeout=10.0,
                               isolation_level=None if immediate else "")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 15000")
        try:
            if immediate:
                # Take the write lock up front. Upgrading a read transaction to
                # a write one mid-way is the classic sqlite deadlock, and trial
                # assignment reads counts before it inserts. Under a 16-thread
                # burst even the 15s busy wait was exceeded once; a few more
                # attempts cost nothing when idle and save a 500 under load.
                for attempt in range(3):
                    try:
                        conn.execute("BEGIN IMMEDIATE")
                        break
                    except sqlite3.OperationalError as exc:
                        if "locked" not in str(exc) or attempt == 2:
                            raise
                        time.sleep(0.2 * (attempt + 1))
            yield conn
            if immediate:
                conn.execute("COMMIT")
            else:
                conn.commit()
        except Exception:
            # Only roll back a transaction that is actually open. A contended
            # BEGIN IMMEDIATE can fail before one exists, and an unconditional
            # ROLLBACK then raises "cannot rollback - no transaction is
            # active", replacing the real error with a misleading one and
            # taking the request down with it.
            try:
                if conn.in_transaction:
                    conn.execute("ROLLBACK") if immediate else conn.rollback()
            except sqlite3.Error:
                pass          # the transaction is already gone; nothing to undo
            raise
        finally:
            conn.close()

    # -- bundle import --------------------------------------------------
    def import_bundle(self, manifest: dict) -> tuple[int, int]:
        """Mirror a bundle into the DB. Idempotent by construction."""
        bundle_id = manifest["bundle_id"]
        with self._connect() as conn:
            for d in manifest["diagrams"]:
                conn.execute("""
                    INSERT OR IGNORE INTO diagram
                    (diagram_id, bundle_id, title, figure_media_id, figure_w, figure_h,
                     source_collection, domain, layout_type, context_dependence,
                     element_count, edge_count, node_count, element_density,
                     connectivity, connectivity_level, hierarchy_depth, has_raster,
                     study_day, complexity)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    d["diagram_id"], bundle_id, d.get("title"), d["figure_media_id"],
                    d.get("figure_w"), d.get("figure_h"), d.get("source_collection"),
                    d.get("domain"), d.get("layout_type"), d.get("context_dependence"),
                    d.get("element_count"), d.get("edge_count"), d.get("node_count"),
                    d.get("element_density"), d.get("connectivity"),
                    d.get("connectivity_level"), d.get("hierarchy_depth"),
                    int(bool(d.get("has_raster"))),
                    d.get("study_day"), d.get("complexity")))

            for n in manifest["narratives"] + manifest.get("attention_checks", []):
                tl = n["timeline"]
                conn.execute("""
                    INSERT OR IGNORE INTO narrative
                    (narrative_id, bundle_id, diagram_id, animation_style, method,
                     context_condition, verification_state, correction_type,
                     correction_magnitude, narrative_version, n_frames, n_steps,
                     duration, spoken_step_fraction, timing_source,
                     is_attention_check, degradation, payload_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    n["narrative_id"], bundle_id, n["diagram_id"], n["animation_style"],
                    n["method"], n["context_condition"], n["verification_state"],
                    n.get("correction_type"), n.get("correction_magnitude"),
                    n.get("narrative_version", 1), n["n_frames"], n["n_steps"],
                    tl["duration"], n.get("spoken_step_fraction"),
                    tl.get("timing_source"), int(bool(n.get("is_attention_check"))),
                    n.get("degradation"),
                    json.dumps({"frames": n["frames"], "timeline": tl,
                                "frame_w": n.get("frame_w"), "frame_h": n.get("frame_h")})))

            counts = (conn.execute("SELECT COUNT(*) c FROM diagram WHERE bundle_id=?",
                                   (bundle_id,)).fetchone()["c"],
                      conn.execute("SELECT COUNT(*) c FROM narrative WHERE bundle_id=?",
                                   (bundle_id,)).fetchone()["c"])
        return counts

    # -- config ---------------------------------------------------------
    def put_config(self, params: dict, state: str, bundle_id: str,
                   note: str = "", author: str = "") -> int:
        """Configuration is versioned, never edited: a trial records the version
        it ran under, so a mid-study change cannot reinterpret earlier data."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO study_config (params_json, state, bundle_id, note, author)"
                " VALUES (?,?,?,?,?)",
                (json.dumps(params), state, bundle_id, note, author))
            return cur.lastrowid

    def active_config(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM study_config ORDER BY config_version DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        out = dict(row)
        out["params"] = json.loads(out.pop("params_json"))
        return out

    # -- participants ---------------------------------------------------
    def create_participant(self, *, education_level: str, config_version: int,
                           bundle_id: str, display_name: str = "",
                           roll_no: str = "", is_expert: bool = False,
                           background: dict | None = None) -> str:
        pid = new_id()
        with self._connect(immediate=True) as conn:
            conn.execute("""
                INSERT INTO participant
                (participant_id, education_level, background_json, config_version,
                 bundle_id, is_expert)
                VALUES (?,?,?,?,?,?)""",
                (pid, education_level, json.dumps(background or {}),
                 config_version, bundle_id, int(is_expert)))
            if display_name or roll_no:
                conn.execute("INSERT INTO participant_pii"
                             " (participant_id, display_name, roll_no) VALUES (?,?,?)",
                             (pid, display_name, roll_no))
        return pid

    def participant(self, participant_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM participant WHERE participant_id=?",
                               (participant_id,)).fetchone()
        return dict(row) if row else None

    def set_stage(self, participant_id: str, stage: str) -> None:
        with self._connect(immediate=True) as conn:
            conn.execute("UPDATE participant SET stage=?, last_seen_at=datetime('now')"
                         " WHERE participant_id=?", (stage, participant_id))

    # -- responses (append-only) ----------------------------------------
    def add_response(self, trial_id: str, participant_id: str, question_id: str,
                     value, ms_since_open: int | None = None) -> int:
        """Append an answer. A revision is a new row, not an update."""
        # immediate=True because this reads before it writes. Upgrading a
        # deferred read transaction to a write one is sqlite's classic deadlock
        # shape: two connections both hold read locks, neither can take the
        # write lock, and SQLITE_BUSY is returned *without* honouring
        # busy_timeout. Taking the write lock up front makes them queue instead.
        with self._connect(immediate=True) as conn:
            revision = conn.execute(
                "SELECT COUNT(*) c FROM response WHERE trial_id=? AND question_id=?",
                (trial_id, question_id)).fetchone()["c"]
            cur = conn.execute("""
                INSERT INTO response
                (trial_id, participant_id, question_id, value, revision, server_ms_since_open)
                VALUES (?,?,?,?,?,?)""",
                (trial_id, participant_id, question_id,
                 json.dumps(value), revision, ms_since_open))
            return cur.lastrowid

    def latest_answers(self, trial_id: str) -> dict:
        """Newest value per question -- what analysis reads."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT question_id, value, MAX(revision) AS revision
                FROM response WHERE trial_id=? GROUP BY question_id""",
                (trial_id,)).fetchall()
        return {r["question_id"]: json.loads(r["value"]) for r in rows}

    def add_event(self, trial_id: str, participant_id: str, type_: str, *,
                  slot: str | None = None, client_seq: int | None = None,
                  t_video: float | None = None, t_wall_client: str | None = None,
                  payload: dict | None = None) -> None:
        with self._connect(immediate=True) as conn:
            conn.execute("""
                INSERT INTO event
                (trial_id, participant_id, type, slot, client_seq, t_video,
                 t_wall_client, payload_json)
                VALUES (?,?,?,?,?,?,?,?)""",
                (trial_id, participant_id, type_, slot, client_seq, t_video,
                 t_wall_client, json.dumps(payload or {})))

    def annotate_participant(self, participant_id: str, kind: str, reason: str,
                             author: str) -> None:
        """Exclusion is an annotation. It never touches a response row."""
        with self._connect(immediate=True) as conn:
            conn.execute("INSERT INTO participant_annotation"
                         " (participant_id, kind, reason, author) VALUES (?,?,?,?)",
                         (participant_id, kind, reason, author))
