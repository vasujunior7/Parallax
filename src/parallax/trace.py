"""Parallax decision trace store — SQLite-backed audit log.

Every call to ``decide()`` can be persisted here so that:

* Post-hoc analysis can replay the full decision history for a frame.
* The escalation trade-off curve (Stage 11) can be swept over different thresholds
  without re-running expensive inference — all scores are in the DB.
* The escalation UI can pull human-readable reason strings directly from this store.

Thread-safety: a single ``threading.Lock`` guards every write.  Multiple readers are
fine (SQLite WAL mode is enabled).

Example usage::

    from parallax.trace import TraceStore
    from parallax.agent import decide

    store = TraceStore("logs/run_2026-09-11.db")
    result = decide(state, config)
    store.log(result, frame_id="visa/pcb1/test/img_0001.JPG")

    # Later — sweep a lower threshold:
    rows = store.all_rows()
    escalations = [r for r in rows if r["ood_score"] > 10.0]

Schema (``decisions`` table):

    decision_id       TEXT PRIMARY KEY
    timestamp_utc     REAL
    frame_id          TEXT
    object_class      TEXT
    decision          TEXT     -- "ACCEPT" | "RELOOK" | "ESCALATE"
    rule_index        INTEGER
    reason            TEXT
    retry_count       INTEGER
    ood_score         REAL
    bsf_residual      REAL
    bsf_top_block_norm REAL
    vision_is_defective INTEGER  -- 0 | 1
    vision_n_defects  INTEGER
    vision_total_area_px REAL
    ood_relook_threshold   REAL
    ood_escalate_threshold REAL
    bsf_relook_threshold   REAL
    bsf_escalate_threshold REAL
    max_retries        INTEGER
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from parallax.agent import AgentResult, Decision


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id            TEXT PRIMARY KEY,
    timestamp_utc          REAL NOT NULL,
    frame_id               TEXT,
    object_class           TEXT NOT NULL,
    decision               TEXT NOT NULL,
    rule_index             INTEGER NOT NULL,
    reason                 TEXT NOT NULL,
    retry_count            INTEGER NOT NULL,
    ood_score              REAL NOT NULL,
    bsf_residual           REAL NOT NULL,
    bsf_top_block_norm     REAL NOT NULL,
    bsf_top_block_coord    TEXT,
    vision_is_defective    INTEGER NOT NULL,
    vision_n_defects       INTEGER NOT NULL,
    vision_total_area_px   REAL NOT NULL,
    ood_relook_threshold   REAL NOT NULL,
    ood_escalate_threshold REAL NOT NULL,
    bsf_relook_threshold   REAL NOT NULL,
    bsf_escalate_threshold REAL NOT NULL,
    max_retries            INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frame_id    ON decisions (frame_id);
CREATE INDEX IF NOT EXISTS idx_decision    ON decisions (decision);
CREATE INDEX IF NOT EXISTS idx_object_class ON decisions (object_class);
"""

_INSERT = """
INSERT OR REPLACE INTO decisions VALUES (
    :decision_id, :timestamp_utc, :frame_id, :object_class,
    :decision, :rule_index, :reason,
    :retry_count, :ood_score, :bsf_residual, :bsf_top_block_norm,
    :bsf_top_block_coord,
    :vision_is_defective, :vision_n_defects, :vision_total_area_px,
    :ood_relook_threshold, :ood_escalate_threshold,
    :bsf_relook_threshold, :bsf_escalate_threshold,
    :max_retries
)
"""


class TraceStore:
    """Append-only SQLite store for agent decisions.

    Parameters
    ----------
    path:
        File path for the SQLite database.  Created (with parent dirs) if absent.
        Pass ``":memory:"`` for an in-process ephemeral store (useful in tests).
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log(self, result: AgentResult, frame_id: Optional[str] = None) -> None:
        """Persist one decision.

        ``frame_id`` overrides ``result.state.frame_id`` if provided; otherwise
        ``result.state.frame_id`` is used (may be ``None``).
        """
        fid = frame_id if frame_id is not None else result.state.frame_id
        s   = result.state
        c   = result.config
        v   = s.verdict

        # Encode the block-coordinate vector as JSON so it survives a round-trip
        # through SQLite TEXT without loss of precision.
        coord_json = (
            json.dumps([float(x) for x in s.bsf_top_block_coord])
            if s.bsf_top_block_coord is not None
            else None
        )

        row = {
            "decision_id":            result.decision_id,
            "timestamp_utc":          result.timestamp_utc,
            "frame_id":               fid,
            "object_class":           s.object_class,
            "decision":               result.decision.name,
            "rule_index":             result.rule_index,
            "reason":                 result.reason,
            "retry_count":            s.retry_count,
            "ood_score":              s.ood_score,
            "bsf_residual":           s.bsf_residual,
            "bsf_top_block_norm":     s.bsf_top_block_norm,
            "bsf_top_block_coord":    coord_json,
            "vision_is_defective":    int(v.is_defective),
            "vision_n_defects":       len(v.defects),
            "vision_total_area_px":   v.total_area_px,
            "ood_relook_threshold":   c.ood_relook_threshold,
            "ood_escalate_threshold": c.ood_escalate_threshold,
            "bsf_relook_threshold":   c.bsf_relook_threshold,
            "bsf_escalate_threshold": c.bsf_escalate_threshold,
            "max_retries":            c.max_retries,
        }
        with self._lock:
            self._conn.execute(_INSERT, row)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def all_rows(self) -> list[dict]:
        """Return every row as a list of dicts (column → value)."""
        with self._lock:
            cur = self._conn.execute("SELECT * FROM decisions ORDER BY timestamp_utc")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def replay(self, frame_id: str) -> list[dict]:
        """All decisions for one frame, oldest first."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM decisions WHERE frame_id = ? ORDER BY timestamp_utc",
                (frame_id,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def escalation_rate(
        self,
        object_class: Optional[str] = None,
    ) -> float:
        """Fraction of decisions that were ESCALATE, optionally filtered by class.

        Returns 0.0 if the store is empty or the filter matches nothing.
        """
        with self._lock:
            if object_class:
                total = self._conn.execute(
                    "SELECT COUNT(*) FROM decisions WHERE object_class = ?",
                    (object_class,),
                ).fetchone()[0]
                esc = self._conn.execute(
                    "SELECT COUNT(*) FROM decisions "
                    "WHERE decision = 'ESCALATE' AND object_class = ?",
                    (object_class,),
                ).fetchone()[0]
            else:
                total = self._conn.execute(
                    "SELECT COUNT(*) FROM decisions"
                ).fetchone()[0]
                esc = self._conn.execute(
                    "SELECT COUNT(*) FROM decisions WHERE decision = 'ESCALATE'"
                ).fetchone()[0]
        return esc / total if total > 0 else 0.0

    def accept_rate(self, object_class: Optional[str] = None) -> float:
        """Fraction of decisions that were ACCEPT."""
        with self._lock:
            if object_class:
                total = self._conn.execute(
                    "SELECT COUNT(*) FROM decisions WHERE object_class = ?",
                    (object_class,),
                ).fetchone()[0]
                acc = self._conn.execute(
                    "SELECT COUNT(*) FROM decisions "
                    "WHERE decision = 'ACCEPT' AND object_class = ?",
                    (object_class,),
                ).fetchone()[0]
            else:
                total = self._conn.execute(
                    "SELECT COUNT(*) FROM decisions"
                ).fetchone()[0]
                acc = self._conn.execute(
                    "SELECT COUNT(*) FROM decisions WHERE decision = 'ACCEPT'"
                ).fetchone()[0]
        return acc / total if total > 0 else 0.0

    def decision_counts(self) -> dict[str, int]:
        """Return ``{decision_name: count}`` for all stored rows."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT decision, COUNT(*) FROM decisions GROUP BY decision"
            )
            return {row[0]: row[1] for row in cur.fetchall()}

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        with self._lock:
            self._conn.close()

    def escalations_for_ui(
        self,
        limit: int = 50,
        object_class: Optional[str] = None,
    ) -> list[dict]:
        """Return the most recent ESCALATE rows in a UI-friendly format.

        Decodes ``bsf_top_block_coord`` from JSON back to a Python list.
        """
        with self._lock:
            if object_class:
                cur = self._conn.execute(
                    "SELECT * FROM decisions WHERE decision='ESCALATE'"
                    " AND object_class=? ORDER BY timestamp_utc DESC LIMIT ?",
                    (object_class, limit),
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM decisions WHERE decision='ESCALATE'"
                    " ORDER BY timestamp_utc DESC LIMIT ?",
                    (limit,),
                )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]

        for row in rows:
            raw = row.get("bsf_top_block_coord")
            row["bsf_top_block_coord"] = json.loads(raw) if raw else None

        return rows

    def __enter__(self) -> "TraceStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()
