"""An append-only audit trail. "Who asked the database what, who approved the change, what happened?"
is the first thing a security review asks, and lesson 20 had no answer.

Design points:
  * APPEND-ONLY, ENFORCED BY THE DATABASE. Triggers abort any UPDATE or DELETE on the table, so a bug (or a
    curious engineer with the app's credentials) cannot quietly rewrite history. In production: a separate
    database/account the app can only INSERT into, or an immutable log sink.
  * METADATA, NOT CONTENT. We record the question, the SQL, row COUNTS and decisions, never the rows. An audit
    log full of customer data is itself a data-protection problem.
  * ONE EVENT PER FACT, written where the fact becomes true. Note where `approval_requested` is written
    (before the interrupt, in a node that runs once) and where `approval_decided` is written (after it):
    code before `interrupt()` runs again on resume, so an event there would be recorded twice.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  event TEXT NOT NULL,
  query_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  detail TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
  BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
  BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con:
            con.executescript(_DDL)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=5.0)

    def record(self, event: str, *, query_id: str, user_id: str, **detail: Any) -> None:
        with closing(self._connect()) as con, con:  # `with con` commits
            con.execute("INSERT INTO audit (ts, event, query_id, user_id, detail) VALUES (?,?,?,?,?)",
                        (datetime.now(UTC).isoformat(timespec="milliseconds"), event, query_id, user_id,
                         json.dumps(detail, default=str, sort_keys=True)))

    def recent(self, limit: int = 50, *, query_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT id, ts, event, query_id, user_id, detail FROM audit"
        args: tuple[Any, ...] = ()
        if query_id is not None:
            sql, args = sql + " WHERE query_id = ?", (query_id,)
        with closing(self._connect()) as con:
            rows = con.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [{"id": r[0], "ts": r[1], "event": r[2], "query_id": r[3], "user_id": r[4],
                 "detail": json.loads(r[5])} for r in reversed(rows)]
