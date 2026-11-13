"""Database access for the production SQL agent. Every function opens its OWN short-lived connection.

What changed from lesson 20 (`sql_agent/db.py`) and why:

  * ONE SHARED CONNECTION -> A CONNECTION PER OPERATION. Lesson 20 built one read-only connection at graph
    build time and shared it (check_same_thread=False, no lock) across every request. Under a threaded
    server that is a race. Here nothing is shared; with Postgres this becomes "borrow from a pool".
  * ONE GLOBAL READ RULE -> A POLICY ENFORCED BY THE DATABASE. The authorizer below allows a read only if
    the caller's role may read that table AND that column. A buggy guard can no longer leak `email` or
    `sqlite_master`: SQLite refuses at compile time (error code SQLITE_AUTH).
  * NO RESOURCE LIMITS -> THREE. A wall-clock statement timeout (progress handler), a cap on the size of
    any value a query can build (`SQLITE_LIMIT_LENGTH`, stops `SELECT zeroblob(1e9)`), and a row cap applied
    when rows LEAVE the database (`fetchmany(max + 1)`), because editing the SQL to add a LIMIT is bypassable.
  * PREVIEW, THEN BLIND APPLY -> APPLY WITH A DRIFT CHECK. Lesson 20 previewed a write, waited for a human,
    then ran the SQL again. If the data changed in between, the human approved something that no longer
    exists. `apply_write` re-runs the statement inside one transaction and rolls back unless it affects
    exactly the number of rows the human approved.
  * FOREIGN KEYS ON for the writer, so a preview reports "FOREIGN KEY constraint failed" before anyone
    is asked to approve something that cannot succeed.

Postgres equivalents: a role per policy with column GRANTs, `SET default_transaction_read_only = on`,
`statement_timeout`, `SELECT ... FOR UPDATE` or a version column for the drift check.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sql_agent.db import create_demo_db as _create_base_db
from sql_prod.config import Settings
from sql_prod.guard import DENIED_FUNCTIONS
from sql_prod.policy import Policy

_OK, _DENY = sqlite3.SQLITE_OK, sqlite3.SQLITE_DENY
_BUSY_TIMEOUT_S = 5.0


class WriteDrift(Exception):
    """The approved write would now affect a different number of rows than the human saw."""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"expected {expected} row(s), the statement would now change {actual}")
        self.expected, self.actual = expected, actual


@dataclass(frozen=True)
class ReadResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool  # more rows existed than max_rows


# --------------------------------------------------------------------------- demo data
def create_demo_db(path: str | Path) -> Path:
    """Lesson 20's shop database plus a PII column (`customers.email`), so column-level policy has
    something to protect. Resets in place and is safe to call repeatedly."""
    path = _create_base_db(path)
    with closing(sqlite3.connect(path)) as con:
        con.execute("ALTER TABLE customers ADD COLUMN email TEXT")
        con.execute("UPDATE customers SET email = lower(replace(name, ' ', '.')) || '@example.com'")
        con.commit()
    return path


# --------------------------------------------------------------------------- error classification
def is_denied(exc: sqlite3.Error) -> bool:
    """True when the DATABASE refused on permission grounds (the authorizer said no)."""
    return bool(getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_AUTH)


def is_timeout(exc: sqlite3.Error) -> bool:
    return bool(getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT)


# --------------------------------------------------------------------------- authorizers
Authorizer = Callable[[int, str | None, str | None, str | None, str | None], int]


def _function_ok(name: str | None) -> int:
    return _DENY if (name or "").lower() in DENIED_FUNCTIONS else _OK


def reader_authorizer(policy: Policy) -> Authorizer:
    """Read-only, restricted to what `policy` may read. Everything not listed is denied (fail closed)."""

    def authorize(action: int, a1: str | None, a2: str | None, db_name: str | None, _src: str | None) -> int:
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return _OK
        if action == sqlite3.SQLITE_FUNCTION:
            return _function_ok(a2)
        if action == sqlite3.SQLITE_READ:
            if db_name is None:  # a recursive CTE reading ITSELF: no database object is touched
                return _OK
            return _OK if policy.can_read_column(a1 or "", a2 or "") else _DENY
        return _DENY  # INSERT/UPDATE/DELETE, DDL, PRAGMA, ATTACH, transactions...

    return authorize


def writer_authorizer(policy: Policy) -> Authorizer:
    """Reads as above, plus INSERT/UPDATE/DELETE on the tables the role may modify, plus BEGIN/ROLLBACK/COMMIT.
    Still no DDL, PRAGMA or ATTACH: a compromised write path can change rows, never the schema."""
    read = reader_authorizer(policy)

    def authorize(action: int, a1: str | None, a2: str | None, db: str | None, src: str | None) -> int:
        if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
            return _OK if policy.can_write_table(a1 or "") else _DENY
        if action == sqlite3.SQLITE_TRANSACTION:
            return _OK
        return read(action, a1, a2, db, src)

    return authorize


# --------------------------------------------------------------------------- connections
def _open(path: Path, settings: Settings, authorizer: Authorizer, *, writer: bool) -> sqlite3.Connection:
    if writer:
        con = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S, isolation_level=None)  # we manage transactions
        con.execute("PRAGMA foreign_keys = ON")  # before the authorizer, which denies PRAGMA
    else:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"  # the OS-level read-only open from lesson 20
        con = sqlite3.connect(uri, uri=True, timeout=_BUSY_TIMEOUT_S)
    con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, settings.max_value_bytes)
    con.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, settings.max_sql_chars * 4)
    deadline = time.monotonic() + settings.statement_timeout_s
    # Called every 1000 VM instructions; a non-zero return aborts the statement ("interrupted").
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    con.set_authorizer(authorizer)
    return con


def _reader(path: Path, policy: Policy, settings: Settings) -> sqlite3.Connection:
    return _open(path, settings, reader_authorizer(policy), writer=False)


def _writer(path: Path, policy: Policy, settings: Settings) -> sqlite3.Connection:
    return _open(path, settings, writer_authorizer(policy), writer=True)


# --------------------------------------------------------------------------- schema (trusted code path)
def schema_text(path: Path, policy: Policy) -> dict[str, str]:
    """table -> a CREATE-style description containing ONLY what `policy` may read.

    Lesson 20 handed the model the full schema. A model cannot leak, or even ask for, a column it has never
    been shown, and the authorizer refuses if it guesses. Read on a trusted connection, not via the agent."""
    out: dict[str, str] = {}
    with closing(sqlite3.connect(path)) as con:
        for table in sorted(policy.readable):
            info = con.execute(f"PRAGMA table_info({table})").fetchall()  # table comes from our own policy
            allowed = [r for r in info if r[1].lower() in policy.readable[table]]
            if not allowed:
                continue
            cols = [f"{r[1]} {r[2]}{' PRIMARY KEY' if r[5] else ''}" for r in allowed]
            names = {r[1].lower() for r in allowed}
            fks = [f"FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})"
                   for fk in con.execute(f"PRAGMA foreign_key_list({table})").fetchall()
                   if fk[3].lower() in names and policy.can_read_table(fk[2])]
            out[table] = f"CREATE TABLE {table} ({', '.join([*cols, *fks])})"
    return out


def ping(path: Path) -> bool:
    try:
        with closing(sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)) as con:
            con.execute("SELECT 1").fetchone()
        return True
    except sqlite3.Error:
        return False


# --------------------------------------------------------------------------- reads
def dry_run(path: Path, policy: Policy, sql: str, settings: Settings) -> None:
    """EXPLAIN compiles the statement: syntax, tables, columns AND the authorizer's verdict, without running it."""
    with closing(_reader(path, policy, settings)) as con:
        con.execute(f"EXPLAIN {sql}").fetchall()


def run_read(path: Path, policy: Policy, sql: str, settings: Settings) -> ReadResult:
    with closing(_reader(path, policy, settings)) as con:
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description or []]
        fetched = cur.fetchmany(settings.max_rows + 1)  # the cap is applied as rows leave the database
    truncated = len(fetched) > settings.max_rows
    return ReadResult(columns, [list(r) for r in fetched[: settings.max_rows]], truncated)


# --------------------------------------------------------------------------- writes
def _rollback(con: sqlite3.Connection) -> None:
    with suppress(sqlite3.Error):  # never let a failed rollback hide the error that caused it
        con.execute("ROLLBACK")


def preview_write(path: Path, policy: Policy, sql: str, settings: Settings) -> int:
    """Run the write inside a transaction, count affected rows, ROLL BACK. Nothing persists."""
    with closing(_writer(path, policy, settings)) as con:
        con.execute("BEGIN IMMEDIATE")
        try:
            return con.execute(sql).rowcount
        finally:
            _rollback(con)


def apply_write(path: Path, policy: Policy, sql: str, expected_rows: int, settings: Settings) -> int:
    """The only function that commits. All-or-nothing: if the statement no longer affects exactly the
    `expected_rows` the approver was shown, roll back and raise WriteDrift."""
    with closing(_writer(path, policy, settings)) as con:
        con.execute("BEGIN IMMEDIATE")  # take the write lock now: nobody changes rows between check and commit
        try:
            n = con.execute(sql).rowcount
            if n != expected_rows:
                raise WriteDrift(expected_rows, n)
            con.execute("COMMIT")
            return n
        except BaseException:
            _rollback(con)
            raise
