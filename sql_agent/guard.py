"""SQL guard: classify model-written SQL BEFORE it touches a database.

Defence in depth (each layer assumes the one before it failed):
  1. this guard       - lexical classification: read / write / blocked, one statement, no comments,
                        table allowlist, forced LIMIT
  2. dry run          - EXPLAIN on the read-only connection catches syntax/column errors cheaply
  3. read-only DB     - mode=ro + authorizer (db.py): the database itself refuses writes
  4. human approval   - the only path to a write, in the graph
  5. least privilege  - in real deployments, a DB role with SELECT only (SQLite has no roles)

A regex/keyword guard is NOT a parser. That is why layer 3 exists. In production prefer a real
SQL parser (sqlglot) in this layer; the structure of the API below does not change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["read", "write", "blocked"]

MAX_ROWS = 100
_WRITE = {"INSERT", "UPDATE", "DELETE"}
_FORBIDDEN_ANYWHERE = {
    "DROP", "ALTER", "CREATE", "PRAGMA", "ATTACH", "DETACH", "VACUUM", "REINDEX",
    "TRUNCATE", "GRANT", "REVOKE", "LOAD_EXTENSION", "SAVEPOINT", "BEGIN",
    "COMMIT", "ROLLBACK",
}
_STRING = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_TABLE_REF = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_LIMIT = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


@dataclass(frozen=True)
class Verdict:
    kind: Kind
    sql: str           # normalised SQL (trailing ';' removed, LIMIT added for reads)
    reason: str = ""


def _blocked(reason: str, sql: str = "") -> Verdict:
    return Verdict("blocked", sql, reason)


def validate_sql(sql: str, allowed_tables: set[str]) -> Verdict:
    raw = (sql or "").strip()
    if not raw:
        return _blocked("empty query")

    # Look at the SQL with string literals blanked out, so keywords inside 'text' don't count
    # and a ';' inside a literal isn't mistaken for a statement separator.
    skeleton = _STRING.sub("''", raw)
    if "--" in skeleton or "/*" in skeleton:
        return _blocked("comments are not allowed (they can hide intent)")
    body = skeleton.rstrip().rstrip(";").rstrip()
    if ";" in body:
        return _blocked("multiple statements are not allowed")

    words = re.findall(r"[A-Za-z_]+", body.upper())
    first = words[0] if words else ""
    if re.search(r"\bREPLACE\s+INTO\b|\bOR\s+REPLACE\b", body, re.IGNORECASE):
        return _blocked("forbidden keyword: REPLACE")  # replace() the FUNCTION stays allowed
    bad = _FORBIDDEN_ANYWHERE.intersection(words)
    if bad:
        return _blocked(f"forbidden keyword: {sorted(bad)[0]}")

    if first in _WRITE:
        kind: Kind = "write"
    elif first in {"SELECT", "WITH"}:
        kind = "write" if _WRITE.intersection(words) else "read"
    else:
        return _blocked(f"statement type not allowed: {first or '?'}")

    if kind == "write" and first in {"UPDATE", "DELETE"} and "WHERE" not in words:
        return _blocked("UPDATE/DELETE without a WHERE clause is not allowed")

    allowed_lower = {t.lower() for t in allowed_tables}
    ctes = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", body, re.IGNORECASE))
    for t in _TABLE_REF.findall(body):
        if t.lower() not in allowed_lower and t.lower() not in {c.lower() for c in ctes}:
            return _blocked(f"table not allowed: {t}")

    # normalised statement: original text minus trailing ';'
    clean = raw.rstrip().rstrip(";").rstrip()
    if kind == "read" and not _LIMIT.search(skeleton):
        clean = f"{clean} LIMIT {MAX_ROWS}"
    return Verdict(kind, clean, "")
