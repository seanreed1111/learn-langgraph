"""SQL guard, version 2: a PARSER, not a keyword scan.

Lesson 20's guard was a regex classifier and said so. Running it against itself found real holes:
`SELECT * FROM customers, sqlite_master` passed (the table regex only saw the name after FROM), and a
LIMIT inside a subquery stopped the outer LIMIT being added. Both are the classic failure of lexical
checks: the checker and the database disagree about what the text means.

This guard uses `sqlglot` to build the same kind of tree the database builds, then allow-lists:
  * exactly ONE statement; it must be SELECT/UNION (read) or INSERT/UPDATE/DELETE (write)
  * every table reference in the tree (comma joins, subqueries, CTEs, UNION arms) is on the allow-list
  * no schema-qualified names (main.sqlite_master), no table-valued functions (pragma_table_info(...))
  * no data-changing statement hidden inside a read, no dangerous functions
  * UPDATE/DELETE need a WHERE; writes only touch tables the caller's role may modify

And one more idea worth the whole change: **run what you validated**. `Verdict.sql` is the statement
re-serialised from the tree the guard approved, not the original string. If the guard's parser and SQLite
ever read the original text differently (a "parser differential"), the database only ever sees what the
guard saw.

It is still not the last line of defence. `db.py` makes the database enforce the same rules (authorizer,
read-only open, limits), because a guard that is never wrong does not exist.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

# sqlglot logs a WARNING when it falls back to a generic "Command" for syntax it does not model.
# That is exactly what we block anyway; keep it out of the service logs.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

Kind = Literal["read", "write", "blocked"]

DENIED_FUNCTIONS = frozenset({"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"})
_STRING = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_WRITE_NODES = (exp.Insert, exp.Update, exp.Delete)


@dataclass(frozen=True)
class Verdict:
    kind: Kind
    sql: str  # the statement to RUN: re-serialised from the approved tree (empty when blocked)
    reason: str = ""


def _blocked(reason: str) -> Verdict:
    return Verdict("blocked", "", reason)


def _function_names(tree: exp.Expression) -> set[str]:
    names: set[str] = set()
    for fn in tree.find_all(exp.Func):
        names.add((fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower())
    return names


def validate_sql(sql: str, readable: Collection[str], writable: Collection[str] = (), *,
                 max_chars: int = 4_000) -> Verdict:
    raw = (sql or "").strip()
    if not raw:
        return _blocked("empty query")
    if len(raw) > max_chars:
        return _blocked(f"query is too long (max {max_chars} characters)")

    # Comments can hide intent from a human reviewer. Look at the text with string literals blanked out,
    # so a '--' inside 'a--b' is data, not a comment.
    skeleton = _STRING.sub("''", raw)
    if "--" in skeleton or "/*" in skeleton:
        return _blocked("comments are not allowed (they can hide intent)")

    try:
        statements = [s for s in sqlglot.parse(raw, read="sqlite") if s is not None]
    except SqlglotError as exc:
        return _blocked(f"could not parse the statement: {str(exc).splitlines()[0][:120]}")
    if len(statements) != 1:
        return _blocked("multiple statements are not allowed" if statements else "empty query")
    stmt = statements[0]

    readable_l = {t.lower() for t in readable}
    writable_l = {t.lower() for t in writable}

    if isinstance(stmt, exp.Select | exp.SetOperation):
        kind: Kind = "read"
        if stmt.find(*_WRITE_NODES):
            return _blocked("a data-changing statement inside a query is not allowed")
    elif isinstance(stmt, _WRITE_NODES):
        kind = "write"
    else:
        return _blocked(f"statement type not allowed: {type(stmt).__name__.upper()}")

    ctes = {c.alias.lower() for c in stmt.find_all(exp.CTE) if c.alias}
    target: str | None = None
    if kind == "write":
        node = stmt.this.this if isinstance(stmt.this, exp.Schema) else stmt.this
        if not isinstance(node, exp.Table) or not isinstance(node.this, exp.Identifier):
            return _blocked("could not determine which table the write changes")
        target = str(node.name).lower()
        if isinstance(stmt, exp.Update | exp.Delete) and stmt.args.get("where") is None:
            return _blocked("UPDATE/DELETE without a WHERE clause is not allowed")

    for table in stmt.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return _blocked("table-valued functions are not allowed")
        if table.db or table.catalog:
            return _blocked(f"schema-qualified table names are not allowed: {table.sql(dialect='sqlite')}")
        name = table.name.lower()
        if name in ctes:
            continue
        if name not in readable_l:
            return _blocked(f"table not allowed: {table.name}")

    if target is not None and target not in writable_l:
        return _blocked(f"your role may not modify table: {target}")

    bad = _function_names(stmt) & DENIED_FUNCTIONS
    if bad:
        return _blocked(f"forbidden function: {sorted(bad)[0]}")

    return Verdict(kind, stmt.sql(dialect="sqlite"), "")
