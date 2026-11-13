"""The production SQL analyst graph.

    START -> guard_input --(empty / too long / role can read nothing)--> reject --------------------+
                |                                                                                    |
                v                                                                                    |
            load_schema   (schema FILTERED by the caller's role)                                     |
                |                                                                                    |
                v                                                                                    |
           write_query <---------------------- bump <-- error, repairs left                          |
                |                                                                                    |
                v                                                                                    |
           check_query  parse-guard, dry run on the role's connection, rolled-back write preview     |
                |-- blocked / not authorized ---------> refuse -------------------------------------+
                |-- nothing to answer ----------------> out_of_scope ------------------------------+
                |-- error, no repairs left -----------> fail ---------------------------------------+
                |-- read  --> run_query -> answer ---------------------------------------------------+
                |                  '-- timeout -------> timed_out ----------------------------------+
                |-- write, 0 rows ---------------------> no_change ---------------------------------+
                '-- write --> approve_write (interrupt) -> apply_write ------------------------------+
                                   '-- not approved / not allowed / expired / mismatch -> cancelled -+
                                                                                                     v
                                                                                       record (audit) -> END

Compared with lesson 20 this adds: caller identity and per-role policy (`context`); typed state with narrow
input/output schemas and a per-turn reset; a parser-based guard; DB-enforced column permissions, timeouts and
size limits; an approval that is bound to the exact statement, to a different person, and to a time window; a
drift check when the write is finally applied; an append-only audit record; timing logs, a node trace and
retries on transient model errors only.

Still a FIXED workflow rather than a tool-calling loop, for the lesson-20 reasons: the steps never change, and
the dangerous decision is a node we own.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, interrupt
from typing_extensions import TypedDict

from rag_prod.observability import timed
from sql_prod import db
from sql_prod.audit import AuditLog
from sql_prod.config import Settings
from sql_prod.guard import validate_sql
from sql_prod.offline import NO_QUERY, ProdOfflineWriter
from sql_prod.policy import ANONYMOUS, SqlContext, policy_for


# ------------------------------------------------------------------ state & schemas
class SqlState(TypedDict, total=False):
    question: str
    query_id: str  # = the thread id: the handle a client uses to poll or decide
    requested_by: str  # who asked; the approver must be someone else
    requested_role: str  # the role the statement was validated under; the write runs under it too
    tables: list[str]
    schema: dict[str, str]
    sql: str  # the statement we will RUN (re-serialised from the guard's tree)
    sql_sha256: str  # fingerprint of `sql`; the approver must present it back
    kind: str  # none | read | write | blocked | error | timeout | noop
    error: str | None
    repairs: int
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    affected: int
    preview_at: float  # when the write was previewed (approval expires from here)
    approved_by: str
    status: str  # answered | refused | out_of_scope | failed | timeout | rejected | applied | cancelled | no_change | stale
    answer: str
    trace: list[str]


class SqlInput(TypedDict):
    question: str


class SqlOutput(TypedDict, total=False):
    """The public contract. Schemas, fingerprints and counters are internals."""

    query_id: str
    status: str
    answer: str
    sql: str
    sql_sha256: str
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    affected: int
    repairs: int
    trace: list[str]


def _transient(exc: Exception) -> bool:
    """Retry network-ish MODEL failures only. A database error is deterministic; retrying it just repeats it."""
    return isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {
        "RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError", "OverloadedError",
    }


RETRY = RetryPolicy(max_attempts=3, initial_interval=0.5, retry_on=_transient)


def _ctx(runtime: Runtime[SqlContext]) -> SqlContext:
    return runtime.context or ANONYMOUS  # no identity -> DENY_ALL policy: fail closed


def _sha(sql: str) -> str:
    return hashlib.sha256(sql.encode()).hexdigest()


def _query_id(config: RunnableConfig) -> str:
    return str(config.get("configurable", {}).get("thread_id", ""))


# ------------------------------------------------------------------ graph factory
def build_graph(settings: Settings, writer=None, audit: AuditLog | None = None, checkpointer=None,
                clock: Callable[[], float] = time.time):
    """Dependencies are PARAMETERS: tests inject a stub writer, a fake clock and a temp database.
    `checkpointer=None` is for Studio/the platform, which supplies persistence. A checkpointer is REQUIRED
    for the approval flow in your own service: the pending approval lives in it (use a durable one)."""
    writer = writer or ProdOfflineWriter()
    audit = audit or AuditLog(settings.audit_db)
    path = Path(settings.db_path)

    def _t(state: SqlState, name: str) -> list[str]:
        return [*state.get("trace", []), name]

    def _audit(event: str, state: SqlState, query_id: str, **detail: Any) -> None:
        audit.record(event, query_id=query_id, user_id=state.get("requested_by", "unknown"), **detail)

    # ---------------------------------------------------------------- entry
    @timed("guard_input")
    def guard_input(state: SqlState, runtime: Runtime[SqlContext], config: RunnableConfig
                    ) -> Command[Literal["load_schema", "reject"]]:
        ctx = _ctx(runtime)
        text = (state.get("question") or "").strip()
        # !! Reset EVERY per-turn field. The checkpointer keeps state for the thread; a stale `status`,
        # `rows` or `repairs` from an earlier turn would leak into this one (lesson 14, section 3.4).
        fresh: dict[str, Any] = {
            "query_id": _query_id(config), "requested_by": ctx.user_id, "requested_role": ctx.role,
            "question": text, "sql": "",
            "sql_sha256": "", "kind": "", "error": None, "repairs": 0, "columns": [], "rows": [],
            "truncated": False, "affected": 0, "preview_at": 0.0, "approved_by": "", "status": "",
            "answer": "", "trace": ["guard_input"],
        }
        reason = ""
        if not text:
            reason = "Please enter a question."
        elif len(text) > settings.max_question_chars:
            reason = f"Question is too long (max {settings.max_question_chars} characters)."
        elif not ctx.policy.readable:
            reason = "Your account does not have access to any data."
        if reason:
            return Command(update={**fresh, "answer": reason}, goto="reject")  # ty: ignore[invalid-explicit-return]
        audit.record("question", query_id=fresh["query_id"], user_id=ctx.user_id, role=ctx.role,
                     question=text[:500])
        return Command(update=fresh, goto="load_schema")  # ty: ignore[invalid-explicit-return]

    @timed("load_schema")
    def load_schema(state: SqlState, runtime: Runtime[SqlContext]) -> dict[str, Any]:
        schema = db.schema_text(path, _ctx(runtime).policy)  # trusted path, filtered by role
        return {"tables": sorted(schema), "schema": schema, "trace": _t(state, "load_schema")}

    # ---------------------------------------------------------------- draft and check
    @timed("write_query")
    def write_query(state: SqlState) -> dict[str, Any]:
        sql = writer.draft(state["question"], state["schema"], state.get("error"), state.get("sql") or None)
        return {"sql": sql, "trace": _t(state, f"write_query#{state['repairs'] + 1}")}

    @timed("check_query")
    def check_query(state: SqlState, runtime: Runtime[SqlContext], config: RunnableConfig) -> dict[str, Any]:
        ctx, policy = _ctx(runtime), _ctx(runtime).policy
        qid = _query_id(config)
        draft = state["sql"]
        if draft == NO_QUERY:
            return {"kind": "none", "trace": _t(state, "check(none)")}

        verdict = validate_sql(draft, policy.readable, [t for t in policy.writable if policy.can_write_table(t)],
                               max_chars=settings.max_sql_chars)
        sha = _sha(verdict.sql) if verdict.sql else ""
        audit.record("sql_checked", query_id=qid, user_id=ctx.user_id, verdict=verdict.kind,
                     reason=verdict.reason, sql=draft[:1000], sql_sha256=sha)
        if verdict.kind == "blocked":
            return {"kind": "blocked", "error": verdict.reason, "trace": _t(state, "check(blocked)")}

        try:
            if verdict.kind == "read":
                db.dry_run(path, policy, verdict.sql, settings)  # compiles under the role's authorizer
                return {"kind": "read", "sql": verdict.sql, "sql_sha256": sha, "error": None,
                        "trace": _t(state, "check(read)")}
            n = db.preview_write(path, policy, verdict.sql, settings)  # executes, counts, ROLLS BACK
        except sqlite3.Error as exc:
            return _classify(state, exc, verdict.sql, "check")
        if n == 0:
            return {"kind": "noop", "sql": verdict.sql, "trace": _t(state, "check(no rows)")}
        if n > settings.max_write_rows:
            return {"kind": "blocked", "sql": verdict.sql,
                    "error": f"write would affect {n} rows (limit {settings.max_write_rows})",
                    "trace": _t(state, "check(too many rows)")}
        # Written here, BEFORE the interrupt, in a node that runs exactly once. (Code before interrupt() in
        # the approval node would run again on resume.)
        audit.record("approval_requested", query_id=qid, user_id=ctx.user_id, sql=verdict.sql,
                     sql_sha256=sha, rows_affected_preview=n)
        return {"kind": "write", "sql": verdict.sql, "sql_sha256": sha, "affected": n, "error": None,
                "preview_at": clock(), "trace": _t(state, f"check(write:{n})")}

    def _classify(state: SqlState, exc: sqlite3.Error, sql: str, where: str) -> dict[str, Any]:
        """Not every database error means 'ask the model to try again'."""
        if db.is_denied(exc):  # the database's permission layer said no: retrying cannot help
            return {"kind": "blocked", "sql": sql, "trace": _t(state, f"{where}(denied)"),
                    "error": "that statement touches data your role is not allowed to access"}
        if db.is_timeout(exc):  # a slow query will be slow again
            return {"kind": "timeout", "sql": sql, "trace": _t(state, f"{where}(timeout)"),
                    "error": f"the statement exceeded {settings.statement_timeout_s:g}s"}
        if isinstance(exc, sqlite3.DataError):  # SQLITE_TOOBIG: a value exceeded the size limit; same query, same result
            return {"kind": "blocked", "sql": sql, "trace": _t(state, f"{where}(too big)"),
                    "error": f"a value in that query exceeded the size limit ({exc})"}
        if isinstance(exc, sqlite3.IntegrityError):  # constraint failures are about the DATA, not the SQL
            return {"kind": "blocked", "sql": sql, "trace": _t(state, f"{where}(constraint)"),
                    "error": f"the change violates a database constraint ({exc})"}
        return {"kind": "error", "sql": sql, "error": str(exc), "trace": _t(state, f"{where}(error)")}

    def after_check(state: SqlState) -> Literal["bump", "refuse", "out_of_scope", "run_query", "approve_write",
                                               "fail", "timed_out", "no_change"]:
        kind = state["kind"]
        if kind == "error":
            return "bump" if state["repairs"] < settings.max_repairs else "fail"
        return {"none": "out_of_scope", "blocked": "refuse", "timeout": "timed_out", "noop": "no_change",
                "read": "run_query", "write": "approve_write"}[kind]

    @timed("bump")
    def bump(state: SqlState) -> dict[str, Any]:
        return {"repairs": state["repairs"] + 1}

    # ---------------------------------------------------------------- read path
    @timed("run_query")
    def run_query(state: SqlState, runtime: Runtime[SqlContext]
                  ) -> Command[Literal["answer", "write_query", "refuse", "fail", "timed_out"]]:
        try:
            result = db.run_read(path, _ctx(runtime).policy, state["sql"], settings)
        except sqlite3.Error as exc:
            upd = _classify(state, exc, state["sql"], "run")
            if upd["kind"] == "blocked":
                return Command(update=upd, goto="refuse")  # ty: ignore[invalid-explicit-return]
            if upd["kind"] == "timeout":
                return Command(update=upd, goto="timed_out")  # ty: ignore[invalid-explicit-return]
            if state["repairs"] < settings.max_repairs:
                return Command(update={**upd, "repairs": state["repairs"] + 1}, goto="write_query")  # ty: ignore[invalid-explicit-return]
            return Command(update=upd, goto="fail")  # ty: ignore[invalid-explicit-return]
        return Command(update={"columns": result.columns, "rows": result.rows, "truncated": result.truncated,
                               "trace": _t(state, f"run({len(result.rows)}{'+' if result.truncated else ''})")},
                       goto="answer")  # ty: ignore[invalid-explicit-return]

    @timed("answer")
    def answer(state: SqlState) -> dict[str, Any]:
        text = writer.answer(state["question"], state["sql"], state["columns"],
                             [tuple(r) for r in state["rows"]], state["truncated"])
        if state["truncated"]:  # deterministic: never depend on the model to mention it
            text += f"\n(Showing the first {settings.max_rows} rows; there are more.)"
        return {"status": "answered", "answer": text, "trace": _t(state, "answer")}

    # ---------------------------------------------------------------- write path
    @timed("approve_write")
    def approve_write(state: SqlState, runtime: Runtime[SqlContext], config: RunnableConfig
                      ) -> Command[Literal["apply_write", "cancelled"]]:
        decision = interrupt({
            "type": "approve_sql_write",
            "query_id": state["query_id"],
            "requested_by": state["requested_by"],
            "sql": state["sql"],
            "sql_sha256": state["sql_sha256"],
            "rows_affected_preview": state["affected"],
            "expires_at": state["preview_at"] + settings.approval_ttl_s,
            "question": "Run this statement against the database?",
        })
        # --- everything below runs ONCE, after the human answered. `runtime.context` is the identity of the
        # caller who RESUMED the run, i.e. the approver, not the person who asked.
        approver = _ctx(runtime)
        wants = isinstance(decision, dict) and decision.get("approve") is True
        shown = decision.get("sql_sha256") if isinstance(decision, dict) else None
        if not wants:
            outcome, reason = "rejected", "the request was rejected"
        elif not approver.policy.can_approve:
            outcome, reason = "denied", f"{approver.user_id} is not allowed to approve writes"
        elif approver.user_id == state["requested_by"]:
            outcome, reason = "denied", "you cannot approve your own request"
        elif clock() - state["preview_at"] > settings.approval_ttl_s:
            outcome, reason = "expired", "the approval window has passed; ask again"
        elif shown != state["sql_sha256"]:
            outcome, reason = "denied", "the approval does not match the statement being run"
        else:
            outcome, reason = "approved", ""
        audit.record("approval_decided", query_id=state["query_id"], user_id=approver.user_id, outcome=outcome,
                     reason=reason, requested_by=state["requested_by"], sql_sha256=state["sql_sha256"])
        if outcome != "approved":
            return Command(update={"error": reason, "trace": _t(state, f"approve({outcome})")},
                           goto="cancelled")  # ty: ignore[invalid-explicit-return]
        return Command(update={"approved_by": approver.user_id, "trace": _t(state, "approve(ok)")},
                       goto="apply_write")  # ty: ignore[invalid-explicit-return]

    @timed("apply_write")
    def apply_write(state: SqlState) -> dict[str, Any]:
        # Approval must never WIDEN privilege: the write runs under the REQUESTER's role, the one the
        # statement was validated and previewed under, not the (more powerful) approver's.
        policy = policy_for(state["requested_role"])
        try:
            n = db.apply_write(path, policy, state["sql"], state["affected"], settings)
        except db.WriteDrift as drift:
            return {"status": "stale", "error": str(drift), "trace": _t(state, "apply(stale)"),
                    "answer": f"The data changed after the preview ({drift}). Nothing was changed; please ask again."}
        except sqlite3.Error as exc:
            return {"status": "failed", "error": str(exc), "trace": _t(state, "apply(error)"),
                    "answer": f"The change could not be applied and nothing was changed: {exc}"}
        return {"status": "applied", "affected": n, "answer": f"Done. {n} row(s) changed.",
                "trace": _t(state, "apply(ok)")}

    # ---------------------------------------------------------------- terminals
    def _terminal(status: str, text: Callable[[SqlState], str]):
        def node(state: SqlState) -> dict[str, Any]:
            return {"status": status, "answer": text(state), "trace": _t(state, status)}

        return timed(status)(node)

    reject = _terminal("rejected", lambda s: s["answer"])
    refuse = _terminal("refused", lambda s: f"I can't run that query: {s['error']}. Nothing was executed.")
    out_of_scope = _terminal("out_of_scope", lambda s: "I can only answer questions about: "
                                                       f"{', '.join(s['tables'])}.")
    fail = _terminal("failed", lambda s: f"I couldn't produce a working query after {settings.max_repairs} "
                                         f"repairs: {s['error']}")
    timed_out = _terminal("timeout", lambda s: f"That query took too long and was stopped ({s['error']}). "
                                               "Try a narrower question.")
    no_change = _terminal("no_change", lambda s: "No rows match, so there is nothing to change.")
    cancelled = _terminal("cancelled", lambda s: f"Cancelled ({s['error'] or 'not approved'}). Nothing was changed.")

    @timed("record")
    def record(state: SqlState, config: RunnableConfig) -> dict[str, Any]:
        _audit("finished", state, _query_id(config), status=state["status"], repairs=state["repairs"],
               rows=len(state["rows"]), truncated=state["truncated"], affected=state["affected"],
               approved_by=state["approved_by"] or None, error=state["error"])
        return {"trace": _t(state, "record")}

    # ---------------------------------------------------------------- wiring
    b = StateGraph(SqlState, input_schema=SqlInput, output_schema=SqlOutput, context_schema=SqlContext)  # ty: ignore[invalid-argument-type]
    b.add_node("guard_input", guard_input, destinations=("load_schema", "reject"))  # ty: ignore[invalid-argument-type]
    b.add_node("load_schema", load_schema)
    b.add_node("write_query", write_query, retry_policy=RETRY)
    b.add_node("check_query", check_query)  # ty: ignore[invalid-argument-type]
    b.add_node("bump", bump)
    b.add_node("run_query", run_query, destinations=("answer", "write_query", "refuse", "fail", "timed_out"))
    b.add_node("answer", answer, retry_policy=RETRY)
    b.add_node("approve_write", approve_write, destinations=("apply_write", "cancelled"))  # ty: ignore[invalid-argument-type]
    b.add_node("apply_write", apply_write)
    for name, fn in [("reject", reject), ("refuse", refuse), ("out_of_scope", out_of_scope), ("fail", fail),
                     ("timed_out", timed_out), ("no_change", no_change), ("cancelled", cancelled)]:
        b.add_node(name, fn)
    b.add_node("record", record)

    b.add_edge(START, "guard_input")
    b.add_edge("load_schema", "write_query")
    b.add_edge("write_query", "check_query")
    b.add_conditional_edges("check_query", after_check, ["bump", "refuse", "out_of_scope", "run_query",
                                                         "approve_write", "fail", "timed_out", "no_change"])
    b.add_edge("bump", "write_query")
    for terminal in ("reject", "refuse", "out_of_scope", "fail", "timed_out", "no_change", "cancelled", "answer",
                     "apply_write"):
        b.add_edge(terminal, "record")
    b.add_edge("record", END)
    return b.compile(checkpointer=checkpointer)
