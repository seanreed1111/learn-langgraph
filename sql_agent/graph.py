"""The SQL analyst graph.

    START -> list_tables -> get_schema -> write_query -> check_query
                                              ^              |
                                              |   blocked    +--> refuse -> END
                                  (error, <= MAX_REPAIRS)    |
                                              +--------------+--> read  -> run_query -> answer -> END
                                                             |
                                                             +--> write -> approve_write -> apply_write -> END
                                                                                |
                                                                                +--> cancelled -> END

Why a fixed workflow and not a free ReAct tool loop (what the official tutorial shows)?
  * The steps are always the same, so a graph is cheaper, faster and easier to test.
  * The dangerous decision (run this SQL?) is a node you control, not something the model can skip.
  * The model only does the two things it is good at: draft SQL, phrase the answer.
A tool loop is the right call when the *set of steps* is unknown (research, debugging).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from sql_agent import db
from sql_agent.guard import validate_sql
from sql_agent.offline import NO_QUERY, OfflineWriter

MAX_REPAIRS = 2          # bounded self-repair: draft, then at most 2 fixes
MAX_WRITE_ROWS = 5       # a write that would touch more rows than this is refused


class SQLState(TypedDict, total=False):
    question: str
    tables: list[str]
    schema: dict[str, str]
    sql: str
    kind: str                # read | write | blocked
    error: str | None        # last dry-run / execution error, fed back to the writer
    repairs: int
    columns: list[str]
    rows: list[list[Any]]
    affected: int
    status: str              # answered | refused | out_of_scope | failed | applied | cancelled
    answer: str


_DEFAULT = object()


def build_graph(db_path: str | Path, writer=None, checkpointer: Any = _DEFAULT):
    """checkpointer: default = in-memory; pass None for Studio/platform (it supplies its own)."""
    writer = writer or OfflineWriter()
    db_path = Path(db_path)
    ro = db.connect_readonly(db_path)

    # ---------------------------------------------------------------- nodes
    def list_tables(state: SQLState) -> dict[str, Any]:
        schema = db.schema_text(db_path)  # trusted path: not model-controlled
        return {"tables": sorted(schema), "schema": schema, "repairs": 0, "error": None}

    def get_schema(state: SQLState) -> dict[str, Any]:
        return {}  # kept as its own node so a real deployment can filter columns / sample rows here

    def write_query(state: SQLState) -> dict[str, Any]:
        sql = writer.draft(state["question"], state["schema"], state.get("error"), state.get("sql"))
        return {"sql": sql}

    def check_query(state: SQLState) -> dict[str, Any]:
        sql = state["sql"]
        if sql == NO_QUERY:
            return {"kind": "none"}
        verdict = validate_sql(sql, set(state["tables"]))
        if verdict.kind == "blocked":
            return {"kind": "blocked", "error": verdict.reason, "sql": sql}
        try:
            if verdict.kind == "read":
                db.dry_run(ro, verdict.sql)  # catches bad columns BEFORE running
            else:
                n = db.preview_write(db_path, verdict.sql)  # executes, counts, ROLLS BACK
                if n > MAX_WRITE_ROWS:
                    return {"kind": "blocked", "sql": verdict.sql,
                            "error": f"write would affect {n} rows (limit {MAX_WRITE_ROWS})"}
                return {"kind": "write", "sql": verdict.sql, "affected": n, "error": None}
        except sqlite3.Error as e:
            return {"kind": "error", "sql": verdict.sql, "error": str(e)}
        return {"kind": "read", "sql": verdict.sql, "error": None}

    def route_check(state: SQLState) -> Literal["refuse", "out_of_scope", "run_query",
                                                "approve_write", "fail"]:
        k = state["kind"]
        if k == "none":
            return "out_of_scope"
        if k == "blocked":
            return "refuse"
        if k == "error":
            return "fail"  # (errors under the repair budget are routed to `bump` by the caller)
        return "run_query" if k == "read" else "approve_write"

    def bump(state: SQLState) -> dict[str, Any]:
        return {"repairs": state["repairs"] + 1}

    def run_query(state: SQLState) -> Command[Literal["answer", "write_query", "fail"]]:
        try:
            cols, rows = db.run_read(ro, state["sql"])
        except sqlite3.Error as e:
            if state["repairs"] < MAX_REPAIRS:
                return Command(goto="write_query", update={"error": str(e),  # ty: ignore[invalid-explicit-return]
                                                           "repairs": state["repairs"] + 1})
            return Command(goto="fail", update={"error": str(e)})  # ty: ignore[invalid-explicit-return]
        return Command(goto="answer", update={"columns": cols, "rows": [list(r) for r in rows]})  # ty: ignore[invalid-explicit-return]

    def answer(state: SQLState) -> dict[str, Any]:
        text = writer.answer(state["question"], state["sql"], state["columns"],
                             [tuple(r) for r in state["rows"]])
        return {"status": "answered", "answer": text}

    def approve_write(state: SQLState) -> Command[Literal["apply_write", "cancelled"]]:
        decision = interrupt({
            "type": "approve_sql_write",
            "sql": state["sql"],
            "rows_affected_preview": state["affected"],
            "question": "Run this statement against the database?",
        })
        approved = decision is True or (isinstance(decision, dict) and decision.get("approve") is True)
        return Command(goto="apply_write" if approved else "cancelled")  # ty: ignore[invalid-explicit-return]

    def apply_write_node(state: SQLState) -> dict[str, Any]:
        n = db.apply_write(db_path, state["sql"])  # the ONLY node holding a write connection
        return {"status": "applied", "affected": n, "answer": f"Done. {n} row(s) changed."}

    def cancelled(state: SQLState) -> dict[str, Any]:
        return {"status": "cancelled", "answer": "Cancelled. Nothing was changed."}

    def refuse(state: SQLState) -> dict[str, Any]:
        return {"status": "refused",
                "answer": f"I can't run that query: {state['error']}. Nothing was executed."}

    def out_of_scope(state: SQLState) -> dict[str, Any]:
        return {"status": "out_of_scope",
                "answer": "I can only answer questions about the shop database "
                          f"({', '.join(state['tables'])})."}

    def fail(state: SQLState) -> dict[str, Any]:
        return {"status": "failed",
                "answer": f"I couldn't produce a working query after {MAX_REPAIRS} repairs: {state['error']}"}

    # ---------------------------------------------------------------- wiring
    g = StateGraph(SQLState)  # ty: ignore[invalid-argument-type]
    g.add_node("list_tables", list_tables)
    g.add_node("get_schema", get_schema)
    g.add_node("write_query", write_query)
    g.add_node("check_query", check_query)
    g.add_node("bump", bump)
    g.add_node("run_query", run_query)
    g.add_node("answer", answer)
    g.add_node("approve_write", approve_write)
    g.add_node("apply_write", apply_write_node)
    for n, f in [("cancelled", cancelled), ("refuse", refuse),
                 ("out_of_scope", out_of_scope), ("fail", fail)]:
        g.add_node(n, f)

    g.add_edge(START, "list_tables")
    g.add_edge("list_tables", "get_schema")
    g.add_edge("get_schema", "write_query")
    g.add_edge("write_query", "check_query")
    # check errors count as a repair attempt; route through `bump` so the counter is in state
    g.add_conditional_edges(
        "check_query",
        lambda s: "bump" if s["kind"] == "error" and s["repairs"] < MAX_REPAIRS else route_check(s),
        ["bump", "refuse", "out_of_scope", "run_query", "approve_write", "fail"],
    )
    g.add_edge("bump", "write_query")
    g.add_edge("answer", END)
    g.add_edge("apply_write", END)
    for n in ("cancelled", "refuse", "out_of_scope", "fail"):
        g.add_edge(n, END)
    return g.compile(checkpointer=InMemorySaver() if checkpointer is _DEFAULT else checkpointer)
