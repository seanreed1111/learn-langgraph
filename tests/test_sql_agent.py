"""Lesson 20 tests: guard, DB-level read-only, approval flow, repair loop, eval gate."""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from langgraph.types import Command  # noqa: E402

from sql_agent import db  # noqa: E402
from sql_agent.evals import run_evals  # noqa: E402
from sql_agent.graph import build_graph  # noqa: E402
from sql_agent.guard import MAX_ROWS, validate_sql  # noqa: E402

TABLES = {"customers", "orders", "products", "order_items"}


@pytest.fixture()
def path(tmp_path):
    return db.create_demo_db(tmp_path / "t.sqlite")


def cfg(t: str) -> dict:
    return {"configurable": {"thread_id": t}}


@pytest.mark.parametrize("sql,reason", [
    ("DROP TABLE customers", "DROP"),
    ("SELECT 1; DELETE FROM orders WHERE id=1", "multiple"),
    ("SELECT * FROM customers -- x", "comments"),
    ("SELECT /* x */ 1", "comments"),
    ("PRAGMA table_info(customers)", "PRAGMA"),
    ("ATTACH DATABASE 'x' AS y", "ATTACH"),
    ("SELECT * FROM sqlite_master", "table not allowed"),
    ("DELETE FROM orders", "WHERE"),
    ("INSERT OR REPLACE INTO orders VALUES (1,1,1,'x')", "REPLACE"),
    ("", "empty"),
    ("EXPLAIN SELECT 1", "not allowed"),
])
def test_guard_blocks(sql, reason):
    v = validate_sql(sql, TABLES)
    assert v.kind == "blocked" and reason.lower() in v.reason.lower()


def test_guard_semicolon_inside_literal_is_fine():
    v = validate_sql("SELECT name FROM customers WHERE name = 'a;b'", TABLES)
    assert v.kind == "read"


def test_guard_keyword_inside_literal_is_fine():
    assert validate_sql("SELECT name FROM customers WHERE name = 'drop'", TABLES).kind == "read"


def test_guard_replace_function_allowed_and_limit_added():
    v = validate_sql("SELECT replace(name,'a','b') FROM customers;", TABLES)
    assert v.kind == "read" and v.sql.endswith(f"LIMIT {MAX_ROWS}") and ";" not in v.sql


def test_guard_keeps_existing_limit():
    assert validate_sql("SELECT * FROM customers LIMIT 3", TABLES).sql.endswith("LIMIT 3")


def test_guard_cte_write_is_write_not_read():
    assert validate_sql("WITH x AS (SELECT 1) DELETE FROM orders WHERE id=1", TABLES).kind == "write"


def test_guard_cte_names_are_not_unknown_tables():
    assert validate_sql("WITH x AS (SELECT * FROM orders) SELECT * FROM x", TABLES).kind == "read"


def test_database_itself_refuses_writes_even_if_guard_is_bypassed(path):
    ro = db.connect_readonly(path)
    for sql in ("DELETE FROM orders", "DROP TABLE customers", "PRAGMA writable_schema=1",
                "ATTACH DATABASE ':memory:' AS x", "UPDATE orders SET status='x'"):
        with pytest.raises(sqlite3.Error):
            ro.execute(sql)
    assert len(db.run_read(ro, "SELECT * FROM orders")[1]) == 8


def test_read_answer(path):
    out = build_graph(path).invoke({"question": "How many customers do we have?"}, cfg("a"))
    assert out["status"] == "answered" and out["rows"] == [[6]]


def test_repair_loop_fixes_bad_first_draft(path):
    out = build_graph(path).invoke({"question": "total revenue by product category"}, cfg("b"))
    assert out["status"] == "answered" and out["repairs"] == 1
    assert dict(map(tuple, out["rows"]))["displays"] == 1680.0


def test_repair_budget_is_bounded(path):
    class Stubborn:
        def draft(self, *a, **k):
            return "SELECT nonexistent FROM customers"

        def answer(self, *a, **k):
            return ""

    out = build_graph(path, Stubborn()).invoke({"question": "x"}, cfg("c"))
    assert out["status"] == "failed" and out["repairs"] == 2


def test_out_of_scope(path):
    out = build_graph(path).invoke({"question": "what is the weather"}, cfg("d"))
    assert out["status"] == "out_of_scope"


def test_write_needs_approval_and_preview_does_not_persist(path):
    g = build_graph(path)
    paused = g.invoke({"question": "Cancel order 6"}, cfg("e"))
    intr = paused["__interrupt__"][0].value
    assert intr["rows_affected_preview"] == 1
    assert db.run_read(db.connect_readonly(path), "SELECT status FROM orders WHERE id=6")[1] == [("pending",)]
    out = g.invoke(Command(resume=True), cfg("e"))
    assert out["status"] == "applied"
    assert db.run_read(db.connect_readonly(path), "SELECT status FROM orders WHERE id=6")[1] == [("cancelled",)]


def test_write_rejected_changes_nothing(path):
    g = build_graph(path)
    g.invoke({"question": "Delete customer 3"}, cfg("f"))
    out = g.invoke(Command(resume=False), cfg("f"))
    assert out["status"] == "cancelled"
    assert db.run_read(db.connect_readonly(path), "SELECT COUNT(*) FROM customers")[1] == [(6,)]


def test_write_touching_too_many_rows_is_blocked(path):
    class Greedy:
        def draft(self, *a, **k):
            return "UPDATE orders SET status='x' WHERE order_year >= 2000"

        def answer(self, *a, **k):
            return ""

    out = build_graph(path, Greedy()).invoke({"question": "x"}, cfg("g"))
    assert out["status"] == "refused" and "rows" in out["answer"]


def test_eval_gate_passes():
    results = run_evals.run()
    assert run_evals.gate(results) == []
