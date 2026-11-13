"""Lesson 24 tests: parser guard, DB-enforced policy and limits, approval rules, audit trail, API, eval gate.

The pyramid, as in lesson 14:
  1. pure functions   (policy, guard)                      - fast, exhaustive
  2. database layer   (authorizer, timeout, caps, drift)   - the layer that must hold when the guard is wrong
  3. graph logic      (stub writers, fake clock)           - budgets, error routing, approval rules
  4. audit + API      (TestClient)                         - identity, ownership, rate limits, restart
  5. evals            (labelled cases + gates)             - regression net
"""
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any, override

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sql_prod import db  # noqa: E402
from sql_prod.api import RateLimiter, create_app  # noqa: E402
from sql_prod.app import USERS, build_app  # noqa: E402
from sql_prod.audit import AuditLog  # noqa: E402
from sql_prod.config import Settings  # noqa: E402
from sql_prod.evals import run_evals  # noqa: E402
from sql_prod.graph import SqlOutput  # noqa: E402
from sql_prod.guard import validate_sql  # noqa: E402
from sql_prod.policy import ANONYMOUS, ROLES, SqlContext, policy_for  # noqa: E402

READABLE = {"customers", "orders", "products", "order_items"}
WRITABLE = {"orders"}
ANALYST, SUPPORT, MANAGER, VIEWER = (policy_for(r) for r in ("analyst", "support", "manager", "viewer"))


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), db_path=tmp_path / "shop.sqlite", audit_db=tmp_path / "audit.sqlite",
                   checkpoint_db=str(tmp_path / "ck.sqlite"), statement_timeout_s=0.3)


@pytest.fixture
def parts(settings):
    db.create_demo_db(settings.db_path)
    return (*build_app(settings), settings)  # graph, audit, settings


def cfg(thread: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread}}


def ask(graph, q: str, user: str = "anna", thread: str = "t") -> dict[str, Any]:
    return graph.invoke({"question": q}, cfg(thread), context=USERS[user])


def decide(graph, thread: str, user: str, *, approve: bool = True, sha: str) -> dict[str, Any]:
    return graph.invoke(Command(resume={"approve": approve, "sql_sha256": sha}), cfg(thread), context=USERS[user])


def pending(out: dict[str, Any]) -> dict[str, Any]:
    return out["__interrupt__"][0].value


def value(settings: Settings, sql: str) -> Any:
    with closing(sqlite3.connect(settings.db_path)) as con:
        return con.execute(sql).fetchone()[0]


class Stub:
    """A writer that returns scripted SQL and counts calls."""

    def __init__(self, *sqls: str) -> None:
        self.sqls, self.calls, self.schemas = list(sqls), 0, []

    def draft(self, question, schema, error=None, prev_sql=None):
        self.schemas.append(schema)
        self.calls += 1
        return self.sqls[min(self.calls - 1, len(self.sqls) - 1)]

    def answer(self, *a, **k):
        return "ok"


# =============================================================== 1. policy
def test_unknown_role_gets_deny_all():
    assert policy_for("intern").readable == {} and policy_for("intern").writable == frozenset()
    assert ANONYMOUS.policy.readable == {}


def test_every_role_may_only_write_tables_it_can_read():
    for name, p in ROLES.items():
        assert all(p.can_write_table(t) for t in p.writable), name


def test_policy_matches_the_real_schema(settings):
    """A new column or table must be a conscious policy decision: manager sees exactly what exists."""
    db.create_demo_db(settings.db_path)
    with closing(sqlite3.connect(settings.db_path)) as con:
        real = {t: {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
                for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {t: set(c) for t, c in MANAGER.readable.items()} == real


# =============================================================== 1. guard
@pytest.mark.parametrize("sql,reason", [
    ("DROP TABLE customers", "DROP"),
    ("SELECT 1; DELETE FROM orders WHERE id=1", "multiple"),
    ("SELECT * FROM customers -- x", "comments"),
    ("SELECT /* x */ 1", "comments"),
    ("PRAGMA table_info(customers)", "PRAGMA"),
    ("ATTACH DATABASE 'x' AS y", "ATTACH"),
    ("EXPLAIN SELECT 1", "not allowed"),
    ("VACUUM", "not allowed"),
    ("SELEC nonsense", "not allowed"),
    ("SELECT * FROM sqlite_master", "table not allowed"),
    # --- the bypasses that fooled lesson 20's regex guard
    ("SELECT * FROM customers, sqlite_master", "table not allowed"),
    ("SELECT * FROM customers JOIN sqlite_master ON 1=1", "table not allowed"),
    ("SELECT * FROM (SELECT * FROM sqlite_master)", "table not allowed"),
    ("SELECT name FROM customers UNION SELECT name FROM sqlite_master", "table not allowed"),
    ("WITH x AS (SELECT * FROM sqlite_master) SELECT * FROM x", "table not allowed"),
    ("SELECT * FROM main.sqlite_master", "schema-qualified"),
    ("SELECT * FROM temp.customers", "schema-qualified"),
    ("SELECT * FROM pragma_table_info('customers')", "table-valued"),
    ("SELECT load_extension('x')", "forbidden function"),
    ("SELECT * FROM customers WHERE id IN (SELECT id FROM orders); SELECT 1", "multiple"),
    # --- writes
    ("DELETE FROM orders", "WHERE"),
    ("UPDATE orders SET status = 'x'", "WHERE"),
    ("UPDATE customers SET name = 'x' WHERE id = 1", "may not modify"),
    ("INSERT INTO customers (name) VALUES ('x')", "may not modify"),
    ("SELECT 1 FROM orders WHERE id IN (DELETE FROM orders WHERE id = 1)", "could not parse"),
    ("", "empty"),
])
def test_guard_blocks(sql, reason):
    v = validate_sql(sql, READABLE, WRITABLE)
    assert v.kind == "blocked" and v.sql == "" and reason.lower() in v.reason.lower(), v


def test_guard_blocks_oversized_statements():
    assert validate_sql("SELECT " + "1," * 5000 + "1", READABLE, max_chars=100).kind == "blocked"


@pytest.mark.parametrize("sql,kind", [
    ("SELECT name FROM customers WHERE name = 'a;b'", "read"),         # ';' inside a literal is data
    ("SELECT name FROM customers WHERE name = 'drop table x'", "read"),  # so is a keyword
    ("SELECT name FROM customers WHERE name = 'a--b'", "read"),
    ("SELECT replace(name, 'a', 'b') FROM customers;", "read"),         # replace() the FUNCTION is fine
    ("WITH x AS (SELECT * FROM orders) SELECT * FROM x", "read"),
    ("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n < 3) SELECT n FROM r", "read"),
    ("SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id", "read"),
    ("SELECT name FROM customers UNION SELECT name FROM products", "read"),
    ("UPDATE orders SET status = 'cancelled' WHERE id = 6", "write"),
    ("DELETE FROM orders WHERE id = 6", "write"),
])
def test_guard_allows(sql, kind):
    assert validate_sql(sql, READABLE, WRITABLE).kind == kind


def test_guard_returns_the_reserialised_statement_not_the_original():
    v = validate_sql("select   name  from customers ;", READABLE)
    assert v.kind == "read" and ";" not in v.sql and v.sql.upper().startswith("SELECT NAME")


def test_reserialised_reads_return_the_same_rows(settings):
    """Running the guard's rewrite must not change what a query means."""
    db.create_demo_db(settings.db_path)
    for sql in ["SELECT c.name, COUNT(o.id) AS n FROM customers c LEFT JOIN orders o ON o.customer_id = c.id GROUP BY c.id ORDER BY c.name",
                "SELECT category, SUM(price) FROM products GROUP BY category ORDER BY 2 DESC",
                "SELECT replace(name, ' ', '_') FROM customers WHERE country IN ('UK', 'USA') ORDER BY id",
                "WITH t AS (SELECT customer_id, COUNT(*) AS n FROM orders GROUP BY 1) SELECT * FROM t WHERE n > 1 ORDER BY 1"]:
        v = validate_sql(sql, READABLE)
        with closing(sqlite3.connect(settings.db_path)) as con:
            assert con.execute(v.sql).fetchall() == con.execute(sql).fetchall(), sql


# =============================================================== 2. database layer
def read(settings, policy, sql):
    return db.run_read(settings.db_path, policy, sql, settings)


def test_authorizer_enforces_columns_even_if_the_guard_is_bypassed(settings):
    db.create_demo_db(settings.db_path)
    assert read(settings, SUPPORT, "SELECT email FROM customers").rows[0][0].endswith("@example.com")
    for sql in ("SELECT email FROM customers", "SELECT * FROM customers",
                "SELECT c.email FROM customers c", "SELECT name FROM customers WHERE email LIKE '%a%'"):
        with pytest.raises(sqlite3.Error) as err:
            read(settings, ANALYST, sql)
        assert db.is_denied(err.value), sql


def test_authorizer_hides_the_catalogue_and_unlisted_tables(settings):
    db.create_demo_db(settings.db_path)
    for policy in (MANAGER, ANALYST):
        for sql in ("SELECT * FROM sqlite_master", "SELECT * FROM customers, sqlite_master",
                    "SELECT * FROM sqlite_schema", "SELECT * FROM sqlite_temp_master"):
            with pytest.raises(sqlite3.Error) as err:
                read(settings, policy, sql)
            assert db.is_denied(err.value), sql
    with pytest.raises(sqlite3.Error) as err:
        read(settings, VIEWER, "SELECT name FROM customers")
    assert db.is_denied(err.value)


def test_authorizer_still_allows_normal_reads(settings):
    db.create_demo_db(settings.db_path)
    assert read(settings, ANALYST, "SELECT COUNT(*) FROM customers").rows == [[6]]
    assert read(settings, ANALYST, "SELECT rowid FROM customers LIMIT 1").rows == [[1]]
    cte = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n < 5) SELECT SUM(n) FROM r"
    assert read(settings, ANALYST, cte).rows == [[15]]


def test_read_connection_refuses_every_write_and_ddl(settings):
    db.create_demo_db(settings.db_path)
    for sql in ("DELETE FROM orders WHERE id=1", "DROP TABLE customers", "PRAGMA writable_schema=1",
                "ATTACH DATABASE ':memory:' AS x", "UPDATE orders SET status='x' WHERE id=1",
                "INSERT INTO orders VALUES (99,1,2024,'x')", "CREATE TABLE evil (x)", "VACUUM"):
        with pytest.raises(sqlite3.Error):
            read(settings, MANAGER, sql)
    assert value(settings, "SELECT COUNT(*) FROM orders") == 8


def test_schema_text_shows_only_what_the_role_may_read(settings):
    db.create_demo_db(settings.db_path)
    analyst = db.schema_text(settings.db_path, ANALYST)
    assert set(analyst) == READABLE and "email" not in analyst["customers"]
    assert "FOREIGN KEY (customer_id) REFERENCES customers(id)" in analyst["orders"]
    assert "email" in db.schema_text(settings.db_path, SUPPORT)["customers"]
    assert set(db.schema_text(settings.db_path, VIEWER)) == {"products"}
    assert db.schema_text(settings.db_path, policy_for("intern")) == {}


def test_statement_timeout_stops_runaway_queries(settings):
    db.create_demo_db(settings.db_path)
    forever = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT COUNT(*) FROM r"
    with pytest.raises(sqlite3.Error) as err:
        read(settings, ANALYST, forever)
    assert db.is_timeout(err.value)


def test_value_size_limit_stops_memory_bombs(settings):
    db.create_demo_db(settings.db_path)
    with pytest.raises(sqlite3.DataError):
        read(settings, ANALYST, "SELECT zeroblob(200000000)")


def test_row_cap_is_applied_when_rows_leave_the_database(settings):
    db.create_demo_db(settings.db_path)
    sql = "SELECT id FROM orders ORDER BY id"
    capped = db.run_read(settings.db_path, ANALYST, sql, replace(settings, max_rows=3))
    assert len(capped.rows) == 3 and capped.truncated
    exact = db.run_read(settings.db_path, ANALYST, sql, replace(settings, max_rows=8))
    assert len(exact.rows) == 8 and not exact.truncated  # boundary: exactly max_rows is NOT truncated
    # a LIMIT hidden in a subquery used to stop lesson 20 adding an outer one; the cap does not care
    sneaky = "SELECT id FROM orders WHERE id IN (SELECT id FROM orders LIMIT 100) ORDER BY id"
    assert db.run_read(settings.db_path, ANALYST, sneaky, replace(settings, max_rows=2)).truncated


def test_preview_counts_rows_and_rolls_back(settings):
    db.create_demo_db(settings.db_path)
    sql = "UPDATE orders SET status = 'cancelled' WHERE order_year = 2024"
    assert db.preview_write(settings.db_path, SUPPORT, sql, settings) == 6
    assert value(settings, "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'") == 1  # only the original one


def test_write_connection_is_bounded_by_the_policy_too(settings):
    db.create_demo_db(settings.db_path)
    with pytest.raises(sqlite3.Error) as err:
        db.preview_write(settings.db_path, SUPPORT, "UPDATE customers SET name='x' WHERE id=1", settings)
    assert db.is_denied(err.value)
    for ddl in ("DROP TABLE orders", "CREATE TABLE evil (x)", "ALTER TABLE orders ADD COLUMN x"):
        with pytest.raises(sqlite3.Error):
            db.preview_write(settings.db_path, MANAGER, ddl, settings)


def test_foreign_keys_are_enforced_so_previews_catch_constraint_failures(settings):
    db.create_demo_db(settings.db_path)
    with pytest.raises(sqlite3.IntegrityError):
        db.preview_write(settings.db_path, MANAGER, "DELETE FROM customers WHERE id = 3", settings)


def test_apply_write_rolls_back_when_the_row_count_drifted(settings):
    db.create_demo_db(settings.db_path)
    sql = "UPDATE orders SET status = 'cancelled' WHERE status = 'pending'"
    assert db.preview_write(settings.db_path, SUPPORT, sql, settings) == 1
    with closing(sqlite3.connect(settings.db_path)) as con, con:
        con.execute("INSERT INTO orders VALUES (99, 1, 2024, 'pending')")
    with pytest.raises(db.WriteDrift) as drift:
        db.apply_write(settings.db_path, SUPPORT, sql, 1, settings)
    assert (drift.value.expected, drift.value.actual) == (1, 2)
    assert value(settings, "SELECT COUNT(*) FROM orders WHERE status = 'pending'") == 2  # nothing was changed
    assert db.apply_write(settings.db_path, SUPPORT, sql, 2, settings) == 2  # and the honest count works


def test_connections_are_per_operation_so_threads_do_not_share_state(settings):
    db.create_demo_db(settings.db_path)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: read(settings, ANALYST, "SELECT COUNT(*) FROM orders").rows, range(64)))
    assert all(r == [[8]] for r in results)


# =============================================================== 3. graph
def test_read_answer_and_trace(parts):
    graph, _, _ = parts
    out = ask(graph, "How many customers do we have?")
    assert out["status"] == "answered" and out["rows"] == [[6]]
    assert out["trace"][0] == "guard_input" and out["trace"][-1] == "record"


def test_repair_loop_fixes_a_bad_first_draft(parts):
    graph, _, _ = parts
    out = ask(graph, "total revenue by product category")
    assert out["status"] == "answered" and out["repairs"] == 1
    assert dict(map(tuple, out["rows"]))["displays"] == 1680.0


def test_repair_budget_is_bounded(settings):
    db.create_demo_db(settings.db_path)
    stub = Stub("SELECT nonexistent FROM customers")
    graph, _ = build_app(settings, writer=stub)
    out = ask(graph, "x")
    assert out["status"] == "failed" and out["repairs"] == settings.max_repairs
    assert stub.calls == settings.max_repairs + 1


@pytest.mark.parametrize("sql,status", [
    ("SELECT email FROM customers", "refused"),                                                # permission
    ("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT COUNT(*) FROM r", "timeout"),
    ("SELECT zeroblob(200000000)", "refused"),                                                  # too big
])
def test_deterministic_database_errors_are_not_retried(settings, sql, status):
    """Asking the model to try again cannot fix a permission, a timeout or a size limit."""
    db.create_demo_db(settings.db_path)
    stub = Stub(sql)
    graph, _ = build_app(settings, writer=stub)
    out = ask(graph, "x")
    assert out["status"] == status and out["repairs"] == 0 and stub.calls == 1


def test_the_model_only_sees_the_schema_its_role_may_read(settings):
    db.create_demo_db(settings.db_path)
    stub = Stub("SELECT 1 FROM products")
    graph, _ = build_app(settings, writer=stub)
    ask(graph, "x", user="anna", thread="a")
    ask(graph, "x", user="vera", thread="b")
    ask(graph, "x", user="sam", thread="c")
    analyst, viewer, support = stub.schemas
    assert "email" not in analyst["customers"] and set(viewer) == {"products"} and "email" in support["customers"]


def test_no_identity_or_unknown_role_means_no_access(parts):
    graph, _, _ = parts
    for ctx in (None, SqlContext("x", "intern")):
        out = graph.invoke({"question": "How many customers do we have?"}, cfg("n" + str(ctx)), context=ctx)
        assert out["status"] == "rejected" and out["rows"] == []


def test_input_validation(parts):
    graph, _, settings = parts
    assert ask(graph, "   ", thread="e")["status"] == "rejected"
    long = ask(graph, "x" * (settings.max_question_chars + 1), thread="l")
    assert long["status"] == "rejected" and "too long" in long["answer"]


def test_per_turn_state_resets_between_turns_on_the_same_thread(parts):
    graph, _, _ = parts
    first = ask(graph, "How many customers do we have?", thread="same")
    assert first["status"] == "answered" and first["rows"] == [[6]]
    second = ask(graph, "What is the weather?", thread="same")
    assert second["status"] == "out_of_scope" and second["rows"] == [] and second["sql"] == ""
    third = ask(graph, "total revenue by product category", thread="same")
    assert third["repairs"] == 1  # the budget was not eaten by earlier turns


def test_output_schema_is_narrow(parts):
    graph, _, _ = parts
    out = ask(graph, "How many customers do we have?")
    assert set(out) <= set(SqlOutput.__annotations__)
    assert "schema" not in out and "requested_by" not in out


def test_truncated_answers_say_so(settings):
    db.create_demo_db(settings.db_path)
    graph, _ = build_app(replace(settings, max_rows=3))
    out = ask(graph, "List all orders")
    assert out["truncated"] and len(out["rows"]) == 3 and "first 3 rows" in out["answer"]


def test_writes_that_match_nothing_need_no_approval(parts):
    graph, _, _ = parts
    out = ask(graph, "cancel order 999", user="sam")
    assert out["status"] == "no_change" and "__interrupt__" not in out


def test_writes_over_the_row_limit_and_constraint_failures_are_refused_before_approval(parts):
    graph, _, settings = parts
    big = ask(graph, "Cancel all 2024 orders", user="sam", thread="big")
    assert big["status"] == "refused" and f"limit {settings.max_write_rows}" in big["answer"]
    fk = ask(graph, "Delete customer 3", user="maya", thread="fk")
    assert fk["status"] == "refused" and "constraint" in fk["answer"]


def test_write_needs_a_different_authorised_approver(parts):
    graph, _, settings = parts
    paused = ask(graph, "Cancel order 6", user="sam", thread="w")
    req = pending(paused)
    assert req["requested_by"] == "sam" and req["rows_affected_preview"] == 1
    assert value(settings, "SELECT status FROM orders WHERE id = 6") == "pending"  # preview did not persist
    out = decide(graph, "w", "maya", sha=req["sql_sha256"])
    assert out["status"] == "applied"
    assert value(settings, "SELECT status FROM orders WHERE id = 6") == "cancelled"


@pytest.mark.parametrize("requester,approver,approve,forged_sha,why", [
    ("sam", "sam", True, False, "not allowed to approve"),   # support cannot approve at all
    ("sam", "anna", True, False, "not allowed to approve"),   # analyst neither
    ("maya", "maya", True, False, "your own request"),        # a manager cannot approve their own write
    ("sam", "maya", False, False, "rejected"),                # explicit no
    ("sam", "maya", True, True, "does not match"),            # approved a different statement than the one run
])
def test_approval_denials_change_nothing(parts, requester, approver, approve, forged_sha, why):
    graph, _, settings = parts
    req = pending(ask(graph, "Cancel order 6", user=requester, thread="d"))
    out = decide(graph, "d", approver, approve=approve, sha="0" * 64 if forged_sha else req["sql_sha256"])
    assert out["status"] == "cancelled" and why in out["answer"]
    assert value(settings, "SELECT status FROM orders WHERE id = 6") == "pending"


def test_approvals_expire(settings):
    db.create_demo_db(settings.db_path)
    now = [1_000.0]
    graph, _ = build_app(replace(settings, approval_ttl_s=60.0), clock=lambda: now[0])
    req = pending(ask(graph, "Cancel order 6", user="sam", thread="x"))
    assert req["expires_at"] == 1_060.0
    now[0] += 61
    out = decide(graph, "x", "maya", sha=req["sql_sha256"])
    assert out["status"] == "cancelled" and "window has passed" in out["answer"]
    assert value(settings, "SELECT status FROM orders WHERE id = 6") == "pending"


def test_data_that_changed_after_the_preview_is_not_written(parts):
    graph, _, settings = parts
    req = pending(ask(graph, "Cancel all pending orders", user="sam", thread="s"))
    assert req["rows_affected_preview"] == 1
    with closing(sqlite3.connect(settings.db_path)) as con, con:
        con.execute("INSERT INTO orders VALUES (99, 1, 2024, 'pending')")
    out = decide(graph, "s", "maya", sha=req["sql_sha256"])
    assert out["status"] == "stale" and "changed after the preview" in out["answer"]
    assert value(settings, "SELECT COUNT(*) FROM orders WHERE status = 'pending'") == 2


def test_approval_never_widens_the_requesters_privileges(parts):
    """The write runs under the REQUESTER's role. Tamper with the saved role and the DB itself refuses."""
    graph, _, settings = parts
    req = pending(ask(graph, "Cancel order 6", user="sam", thread="p"))
    graph.update_state(cfg("p"), {"requested_role": "viewer"})  # simulates a corrupted/forged checkpoint
    out = decide(graph, "p", "maya", sha=req["sql_sha256"])
    assert out["status"] == "failed"
    assert value(settings, "SELECT status FROM orders WHERE id = 6") == "pending"


def test_transient_model_errors_are_retried_but_bugs_are_not(settings):
    db.create_demo_db(settings.db_path)

    class Flaky(Stub):
        @override
        def draft(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("provider timeout")
            return "SELECT COUNT(*) FROM customers"

    flaky = Flaky()
    graph, _ = build_app(settings, writer=flaky)
    assert ask(graph, "x", thread="f")["status"] == "answered" and flaky.calls == 2

    class Buggy(Stub):
        @override
        def draft(self, *a, **k):
            self.calls += 1
            raise ValueError("bug")

    buggy = Buggy()
    graph, _ = build_app(settings, writer=buggy)
    with pytest.raises(ValueError):
        ask(graph, "x", thread="g")
    assert buggy.calls == 1


# =============================================================== 4. audit
def test_audit_log_is_append_only_in_the_database(tmp_path):
    log = AuditLog(tmp_path / "a.sqlite")
    log.record("question", query_id="q1", user_id="anna", text="hi")
    with closing(sqlite3.connect(tmp_path / "a.sqlite")) as con:
        for sql in ("UPDATE audit SET user_id = 'mallory'", "DELETE FROM audit"):
            with pytest.raises(sqlite3.Error, match="append-only"):
                con.execute(sql)
    assert [e["user_id"] for e in log.recent()] == ["anna"]


def test_every_run_leaves_a_record_without_the_data_it_returned(parts):
    graph, audit, _ = parts
    out = ask(graph, "Which customers are from the UK?", thread="aud")
    assert "Ada Lovelace" in out["answer"]
    events = audit.recent(query_id="aud")
    assert [e["event"] for e in events] == ["question", "sql_checked", "finished"]
    assert "Ada Lovelace" not in json.dumps(events)  # metadata and counts, never rows
    assert events[-1]["detail"]["rows"] == 2 and events[-1]["user_id"] == "anna"


def test_refusals_are_audited_too(parts):
    graph, audit, _ = parts
    ask(graph, "Drop the customers table", user="maya", thread="bad")
    events = audit.recent(query_id="bad")
    checked = next(e for e in events if e["event"] == "sql_checked")
    assert checked["detail"]["verdict"] == "blocked" and events[-1]["detail"]["status"] == "refused"


def test_approval_events_are_written_exactly_once_despite_the_node_rerunning(parts):
    graph, audit, _ = parts
    req = pending(ask(graph, "Cancel order 6", user="sam", thread="once"))
    decide(graph, "once", "maya", sha=req["sql_sha256"])
    names = [e["event"] for e in audit.recent(query_id="once")]
    assert names.count("approval_requested") == 1 and names.count("approval_decided") == 1
    decided = next(e for e in audit.recent(query_id="once") if e["event"] == "approval_decided")
    assert decided["user_id"] == "maya" and decided["detail"]["requested_by"] == "sam"
    assert audit.recent(query_id="once")[-1]["detail"]["approved_by"] == "maya"


# =============================================================== 4. API
H = {name: {"X-API-Key": f"key-{name}"} for name in USERS}


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def post_ask(client, user, q):
    return client.post("/ask", json={"question": q}, headers=H[user])


def post_decision(client, user, query_id, sha, approve=True):
    return client.post(f"/queries/{query_id}/decision", json={"approve": approve, "sql_sha256": sha}, headers=H[user])


def test_api_requires_a_valid_key_and_validates_input(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 401
    assert client.post("/ask", json={"question": "hi"}, headers={"X-API-Key": "nope"}).status_code == 401
    assert post_ask(client, "anna", "").status_code == 422
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/readyz").json() == {"ready": True}


def test_api_answers_reads_according_to_role(client):
    ok = post_ask(client, "anna", "Which customers are from the UK?").json()
    assert ok["status"] == "answered" and [r[0] for r in ok["rows"]] == ["Ada Lovelace", "Alan Turing"]
    denied = post_ask(client, "vera", "Which customers are from the UK?").json()
    assert denied["status"] == "refused" and denied["rows"] == []
    assert "@example.com" not in post_ask(client, "anna", "What are the customer emails?").text
    assert "@example.com" in post_ask(client, "sam", "What are the customer emails?").text


def test_api_write_flow_two_people_one_approval(client):
    pend = post_ask(client, "sam", "Cancel order 6").json()
    qid, sha = pend["query_id"], pend["approval"]["sql_sha256"]
    assert pend["status"] == "pending_approval" and pend["approval"]["rows_affected_preview"] == 1

    assert client.get(f"/queries/{qid}", headers=H["sam"]).json()["status"] == "pending_approval"
    assert client.get(f"/queries/{qid}", headers=H["maya"]).status_code == 200   # approvers can look
    assert client.get(f"/queries/{qid}", headers=H["anna"]).status_code == 404   # strangers cannot, and learn nothing
    assert post_decision(client, "sam", qid, sha).status_code == 403              # support cannot approve
    assert post_decision(client, "anna", qid, sha).status_code == 403

    done = post_decision(client, "maya", qid, sha).json()
    assert done["status"] == "applied"
    assert client.get(f"/queries/{qid}", headers=H["sam"]).json()["status"] == "applied"
    assert post_decision(client, "omar", qid, sha).status_code == 409             # already decided: no double apply


def test_api_blocks_self_approval_and_bad_ids(client):
    pend = post_ask(client, "maya", "Cancel order 6").json()
    qid, sha = pend["query_id"], pend["approval"]["sql_sha256"]
    assert post_decision(client, "maya", qid, sha).status_code == 403             # a manager cannot approve their own request
    assert post_decision(client, "omar", qid, sha).json()["status"] == "applied"  # a second manager can
    assert post_decision(client, "maya", "0" * 32, sha).status_code == 404
    assert post_decision(client, "maya", "not-an-id", sha).status_code == 422
    assert post_decision(client, "maya", qid, "xyz").status_code == 422


def test_api_wrong_fingerprint_cancels_instead_of_applying(client):
    pend = post_ask(client, "sam", "Cancel order 6").json()
    out = post_decision(client, "maya", pend["query_id"], "0" * 64).json()
    assert out["status"] == "cancelled"
    assert post_ask(client, "anna", "How many orders were pending in 2024?").json()["rows"] == [[1]]


def test_api_audit_is_for_auditors(client):
    pend = post_ask(client, "sam", "Cancel order 6").json()
    assert client.get("/audit", headers=H["sam"]).status_code == 403
    events = client.get("/audit", params={"query_id": pend["query_id"]}, headers=H["maya"]).json()["events"]
    assert [e["event"] for e in events] == ["question", "sql_checked", "approval_requested"]
    assert client.get("/audit", params={"limit": 0}, headers=H["maya"]).status_code == 422


def test_api_rate_limits_per_user(settings):
    now = [0.0]
    limiter = RateLimiter(2, clock=lambda: now[0])
    with TestClient(create_app(settings, limiter=limiter)) as c:
        assert post_ask(c, "anna", "How many customers do we have?").status_code == 200
        assert post_ask(c, "anna", "How many customers do we have?").status_code == 200
        limited = post_ask(c, "anna", "How many customers do we have?")
        assert limited.status_code == 429 and int(limited.headers["Retry-After"]) >= 1
        assert post_ask(c, "vera", "What is the most expensive product?").status_code == 200  # other users unaffected
        now[0] += 61
        assert post_ask(c, "anna", "How many customers do we have?").status_code == 200


def test_rate_limiter_window():
    now = [0.0]
    limiter = RateLimiter(1, clock=lambda: now[0])
    assert limiter.retry_after("a") is None
    assert limiter.retry_after("a") == pytest.approx(60.0)
    now[0] = 59.0
    assert limiter.retry_after("a") == pytest.approx(1.0)
    now[0] = 60.0
    assert limiter.retry_after("a") is None


def test_api_hides_internal_errors(settings):
    class Boom(Stub):
        @override
        def draft(self, *a, **k):
            raise RuntimeError("secret internal detail")

    with TestClient(create_app(settings, writer=Boom())) as c:
        res = post_ask(c, "anna", "How many customers do we have?")
    assert res.status_code == 502 and "secret" not in res.text and "request_id=" in res.text


def test_pending_approvals_survive_a_restart(settings):
    with TestClient(create_app(settings)) as first:
        pend = post_ask(first, "sam", "Cancel order 6").json()
    with TestClient(create_app(settings)) as second:  # a new process, the same checkpoint database
        done = post_decision(second, "maya", pend["query_id"], pend["approval"]["sql_sha256"]).json()
    assert done["status"] == "applied"


def test_concurrent_requests_do_not_interfere(client):
    def one(i: int) -> str:
        return str(post_ask(client, "anna", "How many customers do we have?").json()["answer"])

    with ThreadPoolExecutor(8) as pool:
        assert set(pool.map(one, range(16))) == {"customers: 6"}


# =============================================================== settings
def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("SQL_MAX_ROWS", "7")
    monkeypatch.setenv("SQL_STATEMENT_TIMEOUT_S", "0.5")
    s = Settings.from_env()
    assert s.max_rows == 7 and s.statement_timeout_s == 0.5 and s.max_write_rows == Settings().max_write_rows


# =============================================================== 5. evals
def test_eval_gate_passes():
    results = run_evals.run()
    assert run_evals.gate(results) == []
    assert results["access"]["leaks"] == 0


def test_eval_gate_fails_on_a_single_leak():
    results = run_evals.run()
    results["access"]["leaks"] = 1
    assert any("leaks" in f for f in run_evals.gate(results))

