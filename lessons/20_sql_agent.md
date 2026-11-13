# Lesson 20 - A SQL analyst agent that is safe to point at a database

Code: `sql_agent/` · Tests: `tests/test_sql_agent.py` · Official page: *Build a custom SQL agent*

```powershell
python -m sql_agent.cli "Which customers are from the UK?"
python -m sql_agent.cli "Cancel order 6" --approve
python -m sql_agent.evals.run_evals
```

Offline by default (a rule-based writer double). With an API key in `.env`, the same graph uses a real model.

## The one idea

An LLM that writes SQL is an LLM that can write `DROP TABLE`. You do not fix that with a better prompt. You fix it with **layers that each assume the previous one failed**, and with a graph in which the dangerous step is a node *you* own.

| Layer | Where | Catches |
|---|---|---|
| 1. Guard | `guard.py` | multi-statement, comments, DDL/PRAGMA/ATTACH, unknown tables, UPDATE/DELETE without WHERE; adds `LIMIT` |
| 2. Dry run | `db.dry_run` (`EXPLAIN`) | bad columns/tables/syntax, before running |
| 3. Read-only DB | `db.connect_readonly` | `mode=ro` + SQLite authorizer: the **database** refuses writes even if layers 1-2 have a bug |
| 4. Write preview | `db.preview_write` | runs the write in a transaction, counts rows, **rolls back**; > 5 rows is refused |
| 5. Human approval | `approve_write` (`interrupt`) | the only path to a write; payload shows exact SQL + row count |
| 6. Separate write connection | `apply_write` node | the only node that holds a writable connection |
| 7. Least privilege | your DB (not SQLite) | a SELECT-only role in Postgres/MySQL |

The guard is a keyword/regex classifier, **not** a parser. That is why layer 3 exists, and `test_database_itself_refuses_writes_even_if_guard_is_bypassed` proves it. In production, use `sqlglot` in layer 1; the interface does not change.

## Architecture

```
START -> list_tables -> get_schema -> write_query -> check_query
                                          ^              |-- blocked ------> refuse
                                          |              |-- none ---------> out_of_scope
                              bump  <-----+-- error (<= 2 repairs) / else fail
                                                         |-- read  -> run_query -> answer
                                                         '-- write -> approve_write -> apply_write | cancelled
```

**Why a fixed workflow, not the official ReAct tool loop?** The steps are always the same, so a graph is cheaper, faster, and testable. The dangerous decision cannot be skipped by the model. The model only drafts SQL and phrases the answer. Use a tool loop when the *set of steps* is unknown.

## Walkthrough

- **Schema comes from trusted code**, not from a model-controlled tool call (`db.schema_text`).
- **Self-repair is bounded.** `check_query` and `run_query` feed the error text back to `write_query`, at most `MAX_REPAIRS = 2` times, then `failed`. The offline writer's first `revenue by category` draft is deliberately wrong (`no such column: price`), so the loop is exercised in the evals (`expect_repairs: 1`).
- **Approval is idempotent-safe.** `interrupt()` re-runs its node from the top on resume, so the preview (which rolls back) sits *before* it and the real write sits in a *separate node after* it.
- **Question text is data.** The real-model prompt says so, but the prompt is not a control. The guard is.
- **Studio:** `build_graph(..., checkpointer=None)` for the platform, which supplies persistence (`langgraph.json` graph `sql`).

## Evals (`sql_agent/evals/`)

| Suite | What it measures | Gate |
|---|---|---|
| READ | **Execution accuracy**: run agent SQL and *gold* SQL, compare **result sets**, not SQL text (many queries are correct) | >= 0.95 |
| SAFETY | hostile + out-of-scope prompts return `refused`/`out_of_scope` **and** the DB content hash is unchanged | 1.00 |
| WRITES | pauses before changing anything; applies exactly the intended change on approve; changes nothing on reject | 1.00 |

Do not trust the status field for safety: the suite hashes the whole database before and after.

Honest caveat: the offline writer is a double, so these numbers prove the **rails and the harness**, not model quality. Run the same suite with a real model to measure the model.

## Known limitations (say these in an interview)

- The guard is conservative: a column literally named like a forbidden keyword will be refused.
- No row-level or column-level security; add per-user views or a role per tenant.
- SQLite only. For Postgres: `SET default_transaction_read_only = on`, statement timeouts, a SELECT-only role.
- No cost control: add `statement_timeout`, and consider refusing queries whose `EXPLAIN` shows huge scans.
- Large schemas will not fit in the prompt: retrieve relevant tables first (lesson 14 techniques).

## Exercises

1. Add a `sqlglot` parse to `validate_sql` and compare its verdicts with the keyword guard on the test list.
2. Add a per-user `tenant_id` filter by querying through a view, and write the eval that proves tenant A cannot see tenant B.
3. Raise `MAX_WRITE_ROWS`, add a case that touches 6 rows, and watch the safety eval catch the regression.
4. Add a statement timeout using `sqlite3.Connection.set_progress_handler`.
5. Swap in a real model and record execution accuracy per question type.

## Next

[Lesson 24](24_production_sql_agent.md) attacks this agent (a comma join reads `sqlite_master`; there is no timeout) and makes it production-shaped: roles and column policy enforced by the database, a parser guard, approval rules, a drift check, an audit trail and an API.
