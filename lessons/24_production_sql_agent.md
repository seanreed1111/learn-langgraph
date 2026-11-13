# LESSON 24 - From a safe SQL agent to a production SQL agent

Lesson 20 gave you a text-to-SQL graph that cannot `DROP TABLE`. This lesson answers the question a security
reviewer asks next: **"Fine. Now put it in front of twenty colleagues and a real database. What breaks?"**

The code lives in [`sql_prod/`](../sql_prod). Everything runs offline (a rule-based writer double); add a key to
`.env` and the same graph uses a real model. The shape of this lesson is the shape of
[lesson 14](14_production_rag.md): the tutorial version, a gap table, the decisions that matter, evals as the
safety net, what is still missing.

```powershell
pytest -q tests/test_sql_prod.py                       # 97 tests
python -m sql_prod.evals.run_evals                     # four suites + gates (exit code 1 on regression)
python -m sql_prod.cli "Cancel order 6" --user sam --approve-as maya --audit
uvicorn sql_prod.api:app --port 8001                   # the service
curl.exe -s localhost:8001/ask -H "X-API-Key: key-sam" -H "content-type: application/json" -d "{\"question\": \"Cancel order 6\"}"
```

---------------------------------------------------------------------------------------------------

## 0. Start by attacking lesson 20

A production review starts by testing what you already trust. Running lesson 20's guard against itself found four
real problems in about ten lines of probing (the probe is in section 3.1):

| Probe against lesson 20 | Result |
|---|---|
| `SELECT * FROM customers, sqlite_master` | **Passed the guard and returned rows.** The table regex only looked at the name after `FROM`; the read-only connection allowed any read, including the catalogue. |
| `SELECT ... WHERE id IN (SELECT ... LIMIT 1)` | The guard saw `LIMIT` anywhere and **skipped adding an outer one**. `LIMIT 100000000` was kept as written. |
| `SELECT zeroblob(200000000)` | A 200 MB value built in memory. |
| `WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT count(*) FROM r` | Allowed, and would run forever: **there was no timeout.** |

None of these is exotic. They are what a lexical check, a single shared connection and a "read-only" flag leave
open. The rest of the lesson is the fix for each, plus the things lesson 20 did not attempt (who is asking, who
approves, what happened).

---------------------------------------------------------------------------------------------------

## 1. The gap: lesson 20 vs. production

| Concern | Lesson 20 | Lesson 24 | File |
|---|---|---|---|
| Guard | Keyword/regex scan; "not a parser" | **`sqlglot` parse tree, allow-lists**, executes the **re-serialised** statement | `guard.py` |
| Who is asking | Nobody. One global table list | **`SqlContext(user, role)` + per-role `Policy`** (table *and column*), unknown role = no access | `policy.py` |
| Permissions enforced by | The guard (text) | **The database**: a SQLite authorizer built from the policy, so a guard bug cannot leak | `db.py` |
| Schema shown to the model | All of it | **Only what the role may read** | `db.py`, `graph.py` |
| Connections | One shared read-only connection, no lock | **One per operation**; nothing shared across threads | `db.py` |
| Resource limits | `LIMIT` text rewrite | **Statement timeout, value-size limit, row cap applied as rows leave the DB** | `db.py`, `config.py` |
| Error handling | Every DB error -> ask the model to repair | **Classified**: denied / timeout / too big / constraint are final; only real SQL errors are repaired | `graph.py` |
| Approval | `interrupt()`, anyone can resume | **Different person, with approver role, within a time window, for the exact statement** | `graph.py`, `api.py` |
| Write apply | Re-run the SQL blindly | **Drift check inside one transaction**; writes run under the *requester's* role | `db.py`, `graph.py` |
| Constraints | Foreign keys off | **On for the writer**, so the preview reports failures before anyone approves | `db.py` |
| State | One flat dict, partly reset | **Typed state, narrow input/output schemas, full per-turn reset** | `graph.py` |
| Audit | None | **Append-only audit log**, enforced by triggers; metadata, never rows | `audit.py` |
| Observability | print | JSON logs with per-node timing, a node trace in every answer | `graph.py` (reuses lesson 14's `@timed`) |
| Resilience | None | Retries on transient *model* errors only; model timeouts | `graph.py`, `llm.py` |
| Serving | CLI | FastAPI: auth, ownership checks, rate limits, durable approvals, health/readiness, audit endpoint | `api.py` |
| Quality control | 3 suites | **4 suites + a leak counter that must be 0 + drift/self-approval cases** | `evals/` |

---------------------------------------------------------------------------------------------------

## 2. Architecture

```
START -> guard_input --(empty / too long / role can read nothing)--> reject ---------------------+
            v                                                                                    |
        load_schema     schema filtered by the caller's role                                     |
            v                                                                                    |
       write_query <------------------------- bump <-- error, repairs left                      |
            v                                                                                    |
       check_query      parse-guard, dry run under the role's authorizer, rolled-back preview    |
            |-- blocked / not authorized ----> refuse ------------------------------------------+
            |-- nothing answerable ----------> out_of_scope ------------------------------------+
            |-- error, no repairs left ------> fail --------------------------------------------+
            |-- read ---> run_query -> answer ---------------------------------------------------+
            |                 '-- timeout ---> timed_out ----------------------------------------+
            |-- write, 0 rows ---------------> no_change ---------------------------------------+
            '-- write ---> approve_write (interrupt) -> apply_write ----------------------------+
                                '-- not approved / not allowed / expired / mismatch -> cancelled +
                                                                                                 v
                                                                                   record (audit) -> END
```

Still a **fixed workflow**, not a tool loop, for the lesson-20 reasons: the steps never change and the dangerous
decision is a node you own. Print the real graph: `graph.get_graph().draw_mermaid()`.

Every path ends in `record`, one node that writes the audit entry. That is a design choice worth copying: when
"every run is audited" is a property of the *graph shape*, you cannot forget it on one branch.

---------------------------------------------------------------------------------------------------

## 3. The decisions that matter (read the code with these in mind)

### 3.1 A parser, and "run what you validated" (`guard.py`)
- `sqlglot.parse(sql, read="sqlite")` builds the same kind of tree the database builds. The guard then
  **allow-lists over the tree**: exactly one statement, a SELECT/UNION or INSERT/UPDATE/DELETE, and *every*
  `Table` node (comma joins, subqueries, UNION arms, CTE bodies) on the role's list. No schema-qualified names
  (`main.sqlite_master`), no table-valued functions (`pragma_table_info(...)`), a short function deny-list.
- **`Verdict.sql` is re-serialised from the approved tree**, not the original string. If the guard's parser and
  SQLite ever read the original text differently (a *parser differential*, the root cause of many injection
  bugs), the database only sees what the guard saw. `test_reserialised_reads_return_the_same_rows` proves the
  rewrite does not change meaning.
- Comments are still refused (they hide intent from a human reviewer), checked on the text with string
  literals blanked out so `'a--b'` is data.
- A parser narrows the gap; it does not remove it. Every parser has edge cases, so section 3.2 exists.

### 3.2 The database enforces the policy (`db.py`, `policy.py`)
The strongest idea in the lesson: **express permissions as data, let the database enforce them, and treat the
guard as a friendly error message in front of that.**
- `Policy` maps each table to the columns a role may read, and lists the tables it may write. Roles:
  `viewer` (products), `analyst` (everything but `customers.email`), `support` (email; may *propose* order
  changes), `manager` (may also propose customer changes, approves others' writes, reads the audit log).
- `reader_authorizer(policy)` is a SQLite authorizer: it is called at **compile time** for every table and
  column a statement touches. Allowed -> compiles. Anything else -> `SQLITE_AUTH`. That covers `SELECT *`
  (expands to include `email`), aliases, subqueries, `WHERE email LIKE ...` (a column you filter on is
  still a column you read), and `sqlite_master`.
- Real databases do this with **GRANTs** (`GRANT SELECT (id, name) ON customers TO analyst`) and row-level
  security. SQLite has neither, so the authorizer plays that role. The *shape* (policy as data, enforced
  below the application) is what carries over.
- Mutation test (run it, section 5): replace the authorizer with "allow everything" and the ACCESS suite
  fails with `leaks=2`. Replace the *guard* with one that misses unknown tables and **nothing fails**, because
  the database still refuses. That is defence in depth working, and a measurable result.
- The model is shown only the columns its role may read (`schema_text`). It cannot leak what it has never seen.

### 3.3 Limits that cannot be talked around (`db.py`)
| Limit | Mechanism | Stops |
|---|---|---|
| Wall-clock timeout | `set_progress_handler` aborts the statement ("interrupted") | `WITH RECURSIVE` forever, accidental cross joins |
| Value size | `setlimit(SQLITE_LIMIT_LENGTH, ...)` | `zeroblob(2e8)`, string-building bombs |
| Row cap | `fetchmany(max_rows + 1)` as rows leave the DB; `truncated` flag | unbounded result sets |
- Why the row cap is *not* a `LIMIT` rewrite: lesson 20's rewrite was bypassable (the subquery case above).
  The cap lives where rows exit, so no SQL spelling avoids it. Fetching one extra row is how you know whether the
  result was truncated, so the answer can say so (a deterministic sentence, not left to the model).
- Postgres equivalents: `statement_timeout`, `SET default_transaction_read_only = on`, pool limits.

### 3.4 Not every database error means "ask the model to try again" (`graph.py: _classify`)
Lesson 20 fed every error back to the writer. Production sorts them:
| Error | Meaning | Route |
|---|---|---|
| `SQLITE_AUTH` | the role may not touch that data | **refuse** (retrying is pointless; it is also a signal worth logging) |
| interrupted | the statement is too slow | **timeout** (a slow query will be slow again) |
| too big | a value exceeded the size limit | **refuse** |
| `IntegrityError` | the *data* forbids the change (foreign key) | **refuse** before anyone is asked to approve |
| anything else | probably a wrong column or syntax | **repair**, at most `max_repairs` times, then `failed` |
Tests: `test_deterministic_database_errors_are_not_retried` asserts the writer is called exactly once.

### 3.5 Approval is a control, so it needs rules (`graph.py: approve_write`, `api.py`)
Lesson 20's approval was a boolean anyone could send. Here, every one of these must hold or the result is
`cancelled` and **nothing changes** (each has a parametrized test and an eval case):
1. **The decision says yes.**
2. **The approver's role may approve** (`Policy.can_approve`).
3. **The approver is not the requester.** Two people for a write.
4. **Within the window** (`approval_ttl_s`). Data moves; a four-day-old yes is not a yes.
5. **For the statement that will run.** The pending response carries `sql_sha256`; the decision must send it
   back. This is "approve what you saw" made mechanical.
- **Whose identity is checked?** `interrupt()` resumes the *same* run with a *new* invocation, and
  `runtime.context` is the context of the **resuming call**. So the code after `interrupt()` sees the approver,
  and the audit trail records the right person. Identity comes from `context` (set by your auth layer), never
  from the resume payload the client sends.
- The API checks 2 and 3 first (so a bad request gets a clean 403/409 without touching the run), and the graph
  checks them again. Never rely on only the outer layer.
- **Idempotency, again:** code before `interrupt()` re-runs on resume. So `approval_requested` is audited in
  `check_query` (runs once), and `approval_decided` after the interrupt. `test_approval_events_are_written_exactly_once...`.

### 3.6 Apply with a drift check, under the requester's role (`db.py: apply_write`)
- The preview (rolled back) tells the human "1 row". Minutes later they say yes. In between, someone inserted
  another pending order, and the same `UPDATE ... WHERE status='pending'` now changes 2. Lesson 20 would
  apply it silently.
- `apply_write` takes `BEGIN IMMEDIATE` (the write lock), runs the statement, and **commits only if the row
  count equals what the approver was shown**; otherwise it rolls back and the node reports `stale`. The data
  change and the check are one atomic step. (Postgres: `SELECT ... FOR UPDATE`, or a version column.)
- The write runs under the **requester's** policy, not the approver's. Approval confirms intent; it must never
  *widen* privilege. `test_approval_never_widens_the_requesters_privileges` forges the saved role and shows the
  database refusing.
- Foreign keys are on for the writer, so `Delete customer 3` is refused during the preview with the database's
  own message, instead of failing after someone approved it.

### 3.7 Typed state, narrow schemas, per-turn reset (`graph.py`)
Same lesson as 14 section 3.4, new instance: `SqlInput` is just `{question}`; `SqlOutput` is the public contract
(`status`, `answer`, `sql`, `rows`, `truncated`, `trace`, ...). The schema text, fingerprints and counters stay
internal. `guard_input` resets **every** per-turn field, because the checkpointer keeps the thread's state:
`test_per_turn_state_resets_between_turns_on_the_same_thread` asks three questions on one thread.

### 3.8 An audit trail you cannot quietly edit (`audit.py`)
- Who asked, which SQL was checked and the verdict, who was asked to approve, who decided and why, how it ended.
- **Append-only in the database**: triggers abort `UPDATE`/`DELETE`. In production use a separate account the app
  can only `INSERT` into, or an immutable sink.
- **Metadata, not content.** SQL, counts, decisions - never result rows. An audit log full of customer data is its
  own compliance problem (`test_every_run_leaves_a_record_without_the_data_it_returned`).

### 3.9 The API layer (`api.py`)
- `/ask` returns a `query_id`. A write returns `status: pending_approval` plus the approval details and
  fingerprint; `/queries/{id}/decision` is called by someone else.
- **A query id is a handle, not a permission.** Reading a query requires being its requester or an approver;
  anyone else gets **404, not 403**, so ids cannot be probed.
- **Durable approvals**: a `SqliteSaver` holds the paused run (`test_pending_approvals_survive_a_restart`).
  Several replicas need Postgres.
- Per-user **rate limit** (429 + `Retry-After`), `/healthz` vs `/readyz` (checks the database), a clean 502 with a
  request id and no internals, and `/audit` for roles that may read it.

### 3.10 Evals are the safety net (`evals/`)
| Suite | Cases | Gate |
|---|---|---|
| READ | 11, as a role, **execution accuracy** (result sets, not SQL text) | >= 0.95 |
| SAFETY | 14: classic hostile prompts, the lesson-20 **bypasses**, resource abuse; the status must match, the **database hash must be unchanged**, and the run must be **in the audit log** | 1.00 |
| ACCESS | 10: each role asks for what it should and should not see; `leaks` counts forbidden strings anywhere in the response | 1.00 and **leaks = 0** |
| WRITES | per case: paused with data unchanged, applied by an authorised other person, and **four denial paths** (reject, unauthorised, self, wrong fingerprint) each leaving data unchanged; plus a **drift** case where the data changes mid-approval | 1.00 |
- Latency (mean, p95) is printed and not gated: gate on what you can defend.
- **Prove the gate can fail.** Do the mutation runs in section 5. A suite that has never failed is a hope, not a control.

---------------------------------------------------------------------------------------------------

## 4. Moving to real models and a real database
1. `.env`: `ANTHROPIC_API_KEY=...`. `sql_prod/llm.py` delimits the question *and the returned rows* as data
   (a customer named "ignore previous instructions" is an attack on the answer step) and tells the model which
   columns exist. Run the evals: expect different READ numbers; the SAFETY/ACCESS gates should not move,
   because they test the rails, not the model.
2. Point `SQL_DB_PATH` at a real file, then swap SQLite for Postgres behind `db.py`: a role per policy with column
   `GRANT`s, `default_transaction_read_only`, `statement_timeout`, a connection pool, row-level security for tenants.
   The graph does not change; it only calls `schema_text`, `dry_run`, `run_read`, `preview_write`, `apply_write`.
3. Audit to a separate database or log sink; checkpoints to `PostgresSaver` (`.setup()` once at deploy).
4. OIDC/JWT at a gateway: groups -> role, replacing the demo API keys.
5. Deploy: the `Dockerfile` (shared image, second service in `docker-compose.yml`), or `langgraph dev` using
   `langgraph.json` entry `sql_prod` (pass `{"user_id": "sam", "role": "support"}` as the run context in Studio; a run
   with no context gets no access). *The Dockerfile and compose service were written but not executed in this
   environment - try them.*

---------------------------------------------------------------------------------------------------

## 5. Try to break it (mutation checks)
Weaken one defence at a time and see which suite objects. This is how you learn what each layer is for:

| Change | Expected |
|---|---|
| Make `reader_authorizer` return OK for everything | ACCESS fails, `leaks=2` (emails) |
| Delete the `expected_rows` comparison in `apply_write` | WRITES fails on the drift case |
| Make `validate_sql` accept unknown tables | **Everything still passes**: the database layer caught it (that is the point) |
| Remove the self-approval check in `approve_write` | WRITES fails on `self` (and a unit test) |
| Raise `max_write_rows` to 10 | the `Cancel all 2024 orders` safety case fails |

---------------------------------------------------------------------------------------------------

## 6. What is still missing (see `PRODUCTION_CHECKLIST.md`)
Row-level security and tenant isolation (the policy is per role, not per row), PII redaction/masking in answers
(an allowed column can still contain sensitive free text), query cost estimation (`EXPLAIN` plans) before running,
approval notifications and a queue of pending requests (there is a poll endpoint, not a worklist), OIDC and
service-to-service auth, async I/O and pooling, metrics, per-user token/cost budgets, large-schema retrieval
(lesson 14 techniques), follow-up questions ("and for 2023?" needs `condense` from lesson 14), canary rollout of
prompt changes. Knowing this list, ordered by risk, is what a senior interviewer is probing for.

---------------------------------------------------------------------------------------------------

## 7. Exercises (do at least 1, 3 and 5)
1. **Tenant isolation.** Add `tenant_id` to `customers` and `orders`, query through per-tenant **views** (or a
   `SELECT`-time filter) and write ACCESS cases proving tenant A cannot read tenant B, including through a join.
2. **Break the guard on purpose.** Find a statement the parser guard allows that it should not (try odd function
   calls, deeply nested CTEs, `WITH ... INSERT`). Confirm the authorizer still stops it, then add it to the guard and the tests.
3. **Mutation-test every layer** in section 5, then add one more mutation of your own and the eval case that catches it.
4. **Cost gate.** Run `EXPLAIN QUERY PLAN` in `check_query` and refuse plans with a full scan of a table over N rows.
5. **Approval worklist.** Add `GET /pending` for approvers (hint: query the checkpointer for threads whose next node is
   `approve_write`) and an expiry sweeper.
6. **PII masking.** Make the authorizer return `SQLITE_IGNORE` for `email` for the analyst role, so `SELECT *` works but
   the column is NULL. Which behaviour is safer for a *model* reading the result, and why?
7. **Multi-turn.** Add lesson 14's `condense` so "and for 2023?" works, and prove with a test that the policy is still
   applied to the rewritten question.

## 8. Interview talking points
- "How do you let an LLM near a production database?" -> policy as data, enforced by the database (grants/authorizer);
  the guard is a courtesy; limits (timeout, size, rows) at the boundary; approval for writes; audit everything.
- "Your guard is regex. Why is that a problem?" -> the checker and the database can disagree about the text. Parse it,
  and run the re-serialised statement. Then assume the parser is wrong too and enforce below it.
- "Who can approve a write?" -> someone other than the requester, with the approver role, inside a time window, for the
  exact statement (fingerprint). Show the five conditions.
- "What if the data changes between approval and execution?" -> apply inside one transaction with a row-count
  (or version) check; roll back and report `stale`.
- "Why 404 instead of 403?" -> do not confirm that a resource you cannot see exists.
- "How do you know the safety work is real?" -> mutation tests: weaken a layer, watch a named suite fail; and the one layer that
  can be removed with no failure is redundant by design.
