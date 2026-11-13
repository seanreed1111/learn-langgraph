# LangGraph Interview Prep

Answer each out loud in 60-90 seconds, then check the lesson noted.

## Fundamentals
1. **What is LangGraph and why not just a while-loop around an LLM?** Low-level orchestration runtime for stateful agents: explicit state, checkpointing/durable execution, human-in-the-loop, streaming, parallelism, observability. A loop gives you none of the persistence/pause/resume. (01, 07, 08)
2. **Explain State, Nodes, Edges.** State is a typed shared schema; nodes are functions returning partial updates; edges (static, conditional, `Command`) decide what runs next. (01, 03)
3. **What is a reducer? What happens without one?** Merge function `(old, new) -> merged` per key. Without it the last write overwrites; parallel writes to the same key raise `INVALID_CONCURRENT_GRAPH_UPDATE`. (02)
4. **What does `add_messages` do beyond append?** Merges by message ID (replace), coerces dicts to messages, supports `RemoveMessage`. (02)
5. **Conditional edge vs `Command`?** Edge = routing separate from work; `Command` = node updates state and routes atomically (handoffs). (03)
6. **What is a super-step?** Pregel-style round: schedule nodes, run in parallel, apply updates via reducers, repeat. (04)
7. **How do you prevent infinite loops?** `recursion_limit`, step counters in state, explicit termination conditions; catch `GraphRecursionError`. (03)

## Agents
8. **Walk through a ReAct agent graph.** llm node -> router on `tool_calls` -> tools node -> back to llm; ends when no tool calls. (05)
9. **`create_agent` vs custom graph?** Prebuilt for standard tool loops (+ middleware); custom when you need custom routing, approval gates, multi-stage workflows, or non-chat state. (06)
10. **How do you handle tool failures?** Catch and return the error as a `ToolMessage` so the model can recover; `ToolNode` handles errors; retries on transient errors only. (05, 12)
11. **Workflow vs agent?** Workflows follow predetermined code paths (prompt chaining, routing, parallelization, orchestrator-worker, evaluator-optimizer); agents let the LLM decide the path dynamically. Use the simplest thing that works.

## State, memory, persistence
12. **Short-term vs long-term memory?** Short-term = per-thread state via checkpointer. Long-term = cross-thread Store with namespaces. (07, 10)
13. **What does a checkpointer enable?** Memory, HITL, fault tolerance, time travel. Which in prod? Postgres. (07)
14. **How would you fork/debug a bad run?** `get_state_history`, pick checkpoint, `update_state`, resume with `invoke(None, cfg)`. (07)
15. **Context window overflow?** Trim/summarize history, store facts in long-term memory, retrieve selectively. (10)

## Human-in-the-loop
16. **How does `interrupt` work and what's the biggest gotcha?** Saves state and pauses; resume with `Command(resume=...)`; the node re-executes from the top, so pre-interrupt side effects must be idempotent. (08)
17. **Design an approval flow for a refund agent.** Interrupt before the side-effecting tool; payload shows exact args; log decision; handle reject path; guard against replays/stale approvals; test that nothing happens pre-approval (see capstone test). (08, capstone)

## Scale & production
18. **Parallelize N document analyses.** `Send` map-reduce with a reducer on the results key. (04)
19. **Stream tokens and progress to a UI?** `astream` with `messages` + `updates`/`custom` modes via SSE/WebSocket. (09)
20. **Deploy it.** `langgraph.json`, `langgraph dev` for local/Studio, Docker/LangSmith Deployment, Postgres checkpointer, env/secrets. (12)
21. **How do you test and evaluate agents?** Unit-test nodes; scripted fake model for deterministic graph tests; trajectory + outcome evals on datasets (LangSmith); regression-run before prompt/model changes. (12)
22. **Multi-agent: when and why not?** Supervisor/handoff patterns; cost, latency, debuggability; start with one agent. (11)
23. **Security?** Prompt injection via tool outputs, least-privilege tools, approval gates for destructive actions, input validation, PII redaction, per-user namespaces/context.

## Advanced (lessons 16-20, 24)
24. **Graph API or Functional API?** Graph: explicit topology, shared typed state with reducers, Studio diagrams, many branches. Functional: wrap existing code, data-dependent loops, less ceremony. Same persistence/HITL/streaming; they compose. (16)
25. **What is the replay rule in the Functional API?** The entrypoint re-executes on resume; `@task` results are replayed from the checkpoint. So side effects and non-determinism (time, uuid, random, HTTP) go inside tasks. (16)
26. **A run crashed halfway. What happens on resume?** `invoke(None, same thread_id)` continues from the last checkpoint; branches that already finished in the failed super-step are restored from pending writes; the failed node re-runs from its top, so its side effects must be idempotent. (17)
27. **Explain durability modes.** `exit` (save at end), `async` (background save), `sync` (save before next step). Sync for money, async for chatty agents. (17)
28. **How does LangGraph execute a graph?** Pregel / bulk-synchronous: plan, execute in parallel on one snapshot, apply writes through channels (reducers run here), checkpoint, repeat. Recursion limit counts super-steps. (18)
29. **Why `InvalidUpdateError` / why didn't my parallel node see the other's write?** Two writes to a `LastValue` channel in one step; and same-snapshot semantics. Fix with a reducer or sequence the nodes. (18)
30. **How do you let an LLM query a production database safely?** Layers: guard, dry-run, read-only connection, rolled-back write preview, human approval, least-privilege DB role, and evals that hash the DB before and after hostile prompts. Prompts are not a control. (20)
31. **How do you evaluate text-to-SQL?** Execution accuracy: compare result sets of agent SQL vs gold SQL, not SQL strings; plus a separate safety suite and a write-gating suite. (20)
32. **What are the common silent failures in LangGraph?** In-place state mutation (checkpoint diverges), typo'd update keys dropped, router typos ending the run with only a log warning, fan-in via separate edges firing twice, `Command(goto)` plus a static edge running both, `try/except` swallowing `interrupt()`, fresh input on a paused thread replacing the request being approved. (19)
33. **How do you stop an agent from reading another user's data?** Never let the model supply identity: inject user/tenant from trusted runtime context (`ToolRuntime`), namespace `thread_id` by authenticated user, 404 on foreign threads. (19)
34. **What is the default recursion limit and why does it matter?** 10007 super-steps in 1.2 (old docs say 25): set your own and keep a budget in state. (19)
35. **Your SQL agent's guard is a regex. What would you change before production?** Parse (sqlglot) and allow-list the tree, run the re-serialised statement, then enforce below the app: per-role table/column policy in the database (grants or an authorizer), statement timeout, value-size limit, row cap as rows leave the DB; approval by a different authorised person, in a window, for the exact statement (fingerprint), applied in one transaction with a row-count drift check; append-only audit of metadata. Prove the gates can fail with mutation tests. (24)

## Whiteboard drills (do without looking)
- Draw and code the ReAct loop (lesson 05) in 10 minutes.
- Add an approval gate in front of one tool.
- Add memory across sessions for a user.
- Explain how you'd debug "the agent called the wrong tool 5% of the time".
