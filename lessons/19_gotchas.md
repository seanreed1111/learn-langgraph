# Lesson 19 - Gotchas: how people make LangGraph harder than it needs to be

Proof: `lessons/19_gotchas.py` (run it) · Pinned by: `tests/test_gotchas.py` · Verified on `langgraph 1.2.21`

Most LangGraph pain at the intermediate/advanced level is not "I don't understand graphs". It is a small set of **silent failures**: the graph runs, nothing raises, and the behavior is wrong. This lesson names them, shows what actually happens, and gives the fix.

Two rules for reading it:

- **[verified]** items run in `19_gotchas.py` and are asserted by a test. If an upgrade changes the behavior, the test fails and this page is stale.
- **[judgment]** items are design advice. They are what I would do, not something a test can prove.

```powershell
python lessons/19_gotchas.py
pytest tests/test_gotchas.py -q
```

---

## Part 1 - Silent failures (the graph "works", the result is wrong)

### 1. Mutating state in place **[verified]**

```python
def node(state):
    state["items"].append("x")   # looks natural in Python
    return {}
```

What actually happens: the caller sees `['x']` in the result, **your own input dict was mutated**, and the **checkpoint saved `[]`**. Memory, resume and time travel now disagree with what the user saw. You will chase this for hours.

Fix: nodes are functions from state to an *update*. Return new values: `return {"items": state["items"] + ["x"]}`. Treat `state` as read-only.

### 2. Resending history, or no reducer on `messages` **[verified]**

With a checkpointer, **the server owns the history**. Two ways people fight that:

- A chat frontend rebuilds `HumanMessage`/`AIMessage` objects from plain text (no ids) and sends the whole conversation every turn. `add_messages` merges by id, so id-less copies are *appended*: you get `hi, reply1, hi, reply1, more, ...`. Fix: send **only the new turn** for a thread that has a checkpointer.
- Declaring `messages: list` without a reducer: each node's return **replaces** the list, so history vanishes. Fix: `Annotated[list, add_messages]` or inherit `MessagesState`.

### 3. Typos in updates are dropped without a word **[verified]**

`return {"ouput": "x"}` (typo) is silently discarded. So are unknown keys in the **input** (`invoke({"q": ..., "bogus": 1})`). This also holds for Pydantic state. A non-dict return *does* raise (`InvalidUpdateError`), which makes the silent case more surprising.

Fix: pin node outputs in unit tests with a tiny guard (`assert_update_keys` in the proof file), and test every node as a plain function (lesson 12). The `Command[...]`/`TypedDict` types help your editor, not the runtime.

---

## Part 2 - Control flow

### 4. `Command(goto=...)` plus a static edge runs both **[verified]**

A node that returns `Command(goto="b")` already routes itself. If you *also* wrote `add_edge("a", "c")`, **both `b` and `c` run**. Pick one routing mechanism per node: `Command` for "update + route together", `add_conditional_edges` for pure routing, `add_edge` for fixed flow.

### 5. A typo'd router target ends the graph quietly **[verified]**

A router that returns `"anwser"` does not raise. The run **finishes with a log warning** ("wrote to unknown channel ... ignoring it") and an empty result. If you do not watch logs, this looks like the model returning nothing.

Fix: give `add_conditional_edges` an explicit path list/map: the same typo then raises `KeyError` instead of vanishing (and the diagram is accurate). Also annotate routers and `Command` with `Literal[...]`, and write a test that drives **every** branch.

### 6. Fan-in with separate edges fires early, and twice **[verified]**

```python
g.add_edge("fast", "join")
g.add_edge("slow2", "join")      # join runs when EITHER arrives
```

With branches of different length, `join` ran **twice** (once with partial data). The fix is one edge with a list: `g.add_edge(["fast", "slow2"], "join")`, which waits for **all**. See lesson 18 for why (super-steps).

### 7. The default recursion limit is 10007, not 25 **[verified]**

Older tutorials (and the earlier version of lesson 03 in this repo) say 25. In 1.2 it is **10007 super-steps**. A confused agent loop will make thousands of paid model calls before it stops.

Fix: **always** pass your own limit, *and* keep a budget in state for the loops you design (`max_rewrites` in lessons 13-15). The limit is a seatbelt, the budget is the design. Remember the limit counts super-steps, not nodes (lesson 18).

---

## Part 3 - Persistence and human-in-the-loop

### 8. `try/except` around `interrupt()` eats the pause **[verified]**

`interrupt()` works by raising `GraphInterrupt`. A broad `except Exception` catches it, so the node **never pauses** and continues with your fallback (the demo "auto-approves" a refund). Never wrap `interrupt()` in a bare handler; catch specific exceptions around *other* calls only.

### 9. Sending fresh input to a paused thread **[verified]**

If a thread is waiting on `interrupt()` and the next request carries ordinary input instead of `Command(resume=...)`, the run **restarts with the new input and pauses again**. The demo shows the human now being asked "approve?" about a **$5000** refund after clicking approve for a **$5** one. The question text is identical, so the UI cannot tell.

Fix: before accepting input, check `graph.get_state(config).next` (or `.tasks[*].interrupts`). If the thread is paused, accept *only* a resume. Include an id of the thing being approved in the interrupt payload and check it on resume (lesson 08).

### 10. Putting un-checkpointable objects in state **[verified]**

DB connections, locks, clients, open files, arbitrary class instances: with a checkpointer on, the run fails with `Type is not msgpack serializable`. It works fine *without* one, which is why it often first fails in staging.

Fix: state holds **data** (ids, strings, dicts, Pydantic models). Dependencies go in a closure/factory (`build_graph(retriever, model)`, as in lessons 14-15 and 20) or in runtime context.

### 11. One `thread_id` for everyone **[verified]**

`thread_id` is the *only* thing separating conversations. Use `"support"` for all users and Bob's first message arrives with Alice's password in his history. Namespace it per user and per conversation (`f"{user_id}:{conversation_id}"`), derive `user_id` from your **authenticated session**, never from the request body. Lesson 15's API does this and returns 404 (not 403) for other people's threads.

---

## Part 4 - Tools and security

### 12. Letting the model say whose data to read **[verified]**

```python
@tool
def orders(user_id: str) -> str: ...
```

`user_id` is now part of the tool's JSON schema, so **the model supplies it**, and a prompt-injected model supplies someone else's. In the demo this returned `orders of victim`.

Fix: take identity from trusted runtime context. With `ToolRuntime`, the argument is **hidden from the schema** and the model's attempt to pass `user_id` is ignored (the demo returns `orders of alice`):

```python
@tool
def orders(runtime: ToolRuntime[Ctx]) -> str:
    return f"orders of {runtime.context.user_id}"

graph.invoke(inputs, context=Ctx(user_id=session.user_id))
```

Rule: **anything that decides authorization comes from your code, never from model output.** Same for tenant ids, roles, and "dry_run=False" flags.

---

## Part 5 - Design mistakes that make everything harder **[judgment]**

These are not bugs; they are the habits that create the bugs above.

1. **Using a graph where a function would do.** If the steps never branch, loop, pause, or need durability, a plain function (or `create_agent`) is simpler. Reach for `StateGraph` when you need explicit control flow, checkpoints, or HITL.
2. **Asking the LLM to route when code can.** Every LLM router is a probabilistic `if`. Route on state you control (a grade, a flag, a count). Spend model calls on judgment, not plumbing (lesson 15 routes with code on structured grader output).
3. **A node per line of code.** Nodes are the unit of checkpointing, retry, streaming and tracing. Split where you want one of those; do not split to look tidy. Side effects each get their own node (lesson 17).
4. **State as a junk drawer.** Every key is a channel you must reason about. Prefer a few typed keys; use `input_schema`/`output_schema` (lesson 02) so internals do not leak to callers.
5. **Unbounded everything.** Loops without budgets, history without trimming, tool calls without timeouts, retries without `retry_on`. Each "unbounded" is a cost incident waiting for a bad day.
6. **Caching the compiled graph "for performance".** Compiling a small graph took about half a millisecond here **[verified]**. What is expensive is the model client, the checkpointer connection and the vector index; build those once, compile freely.
7. **Copying old tutorials.** `create_react_agent` is deprecated in favor of `create_agent` (its docstring says so **[verified]**); per-run data belongs in `context_schema`/`Runtime`, not stuffed into `config["configurable"]`. When code from a blog and the current docs disagree, the docs win; when behavior surprises you, write a 10-line experiment like the ones in `19_gotchas.py`.
8. **Debugging with prints.** You already have `get_state`, `get_state_history`, `stream_mode="updates"`, and Studio/LangSmith traces. Use them first; replay a checkpoint instead of re-running the whole job.
9. **Testing only end-to-end with a real model.** Test nodes as functions, routers as functions, the graph with a scripted fake model, and *every branch* (items 3 and 5 are only catchable that way). Pin behavior you rely on in the library itself, as this lesson does.

## Checklist to paste into a code review

- [ ] No node mutates `state`; every node returns a dict whose keys are all in the state schema
- [ ] `messages` uses `add_messages`; clients send only the new turn
- [ ] Each node routes in exactly one way (`Command` **or** edges)
- [ ] Every router is `Literal`-typed and every branch has a test
- [ ] Fan-in uses `add_edge([...], "join")`
- [ ] `recursion_limit` is set explicitly; loops have a budget in state
- [ ] No bare `except` around `interrupt()`; paused threads accept only a resume
- [ ] State is serializable; dependencies live in closures/context
- [ ] `thread_id` is namespaced by an authenticated user id
- [ ] Authorization inputs (user, tenant, role) never appear in a tool's model-visible schema

## Exercises

1. Break each of items 1-12 in a copy of one of your own graphs and see which ones your existing tests catch. Add tests for the ones they do not.
2. Write a `lint_graph(compiled)` helper that fails if any `add_conditional_edges` lacks an explicit path list.
3. Implement `safe_invoke(graph, thread, payload)` that refuses fresh input to a paused thread (item 9) and returns the pending interrupt instead.
4. Make a pre-commit test that fails if any `@tool` in your project exposes a parameter named `user_id`, `tenant_id`, or `role` in its schema (item 12).
5. After your next `langgraph` upgrade, run `pytest tests/test_gotchas.py`. For every failure, find the changelog entry and update this page.
