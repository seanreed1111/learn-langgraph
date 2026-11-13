"""LESSON 07 - Persistence: checkpointers, threads, time travel
=============================================================
A CHECKPOINTER saves a snapshot of state after every super-step. That one feature
unlocks everything "agentic" in production:

  * conversation memory      (same thread_id => state carries over between invoke() calls)
  * human-in-the-loop        (pause, wait days, resume - lesson 08)
  * fault tolerance          (crash mid-run, resume from last good step)
  * time travel / debugging  (inspect or fork any past state)

Vocabulary
  thread      = one conversation/run, identified by config["configurable"]["thread_id"]
  checkpoint  = a snapshot of state at one step in a thread
  StateSnapshot = .values (state) .next (nodes about to run) .config .metadata

Checkpointers: InMemorySaver (dev/tests), SqliteSaver (local), PostgresSaver (production).

Run:  python lessons/07_persistence_and_time_travel.py
"""

import operator
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    heard: Annotated[list[str], operator.add]
    reply: str


def listen_and_reply(state: State) -> dict:
    # `heard` accumulates across invoke() calls because the thread's state is persisted.
    return {"reply": f"I've heard {len(state['heard'])} message(s): {state['heard']}"}


def build(checkpointer):
    b = StateGraph(State)
    b.add_node("reply", listen_and_reply)
    b.add_edge(START, "reply")
    b.add_edge("reply", END)
    return b.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build(InMemorySaver())

    # --- threads isolate conversations --------------------------------------
    alice = {"configurable": {"thread_id": "alice"}}
    bob = {"configurable": {"thread_id": "bob"}}
    graph.invoke({"heard": ["hi"]}, alice)
    out = graph.invoke({"heard": ["my name is Ada"]}, alice)
    print("alice:", out["reply"])
    print("bob  :", graph.invoke({"heard": ["yo"]}, bob)["reply"])  # separate memory

    # --- inspect current state ----------------------------------------------
    snap = graph.get_state(alice)
    print("\nstate.values:", snap.values)
    print("state.next  :", snap.next, "(empty tuple = finished)")

    # --- history = every checkpoint, newest first ---------------------------
    print("\nHISTORY (newest first):")
    history = list(graph.get_state_history(alice))
    for s in history:
        print(f"  step={s.metadata['step']:>2} next={s.next} heard={s.values.get('heard')}")

    # --- time travel: fork from an old checkpoint -----------------------------
    # Find the checkpoint right after the first message was recorded.
    old = next(s for s in history if s.values.get("heard") == ["hi"] and s.next == ("reply",))
    # update_state creates a NEW checkpoint (a fork) - the original history is untouched.
    # NOTE: it applies your values THROUGH THE REDUCERS, so `heard` (operator.add) is
    # appended to, not replaced - that's why the forked run shows both messages.
    fork_cfg = graph.update_state(old.config, {"heard": ["(edited) I am Grace"]})
    forked = graph.invoke(None, fork_cfg)  # None input = "continue from this checkpoint"
    print("\nforked run :", forked["reply"])

    # --- durable persistence with SQLite ------------------------------------
    with SqliteSaver.from_conn_string("lesson07_checkpoints.sqlite") as saver:
        g2 = build(saver)
        cfg = {"configurable": {"thread_id": "persisted"}}
        print("\nsqlite     :", g2.invoke({"heard": ["remember me"]}, cfg)["reply"])
    # Re-open the file (simulating a process restart): state is still there.
    with SqliteSaver.from_conn_string("lesson07_checkpoints.sqlite") as saver:
        print("after restart:", build(saver).get_state(cfg).values)

# GOTCHAS
# - No checkpointer => no memory, no interrupts. (MISSING_CHECKPOINTER error)
# - Reusing a thread_id CONTINUES that thread. New conversation => new thread_id (use uuid4).
# - Keep state small and serializable - it is written on every step.
# - PostgresSaver needs .setup() once to create tables. Use it in production.
#
# EXERCISES
# 1. Run this twice. Does the sqlite thread keep growing? Why?
# 2. Use graph.update_state(cfg, {...}, as_node="reply") and observe which node runs next.
# 3. Add a checkpointer to the agent from lesson 05 and have a 2-turn conversation.
