"""LESSON 17 - Durable execution and fault tolerance
===================================================
Durable execution = a run can die (exception, deploy, OOM, spot-instance reclaim) and a
later process continues from the last checkpoint instead of starting over.

HOW IT WORKS
  * After every super-step the checkpointer saves a snapshot (lesson 07).
  * If several nodes run in parallel in one super-step and ONE fails, the ones that
    succeeded have their writes saved as PENDING WRITES. On resume they are not re-run.
  * Resume = `graph.invoke(None, same_thread_config)`. `None` means "no new input,
    continue from the checkpoint". A *new* input would start a new run instead.
  * The thread_id is your handle on the work. Persist it with your job record.

DURABILITY MODES  (invoke/stream(..., durability=...))
  "exit"   save only when the run ends/pauses  - fastest, least safe mid-run
  "async"  save in the background while the next step runs - fast, small crash window
  "sync"   save before the next step starts - slowest, strongest guarantee
  Rule of thumb: sync for money/irreversible work, async for chatty agents.

WHAT YOU MUST DO YOURSELF  (the checkpointer cannot save you from these)
  * A node that crashed MID-WAY re-runs from its top. Side effects before the crash happen
    again => make them IDEMPOTENT (idempotency key, upsert, "check then act").
  * One side effect per node, or per @task: the smaller the unit, the smaller the redo.
  * Bound retries; send poison inputs to a dead-letter path rather than looping.

Run:  python lessons/17_durable_execution.py
"""

from __future__ import annotations

import operator
import sqlite3
import tempfile
from pathlib import Path
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

RUNS: dict[str, int] = {}
FAIL = {"flaky_enrich": True}
PAYMENTS: dict[str, float] = {}  # stand-in for a payment provider keyed by idempotency key


class OrderState(TypedDict):
    order_id: str
    amount: float
    log: Annotated[list[str], operator.add]


def _tick(name: str) -> None:
    RUNS[name] = RUNS.get(name, 0) + 1


def validate(state: OrderState) -> dict:
    _tick("validate")
    return {"log": ["validated"]}


def enrich_tax(state: OrderState) -> dict:
    _tick("enrich_tax")
    return {"log": ["tax"]}


def flaky_enrich(state: OrderState) -> dict:
    _tick("flaky_enrich")
    if FAIL["flaky_enrich"]:
        raise RuntimeError("enrichment service down")
    return {"log": ["enriched"]}


def charge(state: OrderState) -> dict:
    """IDEMPOTENT side effect: the provider dedupes on the key, so a redo is harmless."""
    _tick("charge")
    key = f"order-{state['order_id']}"
    PAYMENTS.setdefault(key, state["amount"])  # second call with same key = no-op
    return {"log": [f"charged({key})"]}


def build(path: Path):
    """Each call builds a NEW graph + NEW connection = what a restarted process would do."""
    b = StateGraph(OrderState)
    for name, fn in [
        ("validate", validate),
        ("enrich_tax", enrich_tax),
        ("flaky_enrich", flaky_enrich),
        ("charge", charge),
    ]:
        b.add_node(name, fn)
    b.add_edge(START, "validate")
    b.add_edge("validate", "enrich_tax")  # enrich_tax and flaky_enrich run in PARALLEL
    b.add_edge("validate", "flaky_enrich")
    b.add_edge("enrich_tax", "charge")
    b.add_edge("flaky_enrich", "charge")
    b.add_edge("charge", END)
    conn = sqlite3.connect(path, check_same_thread=False)
    return b.compile(checkpointer=SqliteSaver(conn))


def crash_and_resume(db: Path) -> dict:
    cfg = {"configurable": {"thread_id": "order-42"}}
    first = {"order_id": "42", "amount": 99.0, "log": []}
    try:
        build(db).invoke(first, cfg, durability="sync")
    except RuntimeError as e:
        print("   run 1 crashed:", e)
    snap = build(db).get_state(cfg)  # brand-new graph object reads the saved checkpoint
    print("   saved checkpoint -> next:", snap.next, "| log so far:", snap.values["log"])

    FAIL["flaky_enrich"] = False  # the dependency recovered
    return build(db).invoke(None, cfg, durability="sync")  # None = resume


# ---------------------------------------------------------------- node-level retry
ATTEMPTS = {"n": 0}


def sometimes_fails(state: OrderState) -> dict:
    ATTEMPTS["n"] += 1
    if ATTEMPTS["n"] < 3:
        raise ConnectionError("blip")
    return {"log": [f"ok after {ATTEMPTS['n']} attempts"]}


def retry_graph():
    b = StateGraph(OrderState)
    b.add_node(
        "call",
        sometimes_fails,
        retry_policy=RetryPolicy(
            max_attempts=5,
            initial_interval=0.01,
            jitter=False,
            retry_on=ConnectionError,  # NEVER retry on everything: bugs and bad input won't heal
        ),
    )
    b.add_edge(START, "call")
    b.add_edge("call", END)
    return b.compile()


if __name__ == "__main__":
    db = Path(tempfile.mkdtemp()) / "durable.sqlite"

    print("1) crash in one parallel branch, resume in a 'new process'")
    out = crash_and_resume(db)
    print("   final log:", sorted(out["log"]))
    print("   node run counts:", RUNS)
    print("   -> enrich_tax and validate ran ONCE: its pending writes survived the crash;")
    print("      flaky_enrich ran twice (crash + success).")

    print("\n2) idempotency: replaying `charge` must not double charge")
    cfg = {"configurable": {"thread_id": "order-42"}}
    g = build(db)
    # rewind to just before `charge` and re-execute it (what a redo after a mid-node crash is)
    before_charge = next(s for s in g.get_state_history(cfg) if s.next == ("charge",))
    g.invoke(None, before_charge.config)
    print("   charge ran", RUNS["charge"], "times; provider holds", PAYMENTS)

    print("\n3) node-level retry for transient errors")
    print("   ", retry_graph().invoke({"order_id": "x", "amount": 0, "log": []})["log"])

    print(
        """
EXERCISES
 1. Make `charge` NON-idempotent (append to a list) and redo it. See the double charge. Fix it.
 2. Run one parallel branch that raises ValueError with retry_on=ConnectionError; confirm no retry.
 3. Compare durability="exit" vs "sync": make a node call os._exit(1) in a subprocess and
    check what the saved checkpoint contains in each mode.
 4. Add a 'dead letter' node: after 3 failed resumes route the order to manual review.
"""
    )
