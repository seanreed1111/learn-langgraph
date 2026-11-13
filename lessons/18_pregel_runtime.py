"""LESSON 18 - Under the hood: the Pregel runtime
================================================
`StateGraph(...).compile()` returns a `CompiledStateGraph`, which IS a `Pregel` object.
Everything you have used (reducers, parallelism, interrupts, recursion limit, checkpoints)
falls out of one small algorithm, Bulk Synchronous Parallel, in three phases per super-step:

  PLAN     pick the nodes whose subscribed channels were updated last step
  EXECUTE  run them all in parallel; they see the SAME snapshot, writes are buffered
  UPDATE   apply buffered writes to the channels (reducers run HERE), checkpoint, repeat
  ...stops when no node is triggered (or recursion_limit super-steps are used).

CHANNELS are the real state. Each state key is a channel with an update rule:
  LastValue               keeps one value; two writers in one step => InvalidUpdateError
  BinaryOperatorAggregate your Annotated[T, reducer]; folds many writes
  Topic                   pub/sub queue (used for Send packets)
  EphemeralValue          lives one step (used for edge triggers like `branch:to:a`)
A NODE is "subscribe to these channels, run this function, write to those channels".

WHY THIS MATTERS IN INTERVIEWS / DEBUGGING
  * "Why did my parallel nodes not see each other's updates?"  same super-step snapshot.
  * "INVALID_CONCURRENT_GRAPH_UPDATE"                          two writes to a LastValue key.
  * "Why does my recursion_limit=25 stop a 10-node loop?"        it counts super-steps, not nodes.
  * "Why does the node restart after interrupt()?"              resume re-plans that super-step.
  * Checkpoints exist per super-step, which is also the granularity of time travel.

Run:  python lessons/18_pregel_runtime.py
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langgraph.channels import LastValue
from langgraph.errors import GraphRecursionError, InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import NodeBuilder, Pregel
from typing_extensions import TypedDict


class State(TypedDict):
    x: int
    log: Annotated[list[str], operator.add]


def a(s: State) -> dict[str, Any]:
    return {"log": [f"a saw x={s['x']}"], "x": 100}


def b(s: State) -> dict[str, Any]:
    return {"log": [f"b saw x={s['x']}"]}  # same snapshot as `a`: still sees the OLD x


def c(s: State) -> dict[str, Any]:
    return {"log": [f"c saw x={s['x']}"]}  # next super-step: sees a's write


def parallel_graph():
    g = StateGraph(State)  # ty: ignore[invalid-argument-type]
    g.add_node("a", a)  # ty: ignore[invalid-argument-type]
    g.add_node("b", b)  # ty: ignore[invalid-argument-type]
    g.add_node("c", c)  # ty: ignore[invalid-argument-type]
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", "c")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    return g.compile()


# ---- a conflict: two parallel nodes write a LastValue key
def w1(s: State) -> dict[str, Any]:
    return {"x": 1}


def w2(s: State) -> dict[str, Any]:
    return {"x": 2}


def conflict_graph():
    g = StateGraph(State)  # ty: ignore[invalid-argument-type]
    g.add_node("w1", w1)  # ty: ignore[invalid-argument-type]
    g.add_node("w2", w2)  # ty: ignore[invalid-argument-type]
    g.add_edge(START, "w1")
    g.add_edge(START, "w2")
    g.add_edge("w1", END)
    g.add_edge("w2", END)
    return g.compile()


# ---- a loop that burns super-steps
def loop_node(s: State) -> dict[str, Any]:
    return {"x": s["x"] + 1}


def loop_graph():
    g = StateGraph(State)  # ty: ignore[invalid-argument-type]
    g.add_node("inc", loop_node)  # ty: ignore[invalid-argument-type]
    g.add_edge(START, "inc")
    g.add_conditional_edges("inc", lambda s: "inc" if s["x"] < 1000 else END, ["inc", END])
    return g.compile()


# ---- the same engine, hand-built, with no StateGraph at all
def raw_pregel() -> Pregel[Any]:
    plus_one = NodeBuilder().subscribe_only("a").do(lambda v: v + 1).write_to("b")
    times_ten = NodeBuilder().subscribe_only("b").do(lambda v: v * 10).write_to("c")
    return Pregel(
        nodes={"plus_one": plus_one, "times_ten": times_ten},
        channels={"a": LastValue(int), "b": LastValue(int), "c": LastValue(int)},
        input_channels=["a"],
        output_channels=["b", "c"],
    )


if __name__ == "__main__":
    g = parallel_graph()

    print("1) it is a Pregel object:", type(g).__mro__[1].__name__)
    print("   channels:", {k: type(v).__name__ for k, v in g.channels.items()})
    print("   nodes   :", list(g.nodes))

    print("\n2) super-steps: one update event per node (a and b share a step - see section 3)")
    for step in g.stream({"x": 0, "log": []}, stream_mode="updates"):
        print("   step ->", {node: upd["log"] for node, upd in step.items()})
    print("   a and b ran in the SAME super-step and both saw x=0; c ran next and saw x=100")

    print("\n3) checkpoints = super-steps")
    from langgraph.checkpoint.memory import InMemorySaver

    gc = StateGraph(State)  # ty: ignore[invalid-argument-type]
    for name, fn in [("a", a), ("b", b), ("c", c)]:
        gc.add_node(name, fn)  # ty: ignore[invalid-argument-type]
    gc.add_edge(START, "a")
    gc.add_edge(START, "b")
    gc.add_edge("a", "c")
    gc.add_edge("b", "c")
    gc.add_edge("c", END)
    gc = gc.compile(checkpointer=InMemorySaver())
    cfg: RunnableConfig = {"configurable": {"thread_id": "t"}}
    gc.invoke({"x": 0, "log": []}, cfg)
    for snap in reversed(list(gc.get_state_history(cfg))):
        assert snap.metadata is not None
        print(f"   step {snap.metadata['step']:>2}  next={snap.next}")

    print("\n4) two writers, one LastValue key, one super-step")
    try:
        conflict_graph().invoke({"x": 0, "log": []})
    except InvalidUpdateError as e:
        print("   InvalidUpdateError:", str(e).splitlines()[0])
    print("   fix: give `x` a reducer (Annotated[int, max]) or serialise the writers")

    print("\n5) recursion_limit counts super-steps")
    try:
        loop_graph().invoke({"x": 0, "log": []}, {"recursion_limit": 10})
    except GraphRecursionError:
        print("   GraphRecursionError after 10 super-steps (x was far from 1000)")

    print("\n6) raw Pregel (no StateGraph):", raw_pregel().invoke({"a": 1}))
    for step in raw_pregel().stream({"a": 1}, stream_mode="updates"):
        print("   ", step)

    print(
        """
EXERCISES
 1. Add a third parallel node writing `log`; confirm the reducer folds all three.
 2. Give `x` the reducer `Annotated[int, max]` and re-run section 4. Predict the result.
 3. Use stream_mode="debug" (or "tasks") and list which tasks were planned in each step.
 4. Add a node `d` that both `c` and a new `e` feed into; predict when `d` runs (hint: a node
    with multiple incoming edges waits for ALL of them only via add_edge([..], 'd')).
 5. Extend raw_pregel with a third node and a Topic channel.
"""
    )
