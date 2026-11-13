"""LESSON 03 - Conditional edges, Command, and loops
===================================================
Three ways to choose the next node:

 1. add_conditional_edges(node, router_fn, [targets])   - router reads state, returns a node name
 2. return Command(goto=..., update=...) from a node    - node updates state AND routes (one step)
 3. static add_edge                                     - always go there

Cycles are what make an *agent* different from a pipeline: a graph can loop until
a condition is met. The `recursion_limit` protects you from infinite loops - hitting it raises GraphRecursionError.
!! In langgraph 1.2 the DEFAULT is 10007 super-steps (older tutorials say 25). A runaway agent loop
   will burn thousands of model calls before it stops: ALWAYS pass your own limit (lesson 19).

Run:  python lessons/03_routing_and_loops.py
"""

from typing import Any, Literal

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict


# ---------------------------------------- 1) conditional edges (a classifier)
class TriageState(TypedDict):
    ticket: str
    route: str
    result: str


def classify(state: TriageState) -> dict[str, Any]:
    t = state["ticket"].lower()
    return {"route": "billing" if "invoice" in t or "charge" in t else "tech"}


def billing(state: TriageState) -> dict[str, Any]:
    return {"result": "Routed to BILLING team"}


def tech(state: TriageState) -> dict[str, Any]:
    return {"result": "Routed to TECH support"}


def pick_team(state: TriageState) -> Literal["billing", "tech"]:
    # The Literal return annotation lets LangGraph draw the graph correctly.
    return state["route"]  # type: ignore[return-value]  # ty: ignore[invalid-explicit-return]


t = StateGraph(TriageState)  # ty: ignore[invalid-argument-type]
t.add_node("classify", classify)
t.add_node("billing", billing)
t.add_node("tech", tech)
t.add_edge(START, "classify")
t.add_conditional_edges("classify", pick_team)
t.add_edge("billing", END)
t.add_edge("tech", END)
triage_graph = t.compile()


# --------------------------------------------- 2) Command: update + goto
class LoopState(TypedDict):
    n: int


def increment(state: LoopState) -> Command[Literal["increment", "__end__"]]:
    n = state["n"] + 1
    if n >= 5:
        return Command(update={"n": n}, goto=END)  # ty: ignore[invalid-explicit-return]
    return Command(update={"n": n}, goto="increment")  # loop back to self!  # ty: ignore[invalid-explicit-return]


loop_builder = StateGraph(LoopState)  # ty: ignore[invalid-argument-type]
loop_builder.add_node("increment", increment)
loop_builder.add_edge(START, "increment")
loop_graph = loop_builder.compile()


# ---------------------------------------------- 3) runaway loop protection
def forever(state: LoopState) -> Command[Literal["forever"]]:
    return Command(update={"n": state["n"] + 1}, goto="forever")  # ty: ignore[invalid-explicit-return]


f = StateGraph(LoopState)  # ty: ignore[invalid-argument-type]
f.add_node("forever", forever)
f.add_edge(START, "forever")
runaway_graph = f.compile()

if __name__ == "__main__":
    print(triage_graph.invoke({"ticket": "I was charged twice on my invoice"}))
    print(triage_graph.invoke({"ticket": "App crashes on startup"}))
    print("loop:", loop_graph.invoke({"n": 0}))
    try:
        runaway_graph.invoke({"n": 0}, {"recursion_limit": 10})
    except GraphRecursionError as e:
        print("caught GraphRecursionError:", str(e)[:80], "...")

# WHEN TO USE WHICH
#  - conditional edge : routing logic is separate from the node doing the work (easy to test/read)
#  - Command          : the node that did the work knows where to go next (e.g. handoffs between agents)
#
# EXERCISES
# 1. Add a third route "other" and a fallback.
# 2. Build a "write -> critique -> revise" loop that stops after 3 revisions (use a counter in state).
