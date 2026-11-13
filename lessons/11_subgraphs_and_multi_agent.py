"""LESSON 11 - Subgraphs and multi-agent systems
===============================================
SUBGRAPH = a compiled graph used as a NODE in a parent graph. Use it to
  * encapsulate a team/skill ("research agent", "writer agent")
  * reuse logic, test pieces in isolation, let teams own separate pieces

Two ways to connect:
  A) add_node("name", compiled_subgraph)       - parent and child SHARE state keys (simplest)
  B) call subgraph.invoke(...) inside a node   - different schemas; you map state in/out yourself

MULTI-AGENT PATTERNS
  SUPERVISOR : a central router decides which worker acts next, workers report back.
  HANDOFF    : agents pass control directly with Command(goto="other_agent").
  (Swarm / hierarchical teams are combinations of the two.)

Here the supervisor is rule-based so the lesson runs offline. In real systems the
supervisor is an LLM call with structured output, e.g.
    router = model.with_structured_output(Route)   # Route(next: Literal["researcher","writer","FINISH"])

Run:  python lessons/11_subgraphs_and_multi_agent.py
"""

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict


class TeamState(TypedDict):
    task: str
    notes: str
    draft: str
    steps: list[str]


# -------------------------------------------- a reusable SUBGRAPH (researcher)
def search(state: TeamState) -> dict[str, Any]:
    return {"notes": f"3 sources found about '{state['task']}'"}


def summarize(state: TeamState) -> dict[str, Any]:
    return {"notes": state["notes"] + " -> summarized"}


rb = StateGraph(TeamState)  # ty: ignore[invalid-argument-type]
rb.add_node("search", search)
rb.add_node("summarize", summarize)
rb.add_edge(START, "search")
rb.add_edge("search", "summarize")
rb.add_edge("summarize", END)
researcher = rb.compile()  # <- a compiled graph...


def writer(state: TeamState) -> dict[str, Any]:
    return {"draft": f"Article based on: {state['notes']}", "steps": state["steps"] + ["writer"]}


# ------------------------------------------------------------- supervisor
def supervisor(state: TeamState) -> Command[Literal["researcher", "writer", "__end__"]]:
    # Decide who goes next based on what's missing. (An LLM would do this in real life.)
    if not state.get("notes"):
        return Command(goto="researcher", update={"steps": state["steps"] + ["supervisor->researcher"]})  # ty: ignore[invalid-explicit-return]
    if not state.get("draft"):
        return Command(goto="writer", update={"steps": state["steps"] + ["supervisor->writer"]})  # ty: ignore[invalid-explicit-return]
    return Command(goto=END, update={"steps": state["steps"] + ["supervisor->done"]})  # ty: ignore[invalid-explicit-return]


b = StateGraph(TeamState)  # ty: ignore[invalid-argument-type]
b.add_node("supervisor", supervisor)
b.add_node("researcher", researcher)  # <- ...used directly as a node
b.add_node("writer", writer)
b.add_edge(START, "supervisor")
b.add_edge("researcher", "supervisor")  # workers always report back
b.add_edge("writer", "supervisor")
team = b.compile()

if __name__ == "__main__":
    out = team.invoke({"task": "LangGraph in production", "notes": "", "draft": "", "steps": []})
    print("steps:", out["steps"])
    print("draft:", out["draft"])

    print("\nsubgraph-aware streaming (note the namespaces):")
    for chunk in team.stream(
        {"task": "x", "notes": "", "draft": "", "steps": []},
        stream_mode="updates",
        subgraphs=True,
        version="v2",
    ):
        print(" ns=", chunk["ns"], "->", list(chunk["data"].keys()))  # ty: ignore[missing-union-attribute]

# TRADE-OFFS TO BE ABLE TO DISCUSS
#  - More agents != better. Each hop costs latency + tokens + failure surface. Start with ONE agent
#    with good tools; split only when prompts/tools/contexts conflict or you need independent teams.
#  - Supervisor: easy to reason about/observe, but the router is a bottleneck.
#  - Handoffs: flexible, but harder to debug - rely on tracing (LangSmith).
#
# EXERCISES
# 1. Give the subgraph a DIFFERENT state schema and call it from a wrapper node (pattern B).
# 2. Replace the rule-based supervisor with an LLM + with_structured_output (needs an API key).
# 3. Add a `reviewer` agent that can send work back to `writer` (a critique loop).
