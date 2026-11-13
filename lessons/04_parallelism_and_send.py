"""LESSON 04 - Parallel branches and map-reduce with Send
=======================================================
FAN-OUT / FAN-IN: two edges leaving one node run their targets in the SAME super-step.
A node with several incoming edges waits for all of them (fan-in).

SEND (dynamic fan-out / map-reduce): when you don't know at build time how many
parallel tasks you need (e.g. one per document), a conditional edge can return a
list of `Send(node_name, private_state)` objects. Each Send runs the node once with
its own input. Results are combined with a reducer.

Run:  python lessons/04_parallelism_and_send.py
"""

import operator
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict


# ---------------------------------------------------------- static fan-out
class FanState(TypedDict):
    topic: str
    facts: Annotated[list[str], operator.add]  # reducer REQUIRED for parallel writes
    report: str


def research_wiki(state: FanState) -> dict[str, Any]:
    return {"facts": [f"wiki: {state['topic']} is interesting"]}


def research_news(state: FanState) -> dict[str, Any]:
    return {"facts": [f"news: {state['topic']} trending"]}


def write_report(state: FanState) -> dict[str, Any]:
    # Runs ONCE, after both research nodes finish.
    return {"report": " | ".join(sorted(state["facts"]))}


f = StateGraph(FanState)  # ty: ignore[invalid-argument-type]
f.add_node("research_wiki", research_wiki)
f.add_node("research_news", research_news)
f.add_node("write_report", write_report)
f.add_edge(START, "research_wiki")
f.add_edge(START, "research_news")
f.add_edge("research_wiki", "write_report")
f.add_edge("research_news", "write_report")
f.add_edge("write_report", END)
fan_graph = f.compile()


# ------------------------------------------------------ dynamic map-reduce
class OverallState(TypedDict):
    documents: list[str]
    summaries: Annotated[list[str], operator.add]
    final: str


class DocState(TypedDict):  # private input for each mapped task
    doc: str


def fan_out(state: OverallState) -> list[Send]:
    # One Send per document -> N parallel runs of "summarize".
    return [Send("summarize", {"doc": d}) for d in state["documents"]]


def summarize(state: DocState) -> dict[str, Any]:
    return {"summaries": [state["doc"][:10] + "..."]}


def combine(state: OverallState) -> dict[str, Any]:
    return {"final": f"{len(state['summaries'])} summaries: {sorted(state['summaries'])}"}


m = StateGraph(OverallState)  # ty: ignore[invalid-argument-type]
m.add_node("summarize", summarize)
m.add_node("combine", combine)
m.add_conditional_edges(START, fan_out, ["summarize"])
m.add_edge("summarize", "combine")
m.add_edge("combine", END)
map_graph = m.compile()

if __name__ == "__main__":
    print(fan_graph.invoke({"topic": "LangGraph", "facts": []}))
    docs = ["alpha document body", "bravo document body", "charlie document body"]
    print(map_graph.invoke({"documents": docs, "summaries": []}))

# WHY THIS MATTERS FOR JOBS
# Map-reduce is how production agents do "research N sources", "review N files",
# "call N tools at once". It's the answer to "how would you parallelize an agent?"
#
# EXERCISES
# 1. Add a third parallel branch. Does the report change?
# 2. Make `summarize` slow (time.sleep) and time the run vs. a sequential loop.
# 3. Pass extra private context via Send, e.g. Send("summarize", {"doc": d, "style": "terse"}).
