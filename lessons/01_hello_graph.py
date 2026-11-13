"""LESSON 01 - Your first graph
=============================
Mental model (memorize this, it's 80% of LangGraph):

    STATE  = a shared dict-like object that flows through the graph
    NODES  = plain Python functions:  (state) -> partial state update
    EDGES  = which node runs next
    A graph = StateGraph(State) + add_node + add_edge, then .compile()

LangGraph runs in "super-steps" (inspired by Google's Pregel). In each step the
scheduled nodes run, their updates are merged into state by *reducers*, then the
next nodes are scheduled. Execution stops when nothing is left to run (END).

Run:  python lessons/01_hello_graph.py
"""

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


# 1) STATE: describe the shape of the data flowing through the graph.
class State(TypedDict):
    name: str
    greeting: str
    shout: str


# 2) NODES: each returns ONLY the keys it wants to update (not the whole state).
def make_greeting(state: State) -> dict:
    return {"greeting": f"Hello, {state['name']}!"}


def make_shout(state: State) -> dict:
    return {"shout": state["greeting"].upper()}


# 3) BUILD: register nodes, wire edges, compile.
builder = StateGraph(State)
builder.add_node("make_greeting", make_greeting)
builder.add_node("make_shout", make_shout)
builder.add_edge(START, "make_greeting")  # START is a virtual entry node
builder.add_edge("make_greeting", "make_shout")
builder.add_edge("make_shout", END)  # END is a virtual exit node

graph = builder.compile()  # compile() validates the graph and returns a runnable

if __name__ == "__main__":
    # 4) RUN: invoke returns the FINAL state.
    result = graph.invoke({"name": "Ada"})
    print("final state:", result)

    # Peek at the graph structure. Paste this into https://mermaid.live to see it.
    print(graph.get_graph().draw_mermaid())

    # Every compiled graph is a LangChain "Runnable": invoke / stream / batch / ainvoke...
    print(graph.batch([{"name": "Grace"}, {"name": "Linus"}]))

# EXERCISES
# 1. Add a third node `add_punctuation` that appends "!!!" to `shout`.
# 2. What happens if a node returns a key that isn't in State? (Try it.)
# 3. Remove the START edge. What error do you get at compile()? Read it carefully -
#    LangGraph's errors are good and interviewers like candidates who read them.
