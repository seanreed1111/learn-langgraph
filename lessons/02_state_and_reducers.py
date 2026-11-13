"""LESSON 02 - State, reducers, and schemas
==========================================
Without a reducer, a node's return value OVERWRITES that key.
With a reducer, the update is MERGED with the existing value.

    messages: Annotated[list, add_messages]   # append / merge-by-id
    log:      Annotated[list, operator.add]   # plain append
    count:    int                             # overwrite (last write wins)

Reducers matter most when nodes run IN PARALLEL and write the same key: without
one you get INVALID_CONCURRENT_GRAPH_UPDATE.

Also covered: input/output schemas and private (internal) state channels.

Run:  python lessons/02_state_and_reducers.py
"""

import operator
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from typing_extensions import TypedDict


# ---------------------------------------------------------------- reducers
class State(TypedDict):
    count: int  # overwrite
    log: Annotated[list[str], operator.add]  # append


def step_a(state: State) -> dict[str, Any]:
    return {"count": state["count"] + 1, "log": ["a ran"]}


def step_b(state: State) -> dict[str, Any]:
    return {"count": state["count"] + 10, "log": ["b ran"]}


b = StateGraph(State)  # ty: ignore[invalid-argument-type]
b.add_node("a", step_a)
b.add_node("b", step_b)
b.add_edge(START, "a")
b.add_edge("a", "b")
b.add_edge("b", END)
reducer_graph = b.compile()


# ------------------------------------------------- custom reducer function
def keep_max(old: int, new: int) -> int:
    """A reducer is just (old, new) -> merged."""
    return max(old, new)


class ScoreState(TypedDict):
    best: Annotated[int, keep_max]


def score_1(state: ScoreState) -> dict[str, Any]:
    return {"best": 5}


def score_2(state: ScoreState) -> dict[str, Any]:
    return {"best": 3}


s = StateGraph(ScoreState)  # ty: ignore[invalid-argument-type]
s.add_node("score_1", score_1)
s.add_node("score_2", score_2)
s.add_edge(START, "score_1")
s.add_edge(START, "score_2")  # both run in the SAME super-step (parallel)
s.add_edge("score_1", END)
s.add_edge("score_2", END)
score_graph = s.compile()


# ------------------------------------------------ messages + add_messages
# MessagesState is a built-in state with `messages: Annotated[list[AnyMessage], add_messages]`.
# add_messages appends new messages, and if a message has an existing ID it REPLACES it.
# It also converts dicts like {"role": "user", "content": "hi"} into message objects.
class ChatState(MessagesState):  # subclass to add more keys
    turns: int


def reply(state: ChatState) -> dict[str, Any]:
    return {"messages": [AIMessage(content="Hi there!")], "turns": state.get("turns", 0) + 1}


c = StateGraph(ChatState)  # ty: ignore[invalid-argument-type]
c.add_node("reply", reply)
c.add_edge(START, "reply")
c.add_edge("reply", END)
chat_graph = c.compile()


# --------------------------------------- input / output / private schemas
class InputSchema(TypedDict):
    question: str


class OutputSchema(TypedDict):
    answer: str


class OverallState(InputSchema, OutputSchema):
    scratch: str  # internal working data the caller never sees


def think(state: InputSchema) -> dict[str, Any]:
    return {"scratch": f"thinking about {state['question']}"}


def answer(state: OverallState) -> dict[str, Any]:
    return {"answer": f"42 ({state['scratch']})"}


io = StateGraph(OverallState, input_schema=InputSchema, output_schema=OutputSchema)  # ty: ignore[invalid-argument-type]
io.add_node("think", think)
io.add_node("answer", answer)
io.add_edge(START, "think")
io.add_edge("think", "answer")
io.add_edge("answer", END)
io_graph = io.compile()

if __name__ == "__main__":
    print("reducers   :", reducer_graph.invoke({"count": 0, "log": []}))
    print("custom max :", score_graph.invoke({"best": 0}))  # 5, order-independent
    out = chat_graph.invoke({"messages": [HumanMessage("hello")]})
    print("messages   :", [(m.type, m.content) for m in out["messages"]], "turns =", out["turns"])
    # dicts are coerced into Message objects:
    out = chat_graph.invoke({"messages": [{"role": "user", "content": "from a dict"}]})
    print("dict input :", out["messages"][0].__class__.__name__)
    print("io schemas :", io_graph.invoke({"question": "meaning of life?"}))  # only `answer`

# EXERCISES
# 1. Remove the reducer on `log` and rerun. What changes?
# 2. Make two parallel nodes both write a plain `int` key (no reducer). Read the error.
# 3. Write a reducer that merges dicts (hint: lambda old, new: {**old, **new}).
# 4. Use add_messages with RemoveMessage to delete a message from history.
