"""LESSON 19 - Gotchas: what intermediate/advanced people get wrong (and how to avoid it)
=========================================================================================
Narrative: lessons/19_gotchas.md   (read it first or after; this file is the PROOF.)

Every claim in the lesson is a small function here that RUNS against the installed langgraph
and returns the facts it observed. tests/test_gotchas.py asserts those facts. That is
deliberate: advice about a fast-moving library goes stale, and a failing test is how you find
out that a gotcha was fixed (or a new one appeared) when you upgrade.

Run:  python lessons/19_gotchas.py
"""

from __future__ import annotations

import logging
import operator
import threading
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError, InvalidUpdateError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt


def cfg(thread: str, **extra: Any) -> dict:
    return {"configurable": {"thread_id": thread}, **extra}


# =========================================================== 1. mutating state in place
class ItemsState(TypedDict):
    items: list[str]


def _mutating(s: ItemsState) -> dict:
    s["items"].append("x")  # BAD: edits the object, declares no update
    return {}


def _returning(s: ItemsState) -> dict:
    return {"items": s["items"] + ["x"]}  # GOOD: new value, declared update


def gotcha_mutation() -> dict:
    out = {}
    for name, node in (("bad", _mutating), ("good", _returning)):
        g = StateGraph(ItemsState)
        g.add_node("n", node)
        g.add_edge(START, "n")
        g.add_edge("n", END)
        graph = g.compile(checkpointer=InMemorySaver())
        caller_input = {"items": []}
        returned = graph.invoke(caller_input, cfg(name))
        out[name] = {
            "returned": returned["items"],
            "caller_input_after": list(caller_input["items"]),
            "saved_checkpoint": graph.get_state(cfg(name)).values["items"],
        }
    return out


# =========================================================== 2. resending history / no reducer
def _bot(s: MessagesState) -> dict:
    return {"messages": [AIMessage(f"reply{len(s['messages'])}")]}


def _chat():
    g = StateGraph(MessagesState)
    g.add_node("bot", _bot)
    g.add_edge(START, "bot")
    g.add_edge("bot", END)
    return g.compile(checkpointer=InMemorySaver())


def gotcha_resend_history() -> dict:
    graph = _chat()
    graph.invoke({"messages": [HumanMessage("hi")]}, cfg("a"))
    # BAD: the frontend rebuilds messages from plain text: new objects, NO ids
    graph.invoke({"messages": [HumanMessage("hi"), AIMessage("reply1"), HumanMessage("more")]}, cfg("a"))
    bad = [m.content for m in graph.get_state(cfg("a")).values["messages"]]

    graph = _chat()
    graph.invoke({"messages": [HumanMessage("hi")]}, cfg("b"))
    graph.invoke({"messages": [HumanMessage("more")]}, cfg("b"))  # GOOD: send ONLY the new turn
    good = [m.content for m in graph.get_state(cfg("b")).values["messages"]]
    return {"bad": bad, "good": good}


def gotcha_overwrite_messages() -> list[str]:
    class S(TypedDict):
        messages: list  # BAD: plain list = last-writer-wins, no add_messages reducer

    g = StateGraph(S)
    g.add_node("n", lambda s: {"messages": [AIMessage("hi")]})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    out = g.compile().invoke({"messages": [HumanMessage("a"), HumanMessage("b")]})
    return [m.content for m in out["messages"]]  # history is gone


# =========================================================== 3. silent typos
class OutState(TypedDict):
    q: str
    out: str


def gotcha_typo_keys() -> dict:
    g = StateGraph(OutState)
    g.add_node("n", lambda s: {"ouput": "typo"})  # note the typo
    g.add_edge(START, "n")
    g.add_edge("n", END)
    graph = g.compile()
    silently_ignored = graph.invoke({"q": "a", "out": ""})  # no error, `out` never set
    g2 = StateGraph(OutState)
    g2.add_node("n", lambda s: "oops")
    g2.add_edge(START, "n")
    g2.add_edge("n", END)
    try:
        g2.compile().invoke({"q": "a", "out": ""})
        non_dict = "no error"
    except InvalidUpdateError:
        non_dict = "InvalidUpdateError"
    return {"typo_result": silently_ignored, "non_dict": non_dict,
            "extra_input_key": graph.invoke({"q": "a", "out": "", "bogus": 1})}


def assert_update_keys(update: dict, state_type: type) -> dict:
    """GOOD: a 4-line guard to use in node unit tests (see tests/test_gotchas.py)."""
    unknown = set(update) - set(state_type.__annotations__)
    assert not unknown, f"node returned keys that are not in state: {sorted(unknown)}"
    return update


# =========================================================== 4. Command(goto) + static edge
class LogState(TypedDict):
    log: Annotated[list[str], operator.add]


def gotcha_command_and_edge() -> list[str]:
    def a(s: LogState) -> Command[Literal["b"]]:
        return Command(goto="b", update={"log": ["a"]})

    g = StateGraph(LogState)
    g.add_node("a", a)
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_node("c", lambda s: {"log": ["c"]})
    g.add_edge(START, "a")
    g.add_edge("a", "c")  # BAD: forgot that `a` already routes itself with Command
    g.add_edge("b", END)
    g.add_edge("c", END)
    return g.compile().invoke({"log": []})["log"]  # both b AND c ran


# =========================================================== 5. router typo ends the graph quietly
def gotcha_router_typo() -> dict:
    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = Capture()
    logging.getLogger().addHandler(handler)
    try:
        g = StateGraph(OutState)
        g.add_node("n", lambda s: {})
        g.add_node("answer", lambda s: {"out": "answered"})
        g.add_edge(START, "n")
        g.add_conditional_edges("n", lambda s: "anwser")  # typo
        g.add_edge("answer", END)
        out = g.compile().invoke({"q": "a", "out": ""})
    finally:
        logging.getLogger().removeHandler(handler)
    # GOOD: the same typo with an explicit path list fails LOUDLY
    g2 = StateGraph(OutState)
    g2.add_node("n", lambda s: {})
    g2.add_node("answer", lambda s: {"out": "answered"})
    g2.add_edge(START, "n")
    g2.add_conditional_edges("n", lambda s: "anwser", ["answer", END])
    g2.add_edge("answer", END)
    try:
        g2.compile().invoke({"q": "a", "out": ""})
        with_path_list = "no error"
    except KeyError:
        with_path_list = "KeyError"
    return {"out": out["out"], "warned": any("unknown channel" in r for r in records),
            "with_path_list": with_path_list}


# =========================================================== 6. fan-in with separate edges
def gotcha_fan_in() -> dict:
    def build(list_edge: bool) -> int:
        runs: list[int] = []

        class S(TypedDict):
            x: Annotated[list[str], operator.add]

        g = StateGraph(S)
        g.add_node("fast", lambda s: {"x": ["fast"]})
        g.add_node("slow1", lambda s: {"x": ["slow1"]})
        g.add_node("slow2", lambda s: {"x": ["slow2"]})
        g.add_node("join", lambda s: runs.append(1) or {"x": ["join"]})
        g.add_edge(START, "fast")
        g.add_edge(START, "slow1")
        g.add_edge("slow1", "slow2")
        if list_edge:
            g.add_edge(["fast", "slow2"], "join")  # GOOD: wait for BOTH
        else:
            g.add_edge("fast", "join")  # BAD: join fires when EITHER arrives
            g.add_edge("slow2", "join")
        g.add_edge("join", END)
        g.compile().invoke({"x": []})
        return len(runs)

    return {"separate_edges": build(False), "list_edge": build(True)}


# =========================================================== 7. recursion limit default
def gotcha_recursion_default() -> dict:
    from langgraph._internal._config import DEFAULT_RECURSION_LIMIT

    class S(TypedDict):
        n: int

    g = StateGraph(S)
    g.add_node("inc", lambda s: {"n": s["n"] + 1})
    g.add_edge(START, "inc")
    g.add_conditional_edges("inc", lambda s: "inc" if s["n"] < 10**9 else END, ["inc", END])
    graph = g.compile()
    try:
        graph.invoke({"n": 0}, {"recursion_limit": 50})
        explicit = "no error"
    except GraphRecursionError:
        explicit = "GraphRecursionError at 50"
    return {"default": DEFAULT_RECURSION_LIMIT, "explicit": explicit}


# =========================================================== 8. try/except around interrupt
def gotcha_swallowed_interrupt() -> dict:
    class S(TypedDict):
        out: str

    def bad(s: S) -> dict:
        try:
            answer = interrupt("approve?")
        except Exception:  # BAD: GraphInterrupt is an exception; this eats the pause
            answer = "auto-approved"
        return {"out": str(answer)}

    g = StateGraph(S)
    g.add_node("n", bad)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    out = g.compile(checkpointer=InMemorySaver()).invoke({"out": ""}, cfg("t"))
    return {"paused": "__interrupt__" in out, "out": out["out"]}


# =========================================================== 9. new input on a paused thread
def gotcha_abandoned_interrupt() -> dict:
    class S(TypedDict):
        q: str
        out: str

    g = StateGraph(S)
    g.add_node("n", lambda s: {"out": f"{s['q']}:{interrupt('approve?')}"})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    graph = g.compile(checkpointer=InMemorySaver())
    c = cfg("t")
    graph.invoke({"q": "refund $5", "out": ""}, c)
    # BAD: the next HTTP request carries fresh input instead of Command(resume=...)
    graph.invoke({"q": "refund $5000", "out": ""}, c)
    pending = graph.get_state(c)
    return {"state_q": pending.values["q"], "still_paused": bool(pending.next),
            "pending_question": pending.tasks[0].interrupts[0].value}


# =========================================================== 10. un-checkpointable state
def gotcha_unserializable_state() -> dict:
    class S(TypedDict):
        conn: object

    def put_lock(s: S) -> dict:
        return {"conn": threading.Lock()}

    g = StateGraph(S)
    g.add_node("n", put_lock)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    graph = g.compile(checkpointer=InMemorySaver())
    try:
        graph.invoke({"conn": None}, cfg("t"))
        err = "no error"
    except TypeError as e:
        err = str(e)
    no_checkpointer = StateGraph(S)
    no_checkpointer.add_node("n", put_lock)
    no_checkpointer.add_edge(START, "n")
    no_checkpointer.add_edge("n", END)
    works_without = "conn" in no_checkpointer.compile().invoke({"conn": None})
    return {"error": err, "works_without_checkpointer": works_without}


# =========================================================== 11. one thread_id for everyone
def gotcha_shared_thread() -> dict:
    graph = _chat()
    graph.invoke({"messages": [HumanMessage("my password is hunter2")]}, cfg("support"))  # alice
    bob = graph.invoke({"messages": [HumanMessage("hello")]}, cfg("support"))  # BAD: same id
    leaked = any("hunter2" in str(m.content) for m in bob["messages"])
    bob_ok = graph.invoke({"messages": [HumanMessage("hello")]}, cfg("bob:support"))  # GOOD
    return {"shared_sees_alice": leaked,
            "namespaced_sees_alice": any("hunter2" in str(m.content) for m in bob_ok["messages"])}


# =========================================================== 12. LLM-supplied identity
@dataclass
class Ctx:
    user_id: str


@tool
def orders_bad(user_id: str) -> str:
    """List orders for a user."""  # BAD: the MODEL chooses whose data to read
    return f"orders of {user_id}"


@tool
def orders_good(runtime: ToolRuntime[Ctx]) -> str:
    """List the current user's orders."""  # GOOD: identity comes from trusted runtime context
    return f"orders of {runtime.context.user_id}"


def gotcha_tool_identity() -> dict:
    def fake_llm(s: MessagesState) -> dict:  # a prompt-injected model asking for someone else's data
        return {"messages": [AIMessage("", tool_calls=[
            {"name": "orders_bad", "args": {"user_id": "victim"}, "id": "1"},
            {"name": "orders_good", "args": {"user_id": "victim"}, "id": "2"}])]}

    g = StateGraph(MessagesState, context_schema=Ctx)
    g.add_node("llm", fake_llm)
    g.add_node("tools", ToolNode([orders_bad, orders_good]))
    g.add_edge(START, "llm")
    g.add_edge("llm", "tools")
    g.add_edge("tools", END)
    out = g.compile().invoke({"messages": [HumanMessage("show orders")]}, context=Ctx("alice"))
    results = {m.name: m.content for m in out["messages"][-2:]}
    return {"schema_bad": sorted(orders_bad.tool_call_schema.model_json_schema()["properties"]),
            "schema_good": sorted(orders_good.tool_call_schema.model_json_schema()["properties"]),
            "bad": results["orders_bad"], "good": results["orders_good"]}


# =========================================================== 13. compile cost (a non-problem)
def gotcha_compile_cost_ms() -> float:
    import time

    start = time.perf_counter()
    for _ in range(50):
        g = StateGraph(MessagesState)
        g.add_node("bot", _bot)
        g.add_edge(START, "bot")
        g.add_edge("bot", END)
        g.compile()
    return (time.perf_counter() - start) / 50 * 1000


ALL = [
    ("1  mutating state in place", gotcha_mutation),
    ("2a resending history", gotcha_resend_history),
    ("2b messages without a reducer", gotcha_overwrite_messages),
    ("3  silent typos in updates", gotcha_typo_keys),
    ("4  Command(goto) + static edge", gotcha_command_and_edge),
    ("5  router typo", gotcha_router_typo),
    ("6  fan-in with separate edges", gotcha_fan_in),
    ("7  recursion limit default", gotcha_recursion_default),
    ("8  try/except around interrupt", gotcha_swallowed_interrupt),
    ("9  new input on a paused thread", gotcha_abandoned_interrupt),
    ("10 un-checkpointable state", gotcha_unserializable_state),
    ("11 one thread_id for everyone", gotcha_shared_thread),
    ("12 LLM-supplied identity", gotcha_tool_identity),
    ("13 compile cost (ms, per graph)", gotcha_compile_cost_ms),
]

if __name__ == "__main__":
    for title, fn in ALL:
        print(f"\n--- {title}")
        print("   ", fn())
