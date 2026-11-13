"""Tests = how you prove an agent works. Offline, deterministic, fast.

Run:  pytest -q
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lessons"))
sys.path.insert(0, str(ROOT / "capstone"))

import importlib  # noqa: E402
import os  # noqa: E402

# Force OFFLINE mode even if you have API keys in .env (blank values beat load_dotenv).
for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
    os.environ[_k] = ""

import pytest  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402


def load(name: str):
    return importlib.import_module(name)


def test_nodes_are_plain_functions():
    m = load("01_hello_graph")
    assert m.make_greeting({"name": "Ada", "greeting": "", "shout": ""}) == {"greeting": "Hello, Ada!"}


def test_reducers_accumulate():
    m = load("02_state_and_reducers")
    out = m.reducer_graph.invoke({"count": 0, "log": []})
    assert out == {"count": 11, "log": ["a ran", "b ran"]}


def test_custom_reducer_is_order_independent():
    assert load("02_state_and_reducers").score_graph.invoke({"best": 0}) == {"best": 5}


def test_map_reduce():
    out = load("04_parallelism_and_send").map_graph.invoke({"documents": ["aaa", "bbb"], "summaries": []})
    assert len(out["summaries"]) == 2


def test_agent_calls_tool_then_answers():
    m = load("05_agent_from_scratch")
    out = m.agent.invoke({"messages": [HumanMessage("6*7?")]})
    assert any(getattr(x, "tool_calls", None) for x in out["messages"])
    assert "42" in out["messages"][-1].content


def test_interrupt_and_resume():
    m = load("08_human_in_the_loop")
    cfg = {"configurable": {"thread_id": "t-test"}}
    paused = m.graph.invoke({"customer": "A", "amount": 5.0}, cfg)
    assert "__interrupt__" in paused
    assert m.graph.invoke(Command(resume="approve"), cfg)["outcome"].startswith("Refunded")


def test_agentic_rag_lesson_paths():
    rag = load("13_agentic_rag")

    def run(q):
        return rag.graph.invoke({"messages": [{"role": "user", "content": q}]})["messages"]

    greeting = run("hello!")
    assert not any(getattr(m, "tool_calls", None) for m in greeting)  # no retrieval for chit-chat

    direct = run("What does the blog say about types of reward hacking?")
    assert any(m.type == "tool" for m in direct) and "tamper" in direct[-1].content.lower()

    vague = run("Why do smart machines game their objectives?")
    assert sum(m.type == "human" for m in vague) == 2  # exactly one rewrite happened

    unknown = run("What is the capital of Australia?")
    assert sum(m.type == "human" for m in unknown) == rag.MAX_REWRITES + 1  # loop is capped
    assert "do not know" in unknown[-1].content.lower()


@pytest.mark.parametrize("decision,expect_refund", [("approve", True), ("reject", False)])
def test_capstone_approval_gate(decision, expect_refund):
    cap = load("support_agent")
    cap.REFUNDS.clear()
    agent = cap.build_agent(cap.scripted_model(), InMemorySaver())
    cfg = {"configurable": {"thread_id": f"cap-{decision}"}}
    paused = agent.invoke({"messages": [HumanMessage("refund order 123")]}, cfg)
    assert "__interrupt__" in paused
    assert cap.REFUNDS == []  # NOTHING happened before approval - the key safety property
    agent.invoke(Command(resume=decision), cfg)
    assert bool(cap.REFUNDS) is expect_refund
