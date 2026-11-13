"""Lessons 16-18: functional API, durable execution, Pregel runtime."""
import importlib
import sys
from pathlib import Path

from langchain_core.runnables import RunnableConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lessons"))

import pytest  # noqa: E402
from langgraph.types import Command  # noqa: E402


def load(name):
    return importlib.import_module(name)


def test_functional_parallel_and_loop():
    m = load("16_functional_api")
    assert m.parallel_workflow.invoke(["ab", "cde"]) == {"lengths": [2, 3], "shouted": ["AB", "CDE"]}
    assert m.refine_loop.invoke("x")["rounds"] == 2


def test_functional_task_not_rerun_on_resume():
    m = load("16_functional_api")
    m.CALLS.pop("charge_card", None)
    cfg: RunnableConfig = {"configurable": {"thread_id": "t16"}}
    paused = m.checkout.invoke(5.0, cfg)
    assert "__interrupt__" in paused
    out = m.checkout.invoke(Command(resume="yes"), cfg)
    assert out["shipping"] == "shipped" and m.CALLS["charge_card"] == 1


def test_functional_previous_memory_and_retry():
    m = load("16_functional_api")
    cfg: RunnableConfig = {"configurable": {"thread_id": "acct16"}}
    assert [m.running_total.invoke(n, cfg)["total"] for n in (5, 10, 1)] == [5, 15, 16]
    m.FLAKY["fails_left"] = 2
    assert m.with_retry.invoke(1) == 2


def test_durable_resume_does_not_rerun_finished_parallel_branch(tmp_path):
    m = load("17_durable_execution")
    m.RUNS.clear()
    m.PAYMENTS.clear()
    m.FAIL["flaky_enrich"] = True
    out = m.crash_and_resume(tmp_path / "d.sqlite")
    assert sorted(out["log"]) == ["charged(order-42)", "enriched", "tax", "validated"]
    assert m.RUNS["enrich_tax"] == 1 and m.RUNS["validate"] == 1 and m.RUNS["flaky_enrich"] == 2


def test_retry_only_on_listed_exception():
    m = load("17_durable_execution")
    m.ATTEMPTS["n"] = 0
    assert "ok after 3 attempts" in m.retry_graph().invoke({"order_id": "x", "amount": 0, "log": []})["log"][0]


def test_pregel_snapshot_semantics_and_conflict():
    m = load("18_pregel_runtime")
    out = m.parallel_graph().invoke({"x": 0, "log": []})
    assert "a saw x=0" in out["log"] and "b saw x=0" in out["log"] and "c saw x=100" in out["log"]
    from langgraph.errors import InvalidUpdateError

    with pytest.raises(InvalidUpdateError):
        m.conflict_graph().invoke({"x": 0, "log": []})


def test_recursion_limit_counts_super_steps_and_raw_pregel():
    m = load("18_pregel_runtime")
    from langgraph.errors import GraphRecursionError

    with pytest.raises(GraphRecursionError):
        m.loop_graph().invoke({"x": 0, "log": []}, {"recursion_limit": 10})
    assert m.raw_pregel().invoke({"a": 1}) == {"b": 2, "c": 20}
