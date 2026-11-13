"""Lessons 21-23: LangSmith tracing, evaluation and the production loop (all offline)."""

import importlib
import uuid
from typing import Any

import demo_agent
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from trace_tools import FakeLangSmith, RunRecorder, run_experiment


def load(name: str):
    return importlib.import_module(name)


# ---- the demo agent -------------------------------------------------------------------------------


def test_demo_agent_uses_tools_only_when_needed():
    agent = demo_agent.build_agent()
    assert demo_agent.ask(agent, "Where is order 2?") == {
        "answer": "Here is what I found: Order 2 is processing.", "tools": ["lookup_order"]}
    assert demo_agent.ask(agent, "hello")["tools"] == []


# ---- lesson 21: tracing ---------------------------------------------------------------------------


def test_trace_has_graph_node_llm_and_tool_levels():
    rec = load("21_langsmith_tracing").anatomy()
    assert [r.name for r in rec.roots] == ["support-chat"]
    assert rec.node_names() == ["agent", "tools", "agent"]
    assert [r.name for r in rec.by_type("tool")] == ["lookup_order"]
    assert len(rec.by_type("llm")) == 2


def test_labels_reach_the_root_run_and_thread_id_is_automatic():
    m = load("21_langsmith_tracing")
    rec = RunRecorder()
    demo_agent.ask(demo_agent.build_agent(), "hi", {**m.labelled_config("u-1", "r1"), "callbacks": [rec]})
    root = rec.roots[0]
    assert "release:r1" in root.tags and root.metadata["user_id"] == "u-1"
    assert root.metadata["thread_id"] == "thread-u-1"


def test_custom_configurable_keys_are_not_copied_to_metadata():
    """Gotcha: only thread_id (and a few built-ins) is mirrored; put your own keys in `metadata`."""
    rec = RunRecorder()
    cfg: RunnableConfig = {"configurable": {"thread_id": "t", "tenant": "acme"}, "callbacks": [rec]}
    demo_agent.ask(demo_agent.build_agent(InMemorySaver()), "hi", cfg)
    assert "tenant" not in rec.roots[0].metadata


def test_traceable_helper_nests_under_the_graph_and_project_is_set():
    server = load("21_langsmith_tracing").send_through_client("proj-x")
    runs = {r["name"]: r for r in server.runs()}
    assert {r["session_name"] for r in runs.values()} == {"proj-x"}
    assert runs["normalize_question"]["parent_run_id"] == runs["handle_ticket"]["id"]
    assert runs["support-chat"]["parent_run_id"] == runs["handle_ticket"]["id"]


def test_redaction_hides_inputs_and_outputs_before_upload():
    runs = load("21_langsmith_tracing").send_redacted().runs()
    assert runs and all("4111" not in str(r) for r in runs)
    assert all(r.get("outputs") in ({}, None) for r in runs)


def test_interrupt_and_resume_are_two_traces_of_one_thread():
    rec = load("21_langsmith_tracing").interrupted_conversation()
    assert len(rec.roots) == 2
    assert {r.metadata["thread_id"] for r in rec.roots} == {"refund-7"}


def test_chosen_run_id_becomes_the_root_run_id():
    run_id, rec = load("21_langsmith_tracing").run_with_known_id()
    assert rec.roots[0].id == run_id


def test_zero_sampling_rate_sends_nothing():
    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langsmith import tracing_context

    server = FakeLangSmith()
    client = server.client(tracing_sampling_rate=0.0)
    with tracing_context(enabled=True, client=client):
        demo_agent.ask(demo_agent.build_agent(), "hello")
    wait_for_all_tracers()
    client.flush()
    assert server.runs() == []


def test_fake_server_never_uses_the_network():
    server = FakeLangSmith()
    client = server.client()
    client.create_feedback(run_id=uuid.uuid4(), key="k", score=1.0, session_id=uuid.uuid4())
    client.flush()
    assert any(c.path == "/feedback" for c in server.calls)


# ---- lesson 22: evaluation ------------------------------------------------------------------------


def test_baseline_passes_the_gate():
    m = load("22_langsmith_evaluation")
    result = run_experiment(m.target, m.DATASET, m.evaluators())
    assert m.gate(result) == []
    assert result.means()["exact_trajectory"] == 1.0


def test_gate_catches_a_regression():
    m = load("22_langsmith_evaluation")
    result = run_experiment(m.regressed_target, m.DATASET, m.evaluators())
    failures = m.gate(result)
    assert any(f.startswith("answer_contains") for f in failures)
    assert any(f.startswith("exact_trajectory") for f in failures)


def test_experiment_records_a_crashing_target_as_a_zero_not_a_hole():
    def boom(inputs: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("model down")

    def always_true(outputs: dict[str, Any]) -> dict[str, Any]:
        return {"key": "ok", "score": True}

    result = run_experiment(boom, [{"inputs": {"q": 1}, "outputs": {}}], [always_true])
    assert bool(result.rows[0].error) and result.rows[0].scores == {"always_true": 0.0}


def test_evaluators_receive_only_the_arguments_they_name():
    seen: list[set[str]] = []

    def only_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
        seen.append({"outputs"})
        return {"key": "k", "score": 1}

    run_experiment(lambda i: {"a": 1}, [{"inputs": {}, "outputs": {"x": 1}}], [only_outputs])
    assert seen == [{"outputs"}]


def test_dataset_covers_a_negative_and_an_edge_case():
    m = load("22_langsmith_evaluation")
    tools = [ex["outputs"]["tools"] for ex in m.DATASET]
    assert [] in tools, "needs a no-tool (negative) example"
    assert any("not found" in ex["outputs"]["must_contain"] for ex in m.DATASET), "needs an unknown-order edge case"


# ---- lesson 23: production loop -------------------------------------------------------------------


def _serve_three():
    m = load("23_langsmith_production_loop")
    agent, rec = demo_agent.build_agent(InMemorySaver()), RunRecorder()
    turns = [m.serve(agent, rec, "c1", "Where is order 1?"), m.serve(agent, rec, "c1", "And order 2?"),
             m.serve(agent, rec, "c2", "Can I get a refund?")]
    return m, rec, turns


def test_traces_group_by_thread_and_run_ids_are_the_returned_ones():
    m, rec, turns = _serve_three()
    assert {k: len(v) for k, v in m.traces_by_thread(rec).items()} == {"c1": 2, "c2": 1}
    assert {r.id for r in rec.roots} == {t.run_id for t in turns}


def test_feedback_payload_targets_the_returned_run():
    m, _, turns = _serve_three()
    project = uuid.uuid4()
    payload = m.feedback_payload(turns[2], "user_score", 0.0, "bad", project)
    assert payload == {"run_id": turns[2].run_id, "key": "user_score", "score": 0.0,
                       "session_id": project, "comment": "bad"}


def test_only_badly_rated_turns_become_examples_and_never_freeze_the_bad_answer():
    m, _, turns = _serve_three()
    examples = m.turns_to_examples(turns, {turns[2].run_id: 0.0, turns[0].run_id: 1.0})
    assert [e["inputs"]["question"] for e in examples] == ["Can I get a refund?"]
    assert examples[0]["outputs"]["must_contain"] == []
    assert examples[0]["metadata"]["bad_answer"] == turns[2].answer


def test_innermost_failure_is_the_tool_not_its_parents():
    from unittest.mock import patch

    m = load("23_langsmith_production_loop")
    rec = RunRecorder()
    with patch.object(demo_agent, "ORDERS", None), pytest.raises(AttributeError):
        demo_agent.ask(demo_agent.build_agent(), "Where is order 1?", {"callbacks": [rec]})
    assert {r.name for r in m.failed_runs(rec)} >= {"tools", "lookup_order"}
    culprit = m.innermost_failure(rec)
    assert culprit is not None and culprit.name == "lookup_order"


def test_release_report_counts_tokens_and_traces():
    m, rec, _ = _serve_three()
    report = m.release_report(rec)
    assert report["traces"] == 3 and report["llm_calls"] == 6 and report["errors"] == 0
    assert report["input_tokens"] > 0 and report["output_tokens"] > 0


def test_resume_after_interrupt_is_a_second_trace(tmp_path):
    """Why thumbs-up must target the LAST trace of a thread, not the first."""
    m = load("21_langsmith_tracing")
    graph = m.approval_graph()
    cfg: RunnableConfig = {"configurable": {"thread_id": "x"}}
    assert "__interrupt__" in graph.invoke({"amount": 5, "approved": False}, cfg)
    assert graph.invoke(Command(resume=True), cfg)["approved"] is True
