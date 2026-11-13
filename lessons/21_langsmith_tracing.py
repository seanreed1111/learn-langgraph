"""LESSON 21 - LangSmith tracing for LangGraph
============================================
LangSmith records every run of your graph as a TRACE: a tree of "runs" (graph -> node -> LLM call / tool
call), each with inputs, outputs, timing, errors, tags and metadata. LangGraph is instrumented already,
so turning tracing on needs no code change:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=my-agent-dev        # optional; the project is the folder traces land in

WHAT YOU GET WITHOUT WRITING ANYTHING
  * One root run per graph invocation, one child per node, grandchildren for model and tool calls.
  * `thread_id` copied into the root's metadata, so you can pull up every trace of one conversation.
  * Errors attached to the exact node that raised, plus latency per step.

WHAT YOU SHOULD ADD
  * run_name / tags / metadata per call     -> filter traces by user, tenant, feature flag, release
  * @traceable on plain Python helpers      -> they nest under the node that calls them (no-op when off)
  * a project per environment               -> dev / staging / prod never mix
  * redaction (hide_inputs / anonymizer)    -> PII never leaves your process
  * an explicit run_id when you need it     -> so feedback can be attached later (lesson 23)

OFFLINE HERE: nothing in this lesson needs a key. `FakeLangSmith` (see trace_tools.py) is a fake HTTP server
behind the REAL `langsmith.Client`, so you read the payloads LangSmith would receive. Set the three variables
above and the same code lands in the web UI.

Run:  python lessons/21_langsmith_tracing.py
"""

from __future__ import annotations

import uuid
from typing import Any

from common import banner
from demo_agent import ask, build_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt
from langsmith import traceable, tracing_context
from trace_tools import FakeLangSmith, RunRecorder
from typing_extensions import TypedDict

# ------------------------------------------------------------------------------------------------
# 1. The shape of a trace
# ------------------------------------------------------------------------------------------------


def anatomy() -> RunRecorder:
    """Run the agent once and collect the run tree in memory."""
    rec = RunRecorder()
    ask(build_agent(), "Where is order 2?", {"callbacks": [rec], "run_name": "support-chat"})
    return rec


# ------------------------------------------------------------------------------------------------
# 2. Label your runs: name, tags, metadata
# ------------------------------------------------------------------------------------------------


def labelled_config(user_id: str, release: str) -> RunnableConfig:
    """Everything here is searchable in LangSmith. Tags are flat strings; metadata is key/value."""
    return {
        "run_name": "support-chat",
        "tags": ["support", f"release:{release}"],
        "metadata": {"user_id": user_id, "release": release, "feature_flags": ["new-prompt"]},
        # `thread_id` is copied into metadata automatically. Other `configurable` keys are NOT:
        # put your own keys in `metadata` (gotcha 1 in the exercises).
        "configurable": {"thread_id": f"thread-{user_id}"},
    }


# ------------------------------------------------------------------------------------------------
# 3. Through the real client: @traceable, projects, payloads
# ------------------------------------------------------------------------------------------------


@traceable(name="normalize_question", run_type="tool")
def normalize_question(text: str) -> str:
    """Plain Python, not LangChain. @traceable makes it a child run of whatever calls it."""
    return " ".join(text.split()).strip()


@traceable(name="handle_ticket", run_type="chain")
def handle_ticket(raw_question: str) -> str:
    """A hand-written entry point wrapping the graph: the graph's trace nests UNDER this one."""
    question: str = normalize_question(raw_question)  # ty: ignore[invalid-argument-type]
    answer = str(ask(build_agent(), question, labelled_config("u-42", "2026.11"))["answer"])
    return answer


def send_through_client(project: str = "learn-langgraph-dev") -> FakeLangSmith:
    server = FakeLangSmith()
    client = server.client()
    with tracing_context(enabled=True, client=client, project_name=project):
        handle_ticket("  Where   is order 1?  ")  # ty: ignore[invalid-argument-type]
    wait_for_all_tracers()
    client.flush()
    return server


# ------------------------------------------------------------------------------------------------
# 4. Privacy: hide or redact before anything leaves the process
# ------------------------------------------------------------------------------------------------


def redact_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Runs on EVERY run's inputs before upload. Keep it pure and fast."""
    return {k: "[redacted]" for k in inputs}


def send_redacted() -> FakeLangSmith:
    server = FakeLangSmith()
    client = server.client(hide_inputs=redact_inputs, hide_outputs=True)
    with tracing_context(enabled=True, client=client, project_name="learn-langgraph-dev"):
        ask(build_agent(), "My card number is 4111 1111 1111 1111, where is order 1?")
    wait_for_all_tracers()
    client.flush()
    return server


# ------------------------------------------------------------------------------------------------
# 5. Interrupts: one conversation, MANY traces
# ------------------------------------------------------------------------------------------------


class ApprovalState(TypedDict):
    amount: int
    approved: bool


def approval_graph() -> Any:
    def request_approval(state: ApprovalState) -> dict[str, Any]:
        return {"approved": bool(interrupt(f"Approve a refund of ${state['amount']}?"))}

    builder = StateGraph(ApprovalState)  # ty: ignore[invalid-argument-type]
    builder.add_node("request_approval", request_approval)
    builder.add_edge(START, "request_approval")
    return builder.compile(checkpointer=InMemorySaver())


def interrupted_conversation() -> RunRecorder:
    """Invoke, pause at the interrupt, resume. The pause and the resume are two separate root runs."""
    rec = RunRecorder()
    graph = approval_graph()
    cfg: RunnableConfig = {"configurable": {"thread_id": "refund-7"}, "callbacks": [rec], "run_name": "refund-flow"}
    graph.invoke({"amount": 80, "approved": False}, cfg)
    graph.invoke(Command(resume=True), cfg)
    return rec


# ------------------------------------------------------------------------------------------------
# 6. Choose the run id yourself
# ------------------------------------------------------------------------------------------------


def run_with_known_id() -> tuple[uuid.UUID, RunRecorder]:
    """Pass `run_id` in the config and you know the trace id BEFORE the run finishes. Return it to your
    client app and a thumbs-up button can attach feedback to exactly this trace (lesson 23)."""
    run_id = uuid.uuid4()
    rec = RunRecorder()
    ask(build_agent(), "What is your refund policy?", {"callbacks": [rec], "run_id": run_id})
    return run_id, rec


def main() -> None:
    banner("1. The shape of a trace (what LangSmith stores for ONE invocation)")
    print(anatomy().tree())
    print("\nroot -> graph; children -> nodes; grandchildren -> the model call and the tool call.")

    banner("2. Labels: run_name, tags, metadata")
    rec = RunRecorder()
    ask(build_agent(), "hello", {**labelled_config("u-42", "2026.11"), "callbacks": [rec]})
    root = rec.roots[0]
    print("name    :", root.name)
    print("tags    :", root.tags)
    print("metadata:", {k: v for k, v in root.metadata.items() if k in {"user_id", "release", "thread_id"}})

    banner("3. Through the real client: @traceable nests, the project is set")
    server = send_through_client()
    print(server.tree())
    top = next(r for r in server.runs() if r["name"] == "handle_ticket")
    print("\nproject:", top["session_name"], "| the HTTP calls the client made:",
          sorted({f"{c.method} {c.path.split('/')[1]}" for c in server.calls}))

    banner("4. Privacy: redact before upload")
    for r in send_redacted().runs()[:2]:
        print(f"{r['name']:<12} inputs={r.get('inputs')} outputs={r.get('outputs')}")

    banner("5. Interrupt + resume = two traces, one thread")
    for r in interrupted_conversation().roots:
        print(f"trace ...{str(r.id)[-6:]}  thread_id={r.metadata['thread_id']}")

    banner("6. A run id you choose")
    rid, rec = run_with_known_id()
    print("chosen:", rid, "| stored root id:", rec.roots[0].id)

    banner("Using real LangSmith")
    print("Set LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT, then run any lesson.")
    print("Serverless/short-lived processes: call langchain_core.tracers.langchain.wait_for_all_tracers() before exit,")
    print("or set LANGCHAIN_CALLBACKS_BACKGROUND=false, or traces can be lost when the process ends.")


# EXERCISES
#  1. In `labelled_config`, move "user_id" into `configurable` only. Is it in the root's metadata? Fix it.
#  2. Wrap `ask` in a function decorated with @traceable(run_type="chain") that adds `metadata={"ab": "B"}`
#     via `langsmith.get_current_run_tree()`. Where does it show up in `send_through_client`?
#  3. Make `redact_inputs` mask only digit runs of 8+ characters (card numbers) and keep the rest readable.
#  4. Set the client's `tracing_sampling_rate=0.0` (or env LANGSMITH_TRACING_SAMPLING_RATE). How many runs arrive?
#     When is sampling the right call, and why is it bad for the evals in lesson 22?
#  5. Raise an exception inside the `tools` node (monkeypatch `lookup_order`). Which run carries the error, and
#     does the root run show it too?

if __name__ == "__main__":
    main()
