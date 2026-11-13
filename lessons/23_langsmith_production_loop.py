"""LESSON 23 - The LangSmith production loop
===========================================
Lessons 21 and 22 gave you traces and evals. Production is the loop that connects them:

    SHIP  ->  OBSERVE  ->  COLLECT  ->  EVALUATE  ->  FIX  ->  SHIP
   traces     feedback      failures      dataset      change    gate in CI
   + metadata  + errors      -> examples   + experiment

THE PIECES THAT MAKE THE LOOP WORK
  * thread_id in metadata    one conversation = many traces (every turn, every resume). Group on it.
  * a known run_id           return it to the client; a thumbs-up/down attaches to exactly that trace.
  * feedback                 `client.create_feedback(run_id, key, score=..., comment=...)`. Humans, rules and
                             LLM judges all write the same kind of record, so you can filter on any of them.
  * errors on the node       a failing node's run carries the exception; the parents are marked failed too.
  * failures -> examples     a bad production trace becomes a dataset row; a human writes the expected output.
                             The bug can then never come back unnoticed (lesson 22 gate).
  * cost and latency         token usage and timing live on the LLM runs; track them per release.
  * LangSmith Studio         `langgraph dev` opens your graph locally with the same traces (lesson 12).

IN THE LANGSMITH UI (no code needed, worth knowing): dashboards for latency/error/cost; filters on tags,
metadata and feedback; online evaluators that score a sample of live traces; annotation queues for human
review; alerts; side-by-side comparison of experiments; the prompt playground opened from any LLM run.

OFFLINE HERE: `FakeLangSmith` and `RunRecorder` stand in for the server (see trace_tools.py). The code that
builds feedback payloads and examples is exactly what you would send; only the transport is faked.

Run:  python lessons/23_langsmith_production_loop.py
"""

from __future__ import annotations

import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import demo_agent
from common import banner
from demo_agent import ask, build_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from trace_tools import FakeLangSmith, RunRecorder, latency_ms, token_totals

# ------------------------------------------------------------------------------------------------
# 1. Serve a conversation and remember which trace belongs to which turn
# ------------------------------------------------------------------------------------------------


@dataclass
class Turn:
    run_id: uuid.UUID
    thread_id: str
    question: str
    answer: str


def serve(agent: Any, recorder: RunRecorder, thread_id: str, question: str, release: str = "2026.11") -> Turn:
    """What your API handler does: pick the run id, tag the run, return the id with the answer."""
    run_id = uuid.uuid4()
    config: RunnableConfig = {
        "run_id": run_id,
        "run_name": "support-chat",
        "callbacks": [recorder],
        "tags": [f"release:{release}"],
        "metadata": {"release": release},
        "configurable": {"thread_id": thread_id},
    }
    out = ask(agent, question, config)
    return Turn(run_id=run_id, thread_id=thread_id, question=question, answer=out["answer"])


def traces_by_thread(recorder: RunRecorder) -> dict[str, list[Any]]:
    """The offline equivalent of the UI filter  metadata.thread_id = 'x'."""
    grouped: dict[str, list[Any]] = defaultdict(list)
    for root in recorder.roots:
        grouped[str(root.metadata.get("thread_id"))].append(root)
    return grouped


# ------------------------------------------------------------------------------------------------
# 2. Feedback
# ------------------------------------------------------------------------------------------------


def feedback_payload(turn: Turn, key: str, score: float, comment: str | None = None,
                     project_id: uuid.UUID | None = None) -> dict[str, Any]:
    """The keyword arguments to `Client.create_feedback`. Same shape for a thumbs button, a rule, a judge.

    `session_id` is the PROJECT's id (LangSmith calls projects "sessions"). Recent LangSmith versions warn
    when it is missing and will require it. Look it up once at startup:
        project_id = client.read_project(project_name=os.environ["LANGSMITH_PROJECT"]).id
    """
    payload: dict[str, Any] = {"run_id": turn.run_id, "key": key, "score": score}
    if project_id:
        payload["session_id"] = project_id
    if comment is not None:
        payload["comment"] = comment
    return payload


def send_feedback(client: Any, turn: Turn, key: str, score: float, comment: str | None = None,
                  project_id: uuid.UUID | None = None) -> None:
    client.create_feedback(**feedback_payload(turn, key, score, comment, project_id))


# ------------------------------------------------------------------------------------------------
# 3. Failures -> dataset examples
# ------------------------------------------------------------------------------------------------


def turns_to_examples(turns: list[Turn], scores: dict[uuid.UUID, float], below: float = 0.5) -> list[dict[str, Any]]:
    """Turn badly rated production turns into dataset rows.

    The INPUT is real. The reference OUTPUT is left for a human, because the bad answer you observed is
    precisely what you do not want to freeze as the expected one."""
    return [
        {"inputs": {"question": t.question}, "outputs": {"must_contain": [], "tools": []},
         "metadata": {"source": "production-feedback", "bad_answer": t.answer, "run_id": str(t.run_id)}}
        for t in turns if scores.get(t.run_id, 1.0) < below
    ]


# ------------------------------------------------------------------------------------------------
# 4. Errors: which node failed?
# ------------------------------------------------------------------------------------------------


def failed_runs(recorder: RunRecorder) -> list[Any]:
    return [r for r in recorder.runs if getattr(r, "error", None)]


def innermost_failure(recorder: RunRecorder) -> Any | None:
    """Errors bubble UP: parents of a failing run are marked failed too. The culprit is the failed run
    that has no failed child."""
    failed = failed_runs(recorder)
    parents = {str(r.parent_run_id) for r in failed}
    culprits = [r for r in failed if str(r.id) not in parents]
    return culprits[0] if culprits else None


# ------------------------------------------------------------------------------------------------
# 5. Cost and latency per release
# ------------------------------------------------------------------------------------------------


def release_report(recorder: RunRecorder) -> dict[str, Any]:
    roots = recorder.roots
    latencies = sorted(ms for r in roots if (ms := latency_ms(r)) is not None)
    return {
        "traces": len(roots),
        "p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "max_ms": round(latencies[-1], 1) if latencies else None,
        **token_totals(recorder),
        "errors": len(failed_runs(recorder)),
    }


def main() -> None:
    agent = build_agent(InMemorySaver())
    rec = RunRecorder()

    banner("1. Serve three turns of two conversations")
    turns = [
        serve(agent, rec, "cust-1", "Where is order 1?"),
        serve(agent, rec, "cust-1", "And order 2?"),
        serve(agent, rec, "cust-2", "Can I get a refund?"),
    ]
    for t in turns:
        print(f"  thread={t.thread_id}  run=...{str(t.run_id)[-6:]}  {t.question!r}")
    print("\n  traces per thread:", {k: len(v) for k, v in traces_by_thread(rec).items()})

    banner("2. Feedback attaches to the run id you returned")
    server = FakeLangSmith()
    client = server.client()
    project_id = uuid.uuid4()  # real code: client.read_project(project_name=...).id
    send_feedback(client, turns[2], "user_score", 0.0, "Did not mention the 30-day limit", project_id)
    send_feedback(client, turns[0], "user_score", 1.0, project_id=project_id)
    client.flush()
    print("  payload:", feedback_payload(turns[2], "user_score", 0.0, "Did not mention the 30-day limit", project_id))
    print("  the client called:", sorted({f"{c.method} {c.path}" for c in server.calls}))

    banner("3. Bad turns become dataset examples")
    scores = {turns[2].run_id: 0.0, turns[0].run_id: 1.0}
    for ex in turns_to_examples(turns, scores):
        print("  new example:", ex["inputs"], "| needs a human-written expectation")

    banner("4. Find the failing node")
    bad = RunRecorder()
    with patch.object(demo_agent, "ORDERS", None):  # the orders "database" is gone: lookup_order will raise
        try:
            ask(build_agent(), "Where is order 1?", {"callbacks": [bad], "run_name": "support-chat"})
        except Exception as exc:
            print("  the graph raised:", type(exc).__name__)
    culprit = innermost_failure(bad)
    print("  failed runs:", [r.name for r in failed_runs(bad)])
    print("  culprit    :", culprit.name if culprit else None)

    banner("5. Cost and latency")
    print(" ", release_report(rec))
    print("  Track these per `release:` tag. A new prompt that doubles tokens shows up here before the invoice.")


# EXERCISES
#  1. Add a `helpful` LLM-judge score (lesson 22) to every turn and send it as feedback with
#     feedback_source_type="model". Why keep human and model feedback under different keys?
#  2. `turns_to_examples` leaves `must_contain` empty on purpose. Write the human review step: a function that
#     takes an example and a corrected answer, and fills in the expectation before it joins DATASET.
#  3. Extend `release_report` with a per-node latency table (use `langgraph_node` metadata). Which node is slowest?
#  4. Serve a turn that interrupts (lesson 21, section 5). How many roots share its thread_id, and which one
#     should the thumbs-up attach to?
#  5. Write the rule "alert when errors > 0 or p50 doubles vs the previous release" over two release_report()s.

if __name__ == "__main__":
    main()
