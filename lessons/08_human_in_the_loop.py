"""LESSON 08 - Human-in-the-loop with interrupt()
================================================
`interrupt(payload)` PAUSES the graph mid-node, saves state to the checkpointer, and
returns payload to the caller under result["__interrupt__"]. Later you resume with
`graph.invoke(Command(resume=value), same_config)` and `interrupt()` RETURNS `value`.

Requirements: a checkpointer + the same thread_id on resume.

Four classic HITL patterns (all just different uses of interrupt):
  1. APPROVE / REJECT   - gate a risky action (send email, refund, delete, deploy)
  2. EDIT               - human fixes the agent's draft/tool args before continuing
  3. REVIEW tool calls  - pause before executing tools
  4. ASK for input      - agent asks the user a clarifying question

!! THE #1 GOTCHA: on resume the WHOLE NODE RE-RUNS FROM THE TOP.
   Anything before interrupt() executes again -> keep it idempotent (no double-charging!),
   or put side effects AFTER the interrupt / in a separate node.
!! Don't wrap interrupt() in a bare try/except. Keep payloads JSON-serializable.

Run:  python lessons/08_human_in_the_loop.py
"""

from typing import Literal

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class RefundState(TypedDict):
    customer: str
    amount: float
    decision: str
    outcome: str


def draft_refund(state: RefundState) -> dict:
    return {"decision": "pending"}


def human_approval(state: RefundState) -> Command[Literal["issue_refund", "reject"]]:
    # Execution PAUSES here. The dict goes to whoever is driving the graph (a UI, Slack bot...).
    answer = interrupt(
        {
            "question": f"Approve refund of ${state['amount']:.2f} to {state['customer']}?",
            "options": ["approve", "reject"],
        }
    )
    # After resume, `answer` holds whatever the human sent.
    if answer == "approve":
        return Command(goto="issue_refund", update={"decision": "approved"})
    return Command(goto="reject", update={"decision": "rejected"})


def issue_refund(state: RefundState) -> dict:
    return {"outcome": f"Refunded ${state['amount']:.2f} to {state['customer']}"}


def reject(state: RefundState) -> dict:
    return {"outcome": "Refund rejected by reviewer"}


b = StateGraph(RefundState)
b.add_node("draft_refund", draft_refund)
b.add_node("human_approval", human_approval)
b.add_node("issue_refund", issue_refund)
b.add_node("reject", reject)
b.add_edge(START, "draft_refund")
b.add_edge("draft_refund", "human_approval")
b.add_edge("issue_refund", END)
b.add_edge("reject", END)
graph = b.compile(checkpointer=InMemorySaver())


# ------------------------------------------------ pattern 2: human EDITS state
class DraftState(TypedDict):
    draft: str
    final: str


def write_draft(state: DraftState) -> dict:
    return {"draft": "Dear customer, your order is late. Sorry!"}


def review(state: DraftState) -> dict:
    edited = interrupt({"instruction": "Edit the draft or return it unchanged", "draft": state["draft"]})
    return {"final": edited}


e = StateGraph(DraftState)
e.add_node("write_draft", write_draft)
e.add_node("review", review)
e.add_edge(START, "write_draft")
e.add_edge("write_draft", "review")
e.add_edge("review", END)
edit_graph = e.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    # ---- approve path
    cfg = {"configurable": {"thread_id": "refund-1"}}
    first = graph.invoke({"customer": "Ada", "amount": 49.5}, cfg)
    print("PAUSED. interrupt payload:", first["__interrupt__"][0].value)
    print("next node(s) waiting:", graph.get_state(cfg).next)

    # (Imagine hours pass, the process restarts, a manager clicks a button...)
    final = graph.invoke(Command(resume="approve"), cfg)
    print("RESUMED ->", final["outcome"])

    # ---- reject path on a different thread
    cfg2 = {"configurable": {"thread_id": "refund-2"}}
    graph.invoke({"customer": "Linus", "amount": 999.0}, cfg2)
    print("REJECTED ->", graph.invoke(Command(resume="reject"), cfg2)["outcome"])

    # ---- edit path
    cfg3 = {"configurable": {"thread_id": "draft-1"}}
    paused = edit_graph.invoke({}, cfg3)
    print("\nDraft shown to human:", paused["__interrupt__"][0].value["draft"])
    done = edit_graph.invoke(Command(resume="Dear customer, we apologise - 10% off your next order."), cfg3)
    print("Final:", done["final"])

# EXERCISES
# 1. Add a counter increment BEFORE interrupt() in human_approval and print it. Observe it run twice.
# 2. Add a third option "escalate" that routes to a new node.
# 3. Use interrupt() inside a tool so the agent from lesson 05 asks permission before running it.
# 4. Put a timeout/escalation path in a real app: what do you store so a stale approval can't be replayed?
