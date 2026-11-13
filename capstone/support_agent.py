"""CAPSTONE - Customer-support agent with human approval
=======================================================
Combines everything: custom ReAct loop, tools, conditional routing, interrupt()-based
approval for a dangerous tool, checkpointer memory, streaming, retry, and bounded loops.

    START -> llm --(no tools)--> END
              |
              +--(safe tools)-----------> tools --> llm
              +--(sensitive tool)--> approval --(approved)--> tools --> llm
                                        +--(rejected)--------------> llm

Runs offline with a scripted model. Set an API key in .env to use a real LLM
(then the model decides the tool calls itself).

Run:  python capstone/support_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lessons"))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from common import get_model, tool_call_msg  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.types import Command, RetryPolicy, interrupt  # noqa: E402

ORDERS = {"123": {"item": "Headphones", "total": 79.0, "status": "delivered"}}
REFUNDS: list[dict] = []  # pretend database


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by id and return its details."""
    order = ORDERS.get(order_id)
    return str(order) if order else f"No order {order_id}"


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund `amount` dollars for an order. SENSITIVE: requires human approval."""
    REFUNDS.append({"order_id": order_id, "amount": amount})
    return f"Refunded ${amount:.2f} for order {order_id}"


TOOLS = [lookup_order, issue_refund]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
SENSITIVE = {"issue_refund"}
SYSTEM = SystemMessage(
    "You are a support agent. Look up orders before refunding. Refunds need human approval."
)


def build_agent(model=None, checkpointer=None):
    model = (model or get_model()).bind_tools(TOOLS)

    def llm(state: MessagesState) -> dict:
        return {"messages": [model.invoke([SYSTEM, *state["messages"]])]}

    def route_after_llm(state: MessagesState) -> str:
        calls = state["messages"][-1].tool_calls
        if not calls:
            return END
        return "approval" if any(c["name"] in SENSITIVE for c in calls) else "tools"

    def approval(state: MessagesState) -> Command:
        # Only the interrupt happens here - no side effects before it (node re-runs on resume).
        calls = [c for c in state["messages"][-1].tool_calls if c["name"] in SENSITIVE]
        decision = interrupt({"type": "approve_tool_calls", "calls": calls})
        if decision == "approve":
            return Command(goto="tools")
        rejected = [
            ToolMessage("Rejected by human reviewer. Do not retry.", tool_call_id=c["id"])
            for c in state["messages"][-1].tool_calls
        ]
        return Command(goto="llm", update={"messages": rejected})

    def run_tools(state: MessagesState) -> dict:
        out = []
        for c in state["messages"][-1].tool_calls:
            try:
                result = TOOLS_BY_NAME[c["name"]].invoke(c["args"])
            except Exception as exc:  # noqa: BLE001 - report to the model, don't crash
                result = f"ERROR: {exc}"
            out.append(ToolMessage(str(result), tool_call_id=c["id"]))
        return {"messages": out}

    b = StateGraph(MessagesState)
    b.add_node("llm", llm, retry_policy=RetryPolicy(max_attempts=3))
    b.add_node("approval", approval)
    b.add_node("tools", run_tools)
    b.add_edge(START, "llm")
    b.add_conditional_edges("llm", route_after_llm, ["approval", "tools", END])
    b.add_edge("tools", "llm")
    return b.compile(checkpointer=checkpointer or InMemorySaver())


def scripted_model(refund_amount: float = 79.0):
    return get_model(
        [
            tool_call_msg("lookup_order", {"order_id": "123"}, "c1"),
            tool_call_msg("issue_refund", {"order_id": "123", "amount": refund_amount}, "c2"),
            AIMessage(content="Done - your refund has been processed."),
        ]
    )


if __name__ == "__main__":
    agent = build_agent(scripted_model())
    cfg = {"configurable": {"thread_id": "ticket-001"}, "recursion_limit": 15}

    print(">> User: I want a refund for order 123")
    for chunk in agent.stream(
        {"messages": [HumanMessage("I want a refund for order 123")]},
        cfg,
        stream_mode="updates",
        version="v2",
    ):
        if chunk["type"] == "updates":
            for node, upd in chunk["data"].items():
                if node == "__interrupt__":
                    print("   [PAUSED for human]", upd[0].value)
                else:
                    print(f"   [{node}]", str(upd)[:110])

    print("\n>> Manager clicks APPROVE")
    result = agent.invoke(Command(resume="approve"), cfg)
    print("Agent:", result["messages"][-1].content)
    print("Refunds in DB:", REFUNDS)
