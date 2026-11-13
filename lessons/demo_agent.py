"""A tiny support agent used by the LangSmith lessons (21-23).

It needs to be *deterministic* (so traces and eval scores are repeatable) yet *question-dependent* (so a
dataset of different questions exercises different paths). `SupportBotModel` is a rule-based chat model:
it decides to call a tool, or to answer, from the conversation alone. It also reports `usage_metadata`
so token accounting has something to count. With a real key, swap it for `get_model()` and nothing else changes.
"""

from __future__ import annotations

import re
from typing import Any, override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

ORDERS = {1: "shipped", 2: "processing", 3: "delivered"}


@tool
def lookup_order(order_id: int) -> str:
    """Return the status of an order."""
    status = ORDERS.get(order_id)
    return f"Order {order_id} is {status}." if status is not None else f"Order {order_id} was not found."


@tool
def refund_policy() -> str:
    """Return the refund policy."""
    return "Refunds are available within 30 days of delivery."


TOOLS = [lookup_order, refund_policy]


class SupportBotModel(BaseChatModel):
    """Rule-based stand-in for a tool-calling LLM."""

    @property
    @override
    def _llm_type(self) -> str:
        return "support-bot-offline"

    @override
    def bind_tools(self, tools: Any, **kwargs: Any) -> SupportBotModel:
        return self

    @override
    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            reply = AIMessage(content=f"Here is what I found: {last.content}")
        else:
            text = str(last.content).lower()
            order = re.search(r"order\s*#?(\d+)", text)
            if order:
                reply = AIMessage(content="", tool_calls=[
                    {"name": "lookup_order", "args": {"order_id": int(order.group(1))}, "id": "call_order"}])
            elif "refund" in text:
                reply = AIMessage(content="", tool_calls=[
                    {"name": "refund_policy", "args": {}, "id": "call_refund"}])
            else:
                reply = AIMessage(content="I can help with order status and refunds. What do you need?")
        words_in = sum(len(str(m.content).split()) for m in messages)
        reply.usage_metadata = {"input_tokens": words_in, "output_tokens": max(len(str(reply.content).split()), 1),
                                "total_tokens": words_in + max(len(str(reply.content).split()), 1)}
        return ChatResult(generations=[ChatGeneration(message=reply)])


def build_agent(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Any:
    model = SupportBotModel().bind_tools(TOOLS)

    def agent(state: MessagesState) -> dict[str, Any]:
        return {"messages": [model.invoke(state["messages"])]}

    def route(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(MessagesState)  # ty: ignore[invalid-argument-type]
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def ask(agent: Any, question: str, config: Any = None) -> dict[str, Any]:
    """Run one question; return the final answer and the tools the agent called, in order."""
    state = agent.invoke({"messages": [HumanMessage(content=question)]}, config)
    msgs = state["messages"]
    tools_called = [c["name"] for m in msgs if isinstance(m, AIMessage) for c in m.tool_calls]
    return {"answer": str(msgs[-1].content), "tools": tools_called}
