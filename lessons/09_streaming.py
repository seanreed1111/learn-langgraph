"""LESSON 09 - Streaming
=======================
Users hate staring at spinners. `graph.stream()` / `astream()` yield output as it happens.

Pass version="v2" for a uniform chunk shape:  {"type": <mode>, "ns": (), "data": ...}

  values   - full state after every step
  updates  - only what each node changed          <- best for progress UIs
  messages - LLM tokens as (message_chunk, metadata) <- chat typing effect
  custom   - anything you emit with get_stream_writer() <- progress bars, tool status
  debug / checkpoints / tasks - for debugging (need a checkpointer for some)

You can pass several modes at once: stream_mode=["updates", "custom"].
For subgraphs add subgraphs=True and read chunk["ns"] to see where an event came from.

Run:  python lessons/09_streaming.py
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph

from common import banner, get_model


# ------------------------------------------------ non-LLM graph: values/updates/custom
class State(TypedDict):
    topic: str
    joke: str


def generate(state: State) -> dict:
    writer = get_stream_writer()  # emit custom events from inside any node or tool
    writer({"status": "thinking of a joke..."})
    writer({"status": "polishing..."})
    return {"joke": f"Why did the {state['topic']} cross the road?"}


def shout(state: State) -> dict:
    return {"joke": state["joke"].upper()}


b = StateGraph(State)
b.add_node("generate", generate)
b.add_node("shout", shout)
b.add_edge(START, "generate")
b.add_edge("generate", "shout")
b.add_edge("shout", END)
graph = b.compile()

# ------------------------------------------------ LLM graph: token streaming
model = get_model([AIMessage(content="LangGraph lets you build reliable agents as graphs.")])


def chat(state: MessagesState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


c = StateGraph(MessagesState)
c.add_node("chat", chat)
c.add_edge(START, "chat")
c.add_edge("chat", END)
chat_graph = c.compile()


async def async_demo() -> None:
    # Production services (FastAPI, etc.) use astream - same API, awaited.
    async for chunk in graph.astream({"topic": "chicken"}, stream_mode="updates", version="v2"):
        print("  async chunk:", chunk["data"])


if __name__ == "__main__":
    banner("values: full state each step")
    for chunk in graph.stream({"topic": "chicken"}, stream_mode="values", version="v2"):
        print(" ", chunk["data"])

    banner("updates: only the diff")
    for chunk in graph.stream({"topic": "chicken"}, stream_mode="updates", version="v2"):
        print(" ", chunk["data"])

    banner("custom + updates together")
    for chunk in graph.stream({"topic": "chicken"}, stream_mode=["updates", "custom"], version="v2"):
        print(f"  [{chunk['type']}]", chunk["data"])

    banner("messages: token streaming")
    for chunk in chat_graph.stream(
        {"messages": [HumanMessage("Tell me about LangGraph")]}, stream_mode="messages", version="v2"
    ):
        if chunk["type"] == "messages":
            msg, meta = chunk["data"]
            print(msg.content, end="|", flush=True)  # metadata["langgraph_node"] tells you which node
    print()

    banner("async")
    asyncio.run(async_demo())

# NOTE: LangGraph 1.2 also has "event streaming" (typed projections) recommended for new
# apps - see https://docs.langchain.com/oss/python/langgraph/event-streaming.md
#
# EXERCISES
# 1. Stream with subgraphs=True through the subgraph in lesson 11 and print chunk["ns"].
# 2. Filter token streaming to only one node using metadata["langgraph_node"].
# 3. Expose a graph through FastAPI + Server-Sent Events using astream.
