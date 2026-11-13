"""LESSON 05 - Build a tool-calling (ReAct) agent from scratch
============================================================
This is THE core agent pattern. Understand it deeply and everything else is a variation.

            +--------+  tool_calls?  +-------+
  START --> |  llm   | ------------> | tools |
            +--------+ <------------ +-------+
                |   no tool calls
                v
               END

 llm   node: call the model with the message history (model can request tools)
 tools node: execute each requested tool, append ToolMessage results
 router    : if the last AIMessage has tool_calls -> tools, else END

OFFLINE: uses a scripted fake model that "decides" to call `multiply` and then answers.
ONLINE : add an API key to .env and the same graph uses a real LLM.

Run:  python lessons/05_agent_from_scratch.py
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import END, START, MessagesState, StateGraph

from common import banner, get_model, tool_call_msg


# ------------------------------------------------------------------ tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""  # the docstring is shown to the LLM - write it well!
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


tools = [multiply, add]
tools_by_name = {t.name: t for t in tools}

# ------------------------------------------------------------------ model
# Offline script: step 1 = call multiply, step 2 = give the final answer.
model = get_model(
    [
        tool_call_msg("multiply", {"a": 6, "b": 7}),
        AIMessage(content="6 x 7 = 42."),
    ]
).bind_tools(tools)

SYSTEM = SystemMessage("You are a careful math assistant. Use tools for arithmetic.")


# ------------------------------------------------------------------ nodes
def llm(state: MessagesState) -> dict:
    return {"messages": [model.invoke([SYSTEM, *state["messages"]])]}


def run_tools(state: MessagesState) -> dict:
    last = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        output = tools_by_name[call["name"]].invoke(call["args"])
        # tool_call_id links the result to the request - APIs reject mismatches.
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}


def route(state: MessagesState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


b = StateGraph(MessagesState)
b.add_node("llm", llm)
b.add_node("tools", run_tools)
b.add_edge(START, "llm")
b.add_conditional_edges("llm", route, ["tools", END])
b.add_edge("tools", "llm")  # <-- the loop: after tools, think again
agent = b.compile()

if __name__ == "__main__":
    banner("agent trace")
    result = agent.invoke({"messages": [HumanMessage("What is 6 times 7?")]})
    for m in result["messages"]:
        m.pretty_print()

# KEY INTERVIEW POINTS
# - Why does the loop terminate? The model stops emitting tool_calls.
# - Where do you add safety? recursion_limit, max-iteration counters, tool allowlists, HITL (lesson 08).
# - Tool errors: catch exceptions in run_tools and return them as ToolMessage(status="error")
#   so the model can recover instead of the graph crashing.
#
# EXERCISES
# 1. Handle parallel tool calls (a real model may emit several in one AIMessage).
# 2. Wrap tool execution in try/except and return the error text to the model.
# 3. Add a `llm_calls` counter to state and stop after 5 calls.
