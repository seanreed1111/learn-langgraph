"""LESSON 06 - Prebuilt pieces: ToolNode, tools_condition, create_agent
=====================================================================
You just hand-built the ReAct loop. In real projects you use the prebuilt parts:

  ToolNode         - executes tool calls (parallel, error handling) - replaces `run_tools`
  tools_condition  - the router - replaces `route`
  create_agent     - (from `langchain.agents`) the whole loop in one call, built on LangGraph,
                     with middleware hooks. It REPLACES the older langgraph.prebuilt.create_react_agent.

Know BOTH levels: use create_agent for speed, drop to a custom StateGraph when you
need custom control flow. Interviewers love "when would you NOT use create_agent?"

Run:  python lessons/06_prebuilt_agents.py
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from common import banner, get_model, tool_call_msg


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny and 22C in {city}."


tools = [get_weather]


def fresh_model():
    # A scripted model is single-use (it's an iterator), so build one per agent.
    return get_model(
        [tool_call_msg("get_weather", {"city": "Paris"}), AIMessage(content="Paris: sunny, 22C.")]
    ).bind_tools(tools)


# ---- Level 1: custom graph using ToolNode + tools_condition --------------
model_a = fresh_model()


def llm(state: MessagesState) -> dict:
    return {"messages": [model_a.invoke(state["messages"])]}


b = StateGraph(MessagesState)
b.add_node("llm", llm)
b.add_node("tools", ToolNode(tools))  # handles tool errors + parallel calls for you
b.add_edge(START, "llm")
b.add_conditional_edges("llm", tools_condition)  # -> "tools" or END
b.add_edge("tools", "llm")
custom_agent = b.compile()

if __name__ == "__main__":
    banner("ToolNode + tools_condition")
    out = custom_agent.invoke({"messages": [HumanMessage("Weather in Paris?")]})
    print(out["messages"][-1].content)

    # ---- Level 2: create_agent ------------------------------------------
    banner("create_agent")
    # `model` can be a model object OR a string like "anthropic:claude-sonnet-4-6".
    agent = create_agent(
        model=fresh_model(),
        tools=tools,
        system_prompt="You are a concise weather bot.",
    )
    out = agent.invoke({"messages": [HumanMessage("Weather in Paris?")]})
    print(out["messages"][-1].content)
    print(agent.get_graph().draw_mermaid())  # it's just a compiled LangGraph graph!

# NEXT-LEVEL create_agent FEATURES to read about in the docs:
#   middleware (summarize long histories, human approval for tools, PII redaction, retries),
#   response_format (structured output), context_schema, checkpointer=..., store=...
#
# EXERCISES
# 1. Add a second tool and a question needing both.
# 2. Pass checkpointer=InMemorySaver() to create_agent and continue a conversation (lesson 07).
# 3. Use response_format=<Pydantic model> to get a typed answer back.
