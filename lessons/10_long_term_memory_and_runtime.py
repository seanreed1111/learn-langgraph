"""LESSON 10 - Long-term memory (Store) and runtime context
==========================================================
Two kinds of memory - interviewers ask about this constantly:

  SHORT-TERM  = state inside ONE thread, saved by the checkpointer (lesson 07).
  LONG-TERM   = facts shared ACROSS threads/sessions (user preferences, learned facts),
                saved in a STORE: a namespaced key-value database.

Store API:   store.put(namespace_tuple, key, value_dict)
             store.get(namespace_tuple, key)
             store.search(namespace_tuple, query=..., filter=...)   # query = semantic if embeddings configured
Namespace:   a tuple like ("memories", user_id) - lets you scope data per user/org.

RUNTIME: nodes can take a second parameter `runtime: Runtime[ContextSchema]` giving access to
  runtime.context  - immutable per-invocation data (user_id, db handle, model name...)
  runtime.store    - the store you passed to compile(store=...)
Pass context at call time: graph.invoke(input, config, context=Ctx(...)).
Rule of thumb: context = read-only config for THIS run; state = data that changes during the run.

InMemoryStore is for dev. In production use the Postgres store (or the LangSmith deployment's built-in store).

Run:  python lessons/10_long_term_memory_and_runtime.py
"""

import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


def remember_and_reply(state: MessagesState, runtime: Runtime[Context]) -> dict:
    user_id = runtime.context.user_id
    namespace = ("memories", user_id)  # scoped per user
    text = state["messages"][-1].content

    # WRITE: naive rule - store anything the user states as "I like X".
    if text.lower().startswith("i like "):
        runtime.store.put(namespace, str(uuid.uuid4()), {"fact": text})

    # READ: fetch everything we know about this user (search w/o query lists the namespace).
    facts = [item.value["fact"] for item in runtime.store.search(namespace)]
    return {"messages": [AIMessage(content=f"[{user_id}] known facts: {facts}")]}


b = StateGraph(MessagesState, context_schema=Context)
b.add_node("remember_and_reply", remember_and_reply)
b.add_edge(START, "remember_and_reply")
b.add_edge("remember_and_reply", END)

store = InMemoryStore()
graph = b.compile(checkpointer=InMemorySaver(), store=store)

if __name__ == "__main__":
    ada = Context(user_id="ada")

    # Thread 1: user tells us something.
    t1 = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"messages": [HumanMessage("I like Python")]}, t1, context=ada)

    # Thread 2: a BRAND NEW conversation (no shared short-term state)...
    t2 = {"configurable": {"thread_id": "t2"}}
    out = graph.invoke({"messages": [HumanMessage("what do you know about me?")]}, t2, context=ada)
    print(out["messages"][-1].content)  # ...but long-term memory still knows!

    # Another user is isolated by namespace.
    t3 = {"configurable": {"thread_id": "t3"}}
    out = graph.invoke({"messages": [HumanMessage("hello")]}, t3, context=Context("grace"))
    print(out["messages"][-1].content)

    # You can also use the store directly, outside the graph:
    print("direct get :", [i.value for i in store.search(("memories", "ada"))])

# GOING FURTHER (read the docs "Memory" + "Stores" pages)
#  - Semantic search: InMemoryStore(index={"embed": <embeddings>, "dims": 1536}) then
#    store.search(ns, query="what languages does the user like?")
#  - Let the LLM decide what to remember via memory tools (see the LangMem library)
#  - Trim/summarize long message histories so context windows don't overflow
#
# EXERCISES
# 1. Make context also carry `model_name` and read it in the node.
# 2. Add a delete: "forget everything" -> store.delete(namespace, key) for each item.
# 3. Explain out loud: why can't a checkpointer alone give cross-thread memory?
