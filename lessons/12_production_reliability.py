"""LESSON 12 - Production reliability: retries, caching, errors, testing
=======================================================================
What separates a demo from something a company will pay you to build.

 1. RETRY POLICY  - per-node automatic retries with backoff for flaky APIs
 2. CACHE POLICY  - skip recomputing expensive deterministic nodes (TTL-based)
 3. ERROR HANDLING IN TOOLS - return errors to the LLM instead of crashing the run
 4. RECURSION LIMIT - always bound agent loops (lesson 03)
 5. TESTING       - graphs are plain Python: test nodes alone, test whole graph with a fake model
 6. OBSERVABILITY - LangSmith tracing (set LANGSMITH_TRACING=true + LANGSMITH_API_KEY)
 7. DURABILITY    - Postgres checkpointer; make nodes idempotent; keep side-effects in their own nodes

Run:  python lessons/12_production_reliability.py
"""

from typing_extensions import TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, RetryPolicy


class State(TypedDict):
    query: str
    data: str
    result: str


# ---------------------------------------------------------------- 1) retries
ATTEMPTS = {"n": 0}


def flaky_api(state: State) -> dict:
    ATTEMPTS["n"] += 1
    if ATTEMPTS["n"] < 3:
        raise ConnectionError(f"upstream timeout (attempt {ATTEMPTS['n']})")
    return {"data": f"fresh data for {state['query']}"}


# ---------------------------------------------------------------- 2) caching
CALLS = {"expensive": 0}


def expensive(state: State) -> dict:
    CALLS["expensive"] += 1
    return {"result": state["data"].upper()}


b = StateGraph(State)
b.add_node(
    "flaky_api",
    flaky_api,
    # retry on ConnectionError only (don't retry programmer bugs like KeyError!)
    retry_policy=RetryPolicy(max_attempts=5, initial_interval=0.01, retry_on=ConnectionError),
)
b.add_node("expensive", expensive, cache_policy=CachePolicy(ttl=60))
b.add_edge(START, "flaky_api")
b.add_edge("flaky_api", "expensive")
b.add_edge("expensive", END)
graph = b.compile(cache=InMemoryCache())  # cache backend is passed at compile time


# ------------------------------------------------- 3) tool errors -> the LLM
def safe_tool_call(tool_fn, **kwargs) -> str:
    """Pattern: never let a tool exception kill the agent loop - report it to the model."""
    try:
        return str(tool_fn(**kwargs))
    except Exception as exc:  # noqa: BLE001 - deliberate: surface to the LLM
        return f"ERROR: {type(exc).__name__}: {exc}. Fix the arguments and try again."


if __name__ == "__main__":
    out = graph.invoke({"query": "weather"})
    print("after retries:", out, "| attempts =", ATTEMPTS["n"])

    # Same input again: `expensive` receives identical state, so the cache serves it.
    graph.invoke({"query": "weather"})
    graph.invoke({"query": "weather"})
    print("expensive() ran", CALLS["expensive"], "time(s) across 3 identical runs (cache hits skip it)")

    print(safe_tool_call(lambda a, b: a / b, a=1, b=0))

# 5) TESTING PATTERN (see tests/test_graphs.py):
#      - unit-test each node as a normal function: assert node(state) == {...}
#      - integration-test the graph with a scripted fake model (no network, deterministic)
#      - use InMemorySaver in tests; unique thread_id per test
#
# DEPLOYMENT CHEAT-SHEET (read: docs "Deploy" + "Application structure" + "Run a local server")
#      langgraph.json   -> {"dependencies": ["."], "graphs": {"agent": "./my_agent.py:graph"}, "env": ".env"}
#      pip install "langgraph-cli[inmem]"  then  langgraph dev   -> local server + LangSmith Studio UI
#      langgraph up / build -> Docker image; deploy on LangSmith Deployment or self-host
#      Postgres checkpointer: PostgresSaver.from_conn_string(DB_URI); saver.setup()
#
# EXERCISES
# 1. Make `retry_on` a function that retries only on HTTP 5xx errors.
# 2. Add a key_func to CachePolicy so the cache key ignores irrelevant state fields.
# 3. Add a node-level timeout using `asyncio.wait_for` in an async node.
