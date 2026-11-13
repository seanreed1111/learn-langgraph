"""LESSON 16 - The Functional API: @entrypoint and @task
========================================================
Same engine as the Graph API (checkpoints, interrupts, streaming, retries), but you write
ordinary Python control flow (if / for / while) instead of nodes and edges.

  @task        a unit of work. Calling it returns a future; `.result()` waits for it.
               Its RESULT IS CHECKPOINTED, so on resume / replay it is NOT re-run.
  @entrypoint  the workflow function. Gets a checkpointer, store, retry policy, etc.
               The first positional arg is the input; ONE input only (use a dict for more).
  previous     inject `previous=` to read the value saved by the last run on this thread
               (short-term memory without defining state or reducers).
  entrypoint.final(value=..., save=...)   return one thing, but persist something else.

WHEN TO PICK WHICH (the interview answer)
  Graph API      : you want an explicit, visualisable topology; many branches; shared typed
                   state with reducers; Studio diagrams; teams reviewing the control flow.
  Functional API : you have an existing function/script to make durable; flow is
                   data-dependent loops; you want minimal ceremony.
  They interoperate: a graph can call an entrypoint and vice versa.

THE RULES THAT BITE (all because replay re-executes your entrypoint from the top)
  1. Side effects (HTTP POST, DB write, sending mail) belong INSIDE a @task, never bare in
     the entrypoint - bare code runs again on resume, task results are replayed from the log.
  2. Non-determinism (random, time, uuid) belongs in tasks too, so replay sees the same value.
  3. Task inputs/outputs must be serialisable.
  4. Start tasks first, then `.result()` them: that is how you get parallelism.

Run:  python lessons/16_functional_api.py
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import Command, RetryPolicy, interrupt

CALLS: dict[str, int] = {}


def count(name: str) -> None:
    CALLS[name] = CALLS.get(name, 0) + 1


# ------------------------------------------------------------ 1) tasks + parallelism
@task
def fetch_length(text: str) -> int:
    count("fetch_length")
    return len(text)


@task
def shout(text: str) -> str:
    count("shout")
    return text.upper()


@entrypoint()
def parallel_workflow(texts: list[str]) -> dict[str, Any]:
    # Fan out: start every task BEFORE waiting on any of them -> they run concurrently.
    length_futures = [fetch_length(t) for t in texts]
    shout_futures = [shout(t) for t in texts]
    return {
        "lengths": [f.result() for f in length_futures],
        "shouted": [f.result() for f in shout_futures],
    }


# ------------------------------------------------------------ 2) plain-Python control flow
@task
def draft(topic: str, round_no: int) -> str:
    count("draft")
    return f"draft#{round_no} about {topic}" + (" (polished)" if round_no >= 2 else "")


@task
def score(text: str) -> float:
    return 0.9 if "polished" in text else 0.4


@entrypoint()
def refine_loop(topic: str) -> dict[str, Any]:
    """A data-dependent loop: in the Graph API this needs a conditional edge + counter."""
    round_no, text, s = 0, "", 0.0
    while s < 0.8 and round_no < 5:  # always bound loops
        round_no += 1
        text = draft(topic, round_no).result()
        s = score(text).result()
    return {"text": text, "score": s, "rounds": round_no}


# ------------------------------------------------------------ 3) HITL + durable results
@task
def charge_card(amount: float) -> str:
    count("charge_card")  # a REAL side effect would live here, and run exactly once
    return f"charged ${amount:.2f}"


@entrypoint(checkpointer=InMemorySaver())
def checkout(amount: float) -> dict[str, str]:
    receipt = charge_card(amount).result()  # checkpointed before we pause
    answer = interrupt({"question": f"{receipt}. Ship it?"})
    return {"receipt": receipt, "shipping": "shipped" if answer == "yes" else "held"}


# ------------------------------------------------------------ 4) memory via `previous`
@entrypoint(checkpointer=InMemorySaver())
def running_total(n: int, *, previous: int | None = None) -> Any:
    total = (previous or 0) + n
    # return `total` to the caller AND save it as the next run's `previous`
    return entrypoint.final(value={"total": total}, save=total)


# ------------------------------------------------------------ 5) retry on a task
FLAKY = {"fails_left": 2}


@task(retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.01, jitter=False))
def flaky_api(x: int) -> int:
    count("flaky_api")
    if FLAKY["fails_left"] > 0:
        FLAKY["fails_left"] -= 1
        raise ConnectionError("transient")
    return x + 1


@entrypoint()
def with_retry(x: int) -> int:
    return flaky_api(x).result()


if __name__ == "__main__":
    print("1) parallel:", parallel_workflow.invoke(["ab", "cde"]))

    print("2) loop:", refine_loop.invoke("langgraph"))

    cfg = {"configurable": {"thread_id": "order-1"}}
    out = checkout.invoke(19.99, cfg)
    print("3) paused:", out["__interrupt__"][0].value)
    print("   resumed:", checkout.invoke(Command(resume="yes"), cfg))
    print("   charge_card ran", CALLS["charge_card"], "time(s)  <- not re-run on resume")

    cfg2 = {"configurable": {"thread_id": "acct"}}
    print("4) memory:", [running_total.invoke(n, cfg2)["total"] for n in (5, 10, 1)])

    print("5) retry:", with_retry.invoke(1), f"after {CALLS['flaky_api']} attempts")

    print(
        """
EXERCISES
 1. Port lesson 08's refund workflow to the Functional API. Count the lines you saved.
 2. Put `print('hi')` in the entrypoint body before interrupt(); resume and observe it twice.
    Move it into a @task - observe once. That is the replay rule.
 3. Add `uuid.uuid4()` bare in the entrypoint, resume, and watch the value change. Fix it.
 4. Call an entrypoint from a StateGraph node (they compose).
"""
    )
