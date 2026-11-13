"""LESSON 22 - Evaluating a LangGraph agent with LangSmith
=======================================================
Tracing tells you what happened. EVALUATION tells you whether it was good, and whether your last change made
it better or worse. LangSmith's model has four parts:

    DATASET     a list of examples: inputs, plus the reference outputs you expect (optional)
    TARGET      a function  inputs -> outputs  that runs YOUR graph
    EVALUATORS  functions that score one (inputs, outputs, reference_outputs) -> {"key", "score"}
    EXPERIMENT  one run of the target over the whole dataset; scores are stored and comparable across runs

    langsmith.evaluate(target, data="my-dataset", evaluators=[...], experiment_prefix="prompt-v2")

Because the target returns plain data, the same function serves unit tests, CI gates and the LangSmith UI.

WHAT TO MEASURE FOR AN AGENT (cheapest first)
  1. OUTCOME     did the final answer contain / match what it should?            (code evaluator)
  2. TRAJECTORY  did it call the right tools, in the right order, no extras?     (code evaluator)
  3. QUALITY     is the answer helpful, grounded, polite?                        (LLM-as-judge)
  4. COST/SPEED  tokens and latency per example                                  (from the trace)
  Make the TARGET return the trajectory (tool names) in its outputs. Then trajectory scoring is a plain
  comparison and does not depend on how LangSmith hands traces to evaluators.

OFFLINE HERE: the real `evaluate()` always uploads, so this lesson runs `run_experiment` (trace_tools.py), a
miniature with the SAME contract: same target, same evaluator signatures. Code written here drops into
`langsmith.evaluate` unchanged (see `run_on_langsmith`). Evaluators here are deterministic code, so scores
repeat exactly; LLM-judge scores will not, which is why you also keep the code checks.

Run:  python lessons/22_langsmith_evaluation.py
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from common import banner, get_model, have_real_llm
from demo_agent import ask, build_agent
from trace_tools import Evaluator, ExperimentResult, langsmith_configured, run_experiment

# ------------------------------------------------------------------------------------------------
# 1. The dataset. Keep it small, specific and versioned. Each row is one behavior you care about.
# ------------------------------------------------------------------------------------------------

DATASET: list[dict[str, Any]] = [
    {"inputs": {"question": "Where is order 1?"},
     "outputs": {"must_contain": ["shipped"], "tools": ["lookup_order"]}},
    {"inputs": {"question": "Where is order 2?"},
     "outputs": {"must_contain": ["processing"], "tools": ["lookup_order"]}},
    {"inputs": {"question": "Has order 3 arrived?"},
     "outputs": {"must_contain": ["delivered"], "tools": ["lookup_order"]}},
    {"inputs": {"question": "What is your refund policy?"},
     "outputs": {"must_contain": ["30 days"], "tools": ["refund_policy"]}},
    {"inputs": {"question": "Hi there"},  # a NEGATIVE example: no tool should be called
     "outputs": {"must_contain": ["order status"], "tools": []}},
    {"inputs": {"question": "Where is order 99?"},  # an edge case: unknown order, still must not crash
     "outputs": {"must_contain": ["not found"], "tools": ["lookup_order"]}},
]

# ------------------------------------------------------------------------------------------------
# 2. The target: inputs -> outputs. It returns the trajectory too.
# ------------------------------------------------------------------------------------------------


def target(inputs: dict[str, Any]) -> dict[str, Any]:
    return ask(build_agent(), inputs["question"])


# ------------------------------------------------------------------------------------------------
# 3. Evaluators. Plain functions; ask for the arguments you need BY NAME.
# ------------------------------------------------------------------------------------------------


def answer_contains(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """OUTCOME: every required phrase appears in the answer."""
    text = outputs["answer"].lower()
    ok = all(phrase.lower() in text for phrase in reference_outputs["must_contain"])
    return {"key": "answer_contains", "score": ok}


def exact_trajectory(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """TRAJECTORY: same tools, same order. Strict; relax to a set comparison if order does not matter."""
    return {"key": "exact_trajectory", "score": outputs["tools"] == reference_outputs["tools"]}


def no_redundant_calls(outputs: dict[str, Any]) -> dict[str, Any]:
    """TRAJECTORY (reference-free): never call the same tool twice for one question. Needs no labels."""
    return {"key": "no_redundant_calls", "score": len(outputs["tools"]) == len(set(outputs["tools"]))}


def make_quality_judge() -> Evaluator:
    """QUALITY. With a real model: an LLM judge returns PASS/FAIL. Offline: a labelled heuristic stand-in,
    so the lesson still runs. Either way the evaluator signature is identical."""
    if have_real_llm():
        model = get_model()

        def llm_judge(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
            prompt = ("You grade a support agent. Reply with exactly PASS or FAIL.\n"
                      f"Customer: {inputs['question']}\nAgent: {outputs['answer']}\n"
                      "PASS only if the answer is relevant, specific and polite.")
            verdict = str(model.invoke(prompt).content).strip().upper()
            return {"key": "quality", "score": verdict.startswith("PASS")}

        return llm_judge

    def heuristic_judge(outputs: dict[str, Any]) -> dict[str, Any]:  # OFFLINE STAND-IN, not a real judge
        answer = outputs["answer"].strip()
        return {"key": "quality", "score": bool(answer) and "i don't know" not in answer.lower()}

    return heuristic_judge


def evaluators() -> list[Evaluator]:
    return [answer_contains, exact_trajectory, no_redundant_calls, make_quality_judge()]


# ------------------------------------------------------------------------------------------------
# 4. The CI gate: fail the build when quality regresses
# ------------------------------------------------------------------------------------------------

THRESHOLDS = {"answer_contains": 1.0, "exact_trajectory": 1.0, "no_redundant_calls": 1.0, "quality": 0.9}


def gate(result: ExperimentResult, thresholds: Mapping[str, float] = THRESHOLDS) -> list[str]:
    """Return one message per metric below its threshold. Empty list = pass."""
    means = result.means()
    return [f"{k}: {means.get(k, 0.0):.2f} < {floor:.2f}" for k, floor in thresholds.items()
            if means.get(k, 0.0) < floor]


def regressed_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """SIMULATED REGRESSION: pretend a prompt change made the agent stop using its tools."""
    return ask(build_agent(), "hello")


# ------------------------------------------------------------------------------------------------
# 5. The same experiment on real LangSmith (needs LANGSMITH_TRACING=true + LANGSMITH_API_KEY)
# ------------------------------------------------------------------------------------------------


def run_on_langsmith(dataset_name: str = "learn-langgraph-support", prefix: str = "baseline") -> Any:
    """Upload the dataset once, then run an experiment that shows up in the Experiments tab, comparable to
    every other run on the same dataset. Not executed offline.

    Note: this path needs a LangSmith account and could not be exercised in the offline test-suite.
    """
    from langsmith import Client, evaluate

    client = Client()
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(dataset_name, description="Support-agent regression set (lesson 22)")
        client.create_examples(dataset_name=dataset_name, examples=DATASET)
    return evaluate(  # ty: ignore[no-matching-overload]
        target,                       # the same function the offline harness used
        data=dataset_name,            # a dataset name, or a list of examples
        evaluators=evaluators(),      # the same evaluator functions
        experiment_prefix=prefix,     # experiments are named <prefix>-<random>; compare them in the UI
        metadata={"agent": "support-bot", "model": "offline-rules"},
        max_concurrency=4,
        num_repetitions=1,            # raise for nondeterministic (real-model) targets, then compare means
    )


def main() -> None:
    banner("1. The dataset")
    for ex in DATASET:
        print(f"  {ex['inputs']['question']!r:<32} expect tools={ex['outputs']['tools']}")

    banner("2. Baseline experiment")
    baseline = run_experiment(target, DATASET, evaluators())
    for row in baseline.rows:
        marks = "  ".join(f"{k}={'ok' if v >= 1 else 'FAIL'}" for k, v in row.scores.items())
        print(f"  {row.inputs['question']!r:<32} {marks}")
    print("\n  means:", baseline.means())

    banner("3. The gate on the baseline")
    problems = gate(baseline)
    print("  PASS" if not problems else f"  FAIL: {problems}")

    banner("4. Simulated regression: the gate catches it")
    regressed = run_experiment(regressed_target, DATASET, evaluators())
    print("  means:", regressed.means())
    for message in gate(regressed):
        print("  GATE FAIL ->", message)
    worst = regressed.failures("answer_contains")
    print(f"  {len(worst)} of {len(regressed.rows)} examples broke; inspect their traces to see why (lesson 21).")
    print("  Note the quality heuristic still says 1.0: a weak judge misses a regression the code checks catch.")

    banner("5. On real LangSmith")
    if langsmith_configured():
        print("  Keys detected. Running run_on_langsmith() ...")
        run_on_langsmith()
        print("  Open the project's Experiments tab to compare runs.")
    else:
        print("  Offline: not uploading. With LANGSMITH_TRACING=true and LANGSMITH_API_KEY set, this lesson")
        print("  uploads the dataset and runs the same evaluators via langsmith.evaluate().")



# EXERCISES
#  1. Add an example for "Where is order 7?" and the right expectations. Does the agent pass? What does the
#     dataset need to say about unknown orders, and who decides that: the model, the tool or the policy?
#  2. Write a `latency_under` evaluator that uses a RunRecorder (lesson 21) around the target and fails
#     any example whose root run took longer than 1 second.
#  3. Relax `exact_trajectory` to ignore order and extra read-only calls. What does that cost you in precision?
#  4. With a real key, run `run_experiment(target, DATASET, evaluators())` three times. Which scores wobble?
#     Use num_repetitions in `run_on_langsmith` to average them. Why do you NEVER gate CI on one LLM-judge run?
#  5. Split DATASET into "smoke" (3 rows, every commit) and "full" (every night). What belongs in each?

if __name__ == "__main__":
    main()
