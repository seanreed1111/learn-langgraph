"""Evaluation harness - the thing that lets you change prompts/models/chunking WITHOUT fear.

Metrics (computed per case, aggregated):
  retrieval/citation hit  - did the answer cite the document that actually contains the answer?
  answer contains         - cheap deterministic check on key facts (swap for an LLM judge in real life)
  abstention accuracy     - answer when we can, abstain when we can't (and ONLY then)
  ACL LEAKS               - restricted doc cited or restricted text emitted. Target: exactly 0, always.
  latency                 - wall time per question

Run:   python -m rag_prod.evals.run_evals        (exits non-zero if thresholds fail -> use in CI)
With a real model + key in .env the SAME dataset gives real numbers; compare before/after every change.

Production tips: grow this dataset from real user questions and thumbs-down feedback; keep a frozen
"golden" set for regressions; track scores per commit; add LLM-as-judge for faithfulness/helpfulness
(LangSmith evaluators) once the deterministic checks pass.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag_prod.app import USERS

DATASET = Path(__file__).with_name("dataset.json")

THRESHOLDS = {"citation_hit_rate": 0.9, "contains_rate": 0.9, "abstain_accuracy": 1.0, "acl_leaks": 0}


@dataclass
class CaseResult:
    id: str
    ok: bool
    problems: list[str] = field(default_factory=list)
    status: str = ""
    latency_s: float = 0.0
    leak: bool = False
    cite_hit: bool | None = None
    contains: bool | None = None
    abstain_correct: bool | None = None


def run_case(graph, case: dict) -> CaseResult:
    t0 = time.perf_counter()
    out = graph.invoke(
        {"messages": [{"role": "user", "content": case["question"]}]},
        {"configurable": {"thread_id": f"eval-{case['id']}"}},
        context=USERS[case["user"]],
    )
    res = CaseResult(id=case["id"], ok=True, status=out["status"], latency_s=time.perf_counter() - t0)
    cited = {c["source"] for c in out["citations"]}
    answer = out["answer"].lower()

    def fail(msg: str) -> None:
        res.ok = False
        res.problems.append(msg)

    if "expect_sources" in case:
        res.cite_hit = set(case["expect_sources"]) <= cited
        if not res.cite_hit:
            fail(f"expected citation of {case['expect_sources']}, got {sorted(cited)}")
        if out["status"] != "answered":
            fail(f"expected answered, got {out['status']}")
    if "must_contain" in case:
        res.contains = all(s.lower() in answer for s in case["must_contain"])
        if not res.contains:
            fail(f"answer missing {case['must_contain']}: {out['answer'][:120]!r}")
    if case.get("should_abstain"):
        res.abstain_correct = out["status"] == "abstained"
        if not res.abstain_correct:
            fail(f"expected abstain, got {out['status']}")
    elif "expect_sources" in case:
        res.abstain_correct = out["status"] != "abstained"
    if "expect_status" in case and out["status"] != case["expect_status"]:
        fail(f"expected status {case['expect_status']}, got {out['status']}")
    if set(case.get("must_not_cite", [])) & cited or any(t.lower() in answer for t in case.get("forbidden_text", [])):
        res.leak = True
        fail("ACCESS CONTROL LEAK")
    return res


def summarize(results: list[CaseResult]) -> dict:
    def rate(values: list[bool | None]) -> float:
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else 1.0

    return {
        "cases": len(results),
        "passed": sum(r.ok for r in results),
        "citation_hit_rate": rate([r.cite_hit for r in results]),
        "contains_rate": rate([r.contains for r in results]),
        "abstain_accuracy": rate([r.abstain_correct for r in results]),
        "acl_leaks": sum(r.leak for r in results),
        "avg_latency_s": round(sum(r.latency_s for r in results) / max(len(results), 1), 3),
    }


def run_all(graph) -> tuple[list[CaseResult], dict]:
    results = [run_case(graph, c) for c in json.loads(DATASET.read_text(encoding="utf-8"))]
    return results, summarize(results)


def thresholds_ok(summary: dict) -> bool:
    return (
        summary["citation_hit_rate"] >= THRESHOLDS["citation_hit_rate"]
        and summary["contains_rate"] >= THRESHOLDS["contains_rate"]
        and summary["abstain_accuracy"] >= THRESHOLDS["abstain_accuracy"]
        and summary["acl_leaks"] <= THRESHOLDS["acl_leaks"]
    )


def main() -> int:
    from rag_prod.app import build_app

    graph, _, stats = build_app()
    print(f"index: {stats}\n")
    results, summary = run_all(graph)
    for r in results:
        print(f"{'PASS' if r.ok else 'FAIL'}  {r.id:<34} {r.status:<10} {r.latency_s * 1000:6.0f} ms  {'; '.join(r.problems)}")
    print("\n" + json.dumps(summary, indent=2))
    ok = thresholds_ok(summary)
    print("\nRESULT:", "OK" if ok else "BELOW THRESHOLD")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
