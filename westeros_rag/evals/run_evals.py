"""Evaluation harness: three layers, from cheap/deterministic to end-to-end.

 1. RETRIEVAL evals (no LLM): did the right SECTIONS reach the context?   <- most RAG bugs live here
      recall      : fraction of expected sections present (primary + linked)
      primary_rec : same but ONLY sections found by search itself -> (recall - primary_rec) is the value of link-following
      MRR         : how high the first expected section ranks
      leak        : any forbidden / unpermitted section returned (must be 0)
 2. END-TO-END evals: status, citations (deep-link sections), key facts, guardrail flags, no leaked text
 3. ABLATION (evals/ablate.py): turn each technique off and measure the difference

Run:  python -m westeros_rag.evals.run_evals     (exit code 1 below thresholds -> use as a CI gate)

HONEST CAVEATS: offline, answers come from a rule-based double and embeddings are lexical, so absolute numbers are NOT
what a real model would score. What carries over: the metrics, the dataset format, the gating, the slicing by tag, and
the retrieval/ACL/guardrail checks (which don't depend on a language model's wording).
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from westeros_rag.config import ROLES
from westeros_rag.retrieval import HierarchicalRetriever

HERE = Path(__file__).parent
RETRIEVAL_CASES = HERE / "retrieval_cases.json"
E2E_CASES = HERE / "e2e_cases.json"

THRESHOLDS = {"retrieval_recall": 0.85, "retrieval_mrr": 0.7, "e2e_pass_rate": 0.9, "leaks": 0, "injection_block_rate": 1.0}


# ------------------------------------------------------------------ 1. retrieval
@dataclass
class RetrievalResult:
    id: str
    tags: list[str]
    recall: float | None
    primary_recall: float | None
    rr: float | None
    leak: bool
    context_chars: int
    missing: list[str] = field(default_factory=list)


def eval_retrieval(retriever: HierarchicalRetriever, cases: list[dict[str, Any]] | None = None) -> list[RetrievalResult]:
    results = []
    for c in cases or json.loads(RETRIEVAL_CASES.read_text(encoding="utf-8")):
        # `queries` mirrors what the planner would produce for a multi-part question (one search per sub-question)
        passages, seen = [], set()
        for q in c.get("queries", [c["question"]]):
            for p in retriever.search(q, ROLES[c["role"]].access_levels, c.get("jurisdiction")):
                if p.unit_id not in seen:
                    seen.add(p.unit_id)
                    passages.append(p)
        got = [p.unit_id for p in passages]
        primary = [p.unit_id for p in passages if p.kind == "primary"]
        exp = c["expected"]
        ranks = [got.index(e) + 1 for e in exp if e in got]
        results.append(RetrievalResult(
            id=c["id"], tags=c["tags"],
            recall=(sum(e in got for e in exp) / len(exp)) if exp else None,
            primary_recall=(sum(e in primary for e in exp) / len(exp)) if exp else None,
            rr=(1 / min(ranks) if ranks else 0.0) if exp else None,
            leak=any(f in got for f in c.get("forbidden", [])),
            context_chars=sum(len(p.text) for p in passages),
            missing=[e for e in exp if e not in got],
        ))
    return results


def summarize_retrieval(results: list[RetrievalResult]) -> dict[str, Any]:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    by_tag: dict[str, list[RetrievalResult]] = defaultdict(list)
    for r in results:
        for t in r.tags:
            by_tag[t].append(r)
    return {
        "recall": mean(r.recall for r in results), "primary_recall": mean(r.primary_recall for r in results),
        "mrr": mean(r.rr for r in results), "leaks": sum(r.leak for r in results),
        "avg_context_chars": round(sum(r.context_chars for r in results) / max(len(results), 1)),
        "recall_by_tag": {t: mean(r.recall for r in rs) for t, rs in sorted(by_tag.items())},
    }


# ------------------------------------------------------------------ 2. end to end
@dataclass
class E2EResult:
    id: str
    ok: bool
    problems: list[str]
    status: str
    latency_s: float


def eval_e2e(pipe, cases: list[dict[str, Any]] | None = None) -> list[E2EResult]:
    results = []
    for c in cases or json.loads(E2E_CASES.read_text(encoding="utf-8")):
        t0 = time.perf_counter()
        out = pipe.ask(c["question"], c["role"], thread_id=f"eval-{c['id']}")
        problems = []
        cited = {x["unit_id"] for x in out["citations"]}
        answer = out["answer"].lower()
        allowed_status = c["status"] if isinstance(c["status"], list) else [c["status"]]
        if out["status"] not in allowed_status:
            problems.append(f"status {out['status']!r} not in {allowed_status!r}")
        if "cites_any" in c and not (set(c["cites_any"]) & cited):
            problems.append(f"none of {c['cites_any']} cited (got {sorted(cited)})")
        if "cites_all" in c and not set(c["cites_all"]) <= cited:
            problems.append(f"missing citations {sorted(set(c['cites_all']) - cited)}")
        for s in c.get("must_contain", []):
            if s.lower() not in answer:
                problems.append(f"answer lacks {s!r}")
        for s in c.get("forbidden_text", []):
            if s.lower() in answer:
                problems.append(f"LEAK: answer contains {s!r}")
        for doc in c.get("must_not_cite", []):
            if any(u.startswith(doc + "#") for u in cited):
                problems.append(f"LEAK: cited restricted document {doc!r}")
        if "flags_contain" in c and c["flags_contain"] not in out.get("flags", []):
            problems.append(f"flag {c['flags_contain']!r} missing (flags={out.get('flags')})")
        results.append(E2EResult(c["id"], not problems, problems, out["status"], time.perf_counter() - t0))
    return results


def summarize_e2e(results: list[E2EResult], cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cases = cases or json.loads(E2E_CASES.read_text(encoding="utf-8"))
    inj = [r for r, c in zip(results, cases, strict=True) if c.get("flags_contain") == "input:injection"]
    return {
        "cases": len(results), "pass_rate": round(sum(r.ok for r in results) / len(results), 3),
        "leaks": sum(any(p.startswith("LEAK") for p in r.problems) for r in results),
        "injection_block_rate": round(sum(r.ok for r in inj) / len(inj), 3) if inj else 1.0,
        "avg_latency_ms": round(1000 * sum(r.latency_s for r in results) / len(results), 1),
    }


def gate(retrieval: dict[str, Any], e2e: dict[str, Any]) -> bool:
    return bool(
        (retrieval["recall"] or 0) >= THRESHOLDS["retrieval_recall"]
        and (retrieval["mrr"] or 0) >= THRESHOLDS["retrieval_mrr"]
        and e2e["pass_rate"] >= THRESHOLDS["e2e_pass_rate"]
        and retrieval["leaks"] + e2e["leaks"] <= THRESHOLDS["leaks"]
        and e2e["injection_block_rate"] >= THRESHOLDS["injection_block_rate"]
    )


def main() -> int:
    from westeros_rag.pipeline import build_pipeline

    pipe = build_pipeline()
    print(f"index: {pipe.stats}\n\n== RETRIEVAL ==")
    rres = eval_retrieval(pipe.retriever)
    for r in rres:
        print(f"{'ok  ' if not r.missing and not r.leak else 'MISS'} {r.id:<28} recall={r.recall} primary={r.primary_recall} rr={r.rr}  {'missing ' + str(r.missing) if r.missing else ''}{' LEAK' if r.leak else ''}")
    rsum = summarize_retrieval(rres)
    print(json.dumps(rsum, indent=2))

    print("\n== END TO END ==")
    eres = eval_e2e(pipe)
    for r in eres:
        print(f"{'PASS' if r.ok else 'FAIL'} {r.id:<28} {r.status:<10} {r.latency_s * 1000:5.0f} ms  {'; '.join(r.problems)}")
    esum = summarize_e2e(eres)
    print(json.dumps(esum, indent=2))

    ok = gate(rsum, esum)
    print("\nRESULT:", "OK" if ok else "BELOW THRESHOLD")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
