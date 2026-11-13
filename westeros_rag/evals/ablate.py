"""ABLATION: switch each technique off and measure what it was worth.

    python -m westeros_rag.evals.ablate

This is how you justify (or delete) complexity. "We use hierarchical retrieval with link expansion" is a claim;
this table is evidence. Re-run it on your real corpus + real embeddings before believing any conclusion: the winners
can change with the data, the embedding model and the question mix.
"""

from __future__ import annotations

from dataclasses import replace

from westeros_rag.evals.run_evals import eval_retrieval, summarize_retrieval
from westeros_rag.retrieval import HierarchicalRetriever

CONFIGS = {
    "1 chunks only":                    dict(strategy="flat", use_parent=False, use_links=False, use_themes=False),
    "2 + parent expansion":             dict(strategy="flat", use_parent=True, use_links=False, use_themes=False),
    "3 + theme boost":                  dict(strategy="flat", use_parent=True, use_links=False, use_themes=True),
    "4 + link expansion (DEFAULT)":     dict(strategy="flat", use_parent=True, use_links=True, use_themes=True),
    "5 DEFAULT + summary stage":        dict(strategy="hier", use_parent=True, use_links=True, use_themes=True),
    "6 summary stage, no links":        dict(strategy="hier", use_parent=True, use_links=False, use_themes=True),
}


def run_ablation(pipe, configs: dict[str, dict] | None = None) -> dict[str, dict]:
    out = {}
    for name, overrides in (configs or CONFIGS).items():
        retriever = HierarchicalRetriever(pipe.corpus, pipe.index, pipe.enrich, replace(pipe.settings, **overrides))
        out[name] = summarize_retrieval(eval_retrieval(retriever))
    return out


# A parameter sweep answers "how much should the summary level count?" with data instead of a guess.
# CAUTION: tuning on the same 19 cases you report is optimistic. In real projects tune on a dev split, report on a held-out split.
SUMMARY_WEIGHT_SWEEP = {f"summary_weight={w}": dict(strategy="hier", summary_weight=w) for w in (0.0, 0.25, 0.5, 1.0, 2.0)}


def print_table(table: dict[str, dict]) -> None:
    print(f"{'configuration':<30} {'recall':>7} {'primary':>8} {'MRR':>6} {'ctx chars':>10} {'multi_hop':>10} {'regional':>9} {'acl leaks':>10}")
    for name, m in table.items():
        t = m["recall_by_tag"]
        print(f"{name:<30} {m['recall']:>7} {m['primary_recall']:>8} {m['mrr']:>6} {m['avg_context_chars']:>10} "
              f"{t.get('multi_hop'):>10} {t.get('regional'):>9} {m['leaks']:>10}")


def main() -> None:
    from westeros_rag.pipeline import build_pipeline

    pipe = build_pipeline()
    print("== ABLATION: what is each technique worth? ==")
    print_table(run_ablation(pipe))
    print("\n== SWEEP: how much should the summary level count? (weight 0 = ignore summaries) ==")
    print_table(run_ablation(pipe, SUMMARY_WEIGHT_SWEEP))


if __name__ == "__main__":
    main()
