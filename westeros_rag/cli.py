"""Command line for the ingestion pipeline and for poking at the system.

    python -m westeros_rag.cli lint                       # structural quality gate (exit 1 on problems)
    python -m westeros_rag.cli tree --role commoner       # the document tree this role is allowed to see
    python -m westeros_rag.cli ingest                     # lint -> enrich -> index (incremental), prints stats
    python -m westeros_rag.cli ask "Can a pardon restore a traitor's lands?" --role commoner
    python -m westeros_rag.cli passages "pit and gallows" --role commoner   # retrieval only: debug WHAT was fetched
"""

from __future__ import annotations

import argparse
import sys

from westeros_rag.config import ROLES, WSettings
from westeros_rag.parse import Corpus, lint
from westeros_rag.pipeline import build_pipeline


def cmd_lint(_args) -> int:
    corpus = Corpus.load(WSettings().corpus_dir)
    issues = lint(corpus)
    for i in issues:
        print(i)
    links = sum(len(s.links) for s in corpus.sections.values())
    print(f"{len(corpus.docs)} documents, {len(corpus.sections)} sections, {links} links, {len(issues)} issue(s)")
    return 1 if issues else 0


def cmd_tree(args) -> int:
    corpus, allowed = Corpus.load(WSettings().corpus_dir), set(ROLES[args.role].access_levels)
    for doc in corpus.docs.values():
        if doc.meta.get("access", "internal") not in allowed:
            continue
        print(f"{doc.doc_id}  ({doc.meta.get('domain')}, {doc.meta.get('jurisdiction')}, {doc.meta.get('access')})")
        for s in doc.sections[1:]:
            print(f"{'  ' * (s.level - 1)}- {s.title}  [{len(s.links)} links]")
    return 0


def cmd_ingest(_args) -> int:
    pipe = build_pipeline()
    print(f"corpus ok | index: {pipe.stats} | LLM enrichment calls this run: {pipe.llm_calls}")
    return 0


def cmd_ask(args) -> int:
    out = build_pipeline().ask(args.question, args.role)
    print(f"[{out['status']}] {out['answer']}")
    for c in out["citations"]:
        print(f"  cite [{c['n']}] {c['section']}  ->  {c['link']}  ({c['kind']})")
    print(f"  trace: {' > '.join(out['trace'])}\n  flags: {out.get('flags')}\n  {out.get('disclaimer', '')}")
    return 0


def cmd_passages(args) -> int:
    pipe = build_pipeline()
    for p in pipe.retriever.search(args.query, ROLES[args.role].access_levels, args.jurisdiction):
        print(f"{p.kind:<8} {p.score:.4f}  {p.unit_id}   via={p.via}\n         {p.heading_path}\n         {p.summary[:110]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="westeros_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lint").set_defaults(fn=cmd_lint)
    sub.add_parser("ingest").set_defaults(fn=cmd_ingest)
    t = sub.add_parser("tree")
    t.add_argument("--role", default="commoner", choices=ROLES)
    t.set_defaults(fn=cmd_tree)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--role", default="commoner", choices=ROLES)
    a.set_defaults(fn=cmd_ask)
    p = sub.add_parser("passages")
    p.add_argument("query")
    p.add_argument("--role", default="commoner", choices=ROLES)
    p.add_argument("--jurisdiction", default=None)
    p.set_defaults(fn=cmd_passages)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
