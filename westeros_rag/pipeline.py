"""Composition root: parse -> lint -> enrich -> index -> retrieve -> graph. Run once at startup / deploy time.

    pipe = build_pipeline()                                   # offline by default
    out  = pipe.ask("Can a pardon restore a traitor's lands?", role="commoner")
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from rag_prod.llm import get_embeddings
from westeros_rag.config import ROLES, LawContext, WSettings
from westeros_rag.enrich import Enricher
from westeros_rag.graph import build_graph
from westeros_rag.guardrails import LeakDetector
from westeros_rag.index import SyncStats, VectorIndex
from westeros_rag.offline import OfflineLawModel
from westeros_rag.parse import Corpus, Issue, lint
from westeros_rag.retrieval import HierarchicalRetriever


class CorpusError(RuntimeError):
    """Raised when the corpus fails its lint gate. Do not index a corpus you know is broken."""


def make_model():
    from common import (  # lessons/common.py (path set by rag_prod.llm)
        have_real_llm,
        make_real_chat_model,
        model_name,
    )

    if have_real_llm():
        return make_real_chat_model(timeout=30, max_retries=2), model_name()
    return OfflineLawModel(), "offline"


@dataclass
class Pipeline:
    settings: WSettings
    corpus: Corpus
    enrich: dict[str, Any]
    index: VectorIndex
    retriever: HierarchicalRetriever
    graph: CompiledStateGraph[Any, Any, Any, Any]
    stats: SyncStats
    issues: list[Issue]
    llm_calls: int

    def ask(self, question: str, role: str = "commoner", thread_id: str = "default") -> dict[str, Any]:
        ctx: LawContext = ROLES[role]
        return self.graph.invoke({"messages": [{"role": "user", "content": question}]},
                                 {"configurable": {"thread_id": f"{ctx.user_id}:{thread_id}"}}, context=ctx)


def build_pipeline(settings: WSettings | None = None, model=None, checkpointer=None, fail_on_lint: bool = True, **overrides) -> Pipeline:
    settings = replace(settings or WSettings(), **overrides)
    corpus = Corpus.load(settings.corpus_dir)
    issues = lint(corpus)
    if issues and fail_on_lint:
        raise CorpusError("corpus failed lint:\n" + "\n".join(f"  {i}" for i in issues))
    chat, model_id = (model, "custom") if model is not None else make_model()
    enricher = Enricher(chat, settings.enrich_cache, model_id)
    enrich = enricher.enrich(corpus)
    embeddings, fingerprint = get_embeddings()
    index = VectorIndex(settings, embeddings, fingerprint)
    stats = index.sync(corpus, enrich)
    retriever = HierarchicalRetriever(corpus, index, enrich, settings)
    graph = build_graph(settings, chat, retriever, LeakDetector(corpus), checkpointer or InMemorySaver())
    return Pipeline(settings, corpus, enrich, index, retriever, graph, stats, issues, enricher.llm_calls)
