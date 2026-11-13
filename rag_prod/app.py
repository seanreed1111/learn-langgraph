"""Composition root: wires settings -> models -> index -> retriever -> graph.
Everything that touches the outside world is created here, once, and injected."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from rag_prod.config import Settings
from rag_prod.graph import RagContext, build_graph
from rag_prod.ingest import Index, SyncStats
from rag_prod.llm import get_chat_model, get_embeddings
from rag_prod.retrieval import HybridRetriever

# Demo identities. In production the API layer authenticates (OIDC/JWT/API gateway) and maps the
# caller to groups -> access levels. Never trust a client-supplied access level.
USERS = {
    "anon": RagContext("anon", ("public",)),
    "alice": RagContext("alice", ("public", "internal")),
    "bob": RagContext("bob", ("public", "internal", "hr")),
    "cara": RagContext("cara", ("public", "internal", "hr", "engineering")),
}


def build_app(settings: Settings | None = None, checkpointer=None):
    """Returns (graph, index, sync_stats)."""
    settings = settings or Settings.from_env()
    embeddings, fingerprint = get_embeddings()
    index = Index(settings, embeddings, fingerprint)
    stats = index.sync()
    graph = build_graph(settings, get_chat_model(settings), HybridRetriever(index, settings), checkpointer or InMemorySaver())
    return graph, index, stats
