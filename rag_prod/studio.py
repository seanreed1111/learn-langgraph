"""Entrypoint for `langgraph dev` (LangSmith Studio + local server). See langgraph.json.

The LangGraph platform manages persistence itself, so we compile WITHOUT our own checkpointer here.
Per-request identity arrives via the `context` field of a run (RagContext).
"""

from rag_prod.config import Settings
from rag_prod.graph import build_graph
from rag_prod.ingest import Index
from rag_prod.llm import get_chat_model, get_embeddings
from rag_prod.retrieval import HybridRetriever

_settings = Settings.from_env()
_embeddings, _fingerprint = get_embeddings()
_index = Index(_settings, _embeddings, _fingerprint)
_index.sync()

graph = build_graph(_settings, get_chat_model(_settings), HybridRetriever(_index, _settings), checkpointer=None)
