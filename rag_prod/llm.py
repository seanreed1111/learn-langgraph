"""Model factory. One place decides which chat model / embeddings the service uses.

Real mode: any provider key in the environment -> init_chat_model with a timeout and retries.
Offline mode: deterministic test doubles so CI and evals need no network or money.
"""

from __future__ import annotations

import os
import sys

from rag_prod.config import ROOT, Settings

sys.path.insert(0, str(ROOT / "lessons"))
from common import have_real_llm, make_real_chat_model
from offline_llm import HashingEmbeddings, OfflineChatModel


def get_chat_model(settings: Settings):
    if have_real_llm():
        return make_real_chat_model(
            timeout=settings.llm_timeout_s,  # never wait forever on a provider
            max_retries=settings.llm_max_retries,
        )
    return OfflineChatModel()


def get_embeddings():
    """Returns (embeddings, fingerprint). The fingerprint is stored with the index: if you change
    embedding models, vectors from the old model are meaningless and the index MUST be rebuilt."""
    if bool(os.getenv("OPENAI_API_KEY")):
        try:
            from langchain_openai import OpenAIEmbeddings

            name = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
            return OpenAIEmbeddings(model=name), f"openai:{name}"
        except ImportError:
            pass
    return HashingEmbeddings(), "offline-hashing-1024"
