"""Model factory. One place decides which chat model / embeddings the service uses.

Real mode: any provider key in the environment -> init_chat_model with a timeout and retries.
Offline mode: deterministic test doubles so CI and evals need no network or money.
"""

from __future__ import annotations

import os
import sys

from rag_prod.config import ROOT, Settings

sys.path.insert(0, str(ROOT / "lessons"))
from common import DEFAULT_MODEL, have_real_llm  # noqa: E402
from offline_llm import HashingEmbeddings, OfflineChatModel  # noqa: E402


def get_chat_model(settings: Settings):
    if have_real_llm():
        from langchain.chat_models import init_chat_model

        return init_chat_model(
            os.getenv("LG_MODEL", DEFAULT_MODEL),
            temperature=0,
            timeout=settings.llm_timeout_s,  # never wait forever on a provider
            max_retries=settings.llm_max_retries,
        )
    return OfflineChatModel()


def get_embeddings():
    """Returns (embeddings, fingerprint). The fingerprint is stored with the index: if you change
    embedding models, vectors from the old model are meaningless and the index MUST be rebuilt."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            name = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
            return OpenAIEmbeddings(model=name), f"openai:{name}"
        except ImportError:
            pass
    return HashingEmbeddings(), "offline-hashing-1024"
