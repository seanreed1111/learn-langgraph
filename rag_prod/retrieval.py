"""Hybrid retrieval with ACCESS CONTROL enforced at query time.

Why hybrid?  Vector search is great at meaning ("get money back" ~ "reimbursement") but weak on exact
tokens (error codes, SKUs, names, "Sev1"). Keyword search (BM25) is the reverse. Fuse both with
Reciprocal Rank Fusion (RRF): robust, needs no score calibration, and is a standard production baseline.

Why filter at retrieval?  ACLs MUST be applied BEFORE text can reach the LLM. Never retrieve
everything and "ask the model not to mention" restricted content - that is a data leak waiting to happen.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from langchain_core.documents import Document

from rag_prod.config import Settings
from rag_prod.ingest import Index

_STOP = set("the a an and or of to in on for with is are was were be it this that what how do does i my me you we".split())


def tokenize(text: str) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks]  # crude plural stemming


class BM25:
    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(tokenize(d.page_content)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        q = tokenize(query)
        scored = []
        for doc, tf, ln in zip(self.docs, self.tf, self.len):
            s = 0.0
            for t in q:
                if t in tf:
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            if s > 0:
                scored.append((doc, s))
        return sorted(scored, key=lambda x: -x[1])[:k]


def rrf(rankings: list[list[Document]], k: int = 60) -> list[tuple[Document, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank)
            by_id[doc.id] = doc
    return sorted(((by_id[i], s) for i, s in scores.items()), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, index: Index, settings: Settings) -> None:
        self.index, self.settings = index, settings

    def search(self, query: str, allowed_access: frozenset[str] | set[str]) -> list[Document]:
        allowed = set(allowed_access)
        if not allowed:
            return []  # fail closed

        def permitted(doc: Document) -> bool:
            return doc.metadata.get("access") in allowed

        vec = [d for d, _ in self.index.store.similarity_search_with_score(query, k=self.settings.top_k, filter=permitted)]
        pool = [d for d in self.index.all_documents() if permitted(d)]
        kw = [d for d, _ in BM25(pool).search(query, self.settings.top_k)]

        fused = rrf([vec, kw])[: self.settings.final_k]
        return [Document(id=d.id, page_content=d.page_content, metadata={**d.metadata, "rrf_score": round(s, 5)})
                for d, s in fused]
