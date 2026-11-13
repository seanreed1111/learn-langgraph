"""STEP 4 - RETRIEVAL: hierarchical, theme-aware, link-following, access-controlled.

    query ──► STAGE B  search CHUNK entries    (hybrid vector+BM25) ─► candidate PASSAGES ("which text?")      [always]
          └─► STAGE A  search SUMMARY entries  (hybrid vector+BM25) ─► candidate SECTIONS ("which section?")  [opt-in]
                        unit ranking = RRF(rank by best chunk, weight * rank by summary) + small theme bonus
          ─► PARENT EXPANSION  hand the LLM the whole unit (H2 subtree), not the 450-char chunk that matched
          ─► LINK EXPANSION    follow the unit's cross-references (relevance-ranked, access-checked, cycle-safe, budgeted)
          ─► CONTEXT BUDGET    trim to a character budget; primary evidence first

Filters applied BEFORE any text can reach the model:  access level (always) and jurisdiction (when the user names a region).
Themes are only a SOFT boost: an LLM-chosen tag may be wrong, and a wrong HARD filter silently hides the answer.

Every stage is switchable through WSettings so evals can ablate it (see evals/ablate.py).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.documents import Document

from rag_prod.retrieval import BM25, rrf, tokenize
from westeros_rag.config import WSettings
from westeros_rag.index import VectorIndex
from westeros_rag.parse import Corpus


@dataclass
class Passage:
    unit_id: str
    doc_id: str
    heading_path: str
    text: str
    summary: str
    kind: str  # "primary" (matched the query) | "linked" (reached through a cross-reference)
    score: float
    via: str | None = None  # for linked passages: the unit whose link led here
    themes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["themes"] = list(self.themes)
        return d


class HierarchicalRetriever:
    def __init__(self, corpus: Corpus, index: VectorIndex, enrich: dict[str, dict[str, Any]], settings: WSettings) -> None:
        self.corpus, self.index, self.enrich, self.s = corpus, index, enrich, settings

    # ------------------------------------------------------------------ helpers
    def _permitted(self, meta: dict[str, Any], allowed: set[str], jurisdiction: str | None) -> bool:
        return meta["access"] in allowed and (jurisdiction is None or meta["jurisdiction"] in ("all", jurisdiction))

    def _ranked(self, query: str, kind: str, k: int, allowed: set[str], jurisdiction: str | None):
        ok = lambda d: d.metadata["kind"] == kind and self._permitted(d.metadata, allowed, jurisdiction)  # noqa: E731
        vec = [d for d, _ in self.index.store.similarity_search_with_score(query, k=k, filter=ok)]
        pool = [d for d in self.index.entries(kind) if self._permitted(d.metadata, allowed, jurisdiction)]
        kw = [d for d, _ in BM25(pool).search(query, k)]
        return rrf([vec, kw])

    # ------------------------------------------------------------------ main entry
    def search(self, query: str, allowed_access, jurisdiction: str | None = None, themes=()) -> list[Passage]:
        s, allowed = self.s, set(allowed_access)
        if not allowed:
            return []  # fail closed
        themes = set(themes)

        # Stage B: which UNITS do the best-matching CHUNKS belong to?  (ordered, deduped)
        unit_chunks: dict[str, list[Document]] = defaultdict(list)
        by_chunks: list[str] = []
        for doc, _score in self._ranked(query, "chunk", s.top_chunks, allowed, jurisdiction):
            u = doc.metadata["unit_id"]
            unit_chunks[u].append(doc)
            if u not in by_chunks:
                by_chunks.append(u)
        rankings = [by_chunks]

        # Stage A: which UNITS do the best-matching SUMMARIES describe?  Fused with stage B by RANK (RRF) - not by
        # adding raw scores, which live on different scales and double-count (my first version did exactly that,
        # and the ablation in evals/ablate.py showed it HURT recall).
        if s.strategy == "hier":
            ranked_summaries = self._ranked(query, "summary", s.top_sections, allowed, jurisdiction)[: s.top_sections]
            rankings.append([doc.metadata["unit_id"] for doc, _ in ranked_summaries])

        unit_score: dict[str, float] = defaultdict(float)
        for weight, ranking in zip([1.0, s.summary_weight], rankings, strict=False):  # summaries are a COARSER signal: weight < 1
            for rank, u in enumerate(ranking, start=1):
                unit_score[u] += weight / (60 + rank)
        if s.use_themes and themes:  # SOFT boost only: a wrong LLM-chosen tag must never hide the right answer
            for u in unit_score:
                unit_score[u] += s.theme_bonus * len(themes & set(self.enrich[u]["themes"]))

        ranked = sorted(unit_score.items(), key=lambda x: -x[1])[: s.final_units]
        primaries = [self._passage(u, sc, "primary", unit_chunks.get(u, [])) for u, sc in ranked]
        linked = self._follow_links(primaries, query, allowed, jurisdiction) if s.use_links else []
        return self._within_budget(primaries + linked)

    # ------------------------------------------------------------------ building passages
    def _passage(self, unit_id: str, score: float, kind: str, chunks: list[Document], via: str | None = None) -> Passage:
        meta = self.enrich[unit_id]
        sec = self.corpus.sections[unit_id]
        if self.s.use_parent or not chunks:
            text = self.corpus.unit_text(unit_id) or meta["summary"]  # PARENT: the whole coherent section
        else:
            text = "\n\n".join(c.page_content for c in chunks[:2])  # CHILD only: just what matched
        return Passage(unit_id, sec.doc_id, " > ".join(sec.heading_path), text, meta["summary"], kind, round(score, 5), via, tuple(meta["themes"]))

    def _follow_links(self, primaries: list[Passage], query: str, allowed: set[str], jurisdiction: str | None) -> list[Passage]:
        """Cross-references are the author's own relevance edges. Follow them, but keep it SAFE:
        visited-set (cycles), access + jurisdiction checks (no leaking through links), relevance-ranked, capped."""
        q = set(tokenize(query))
        visited = {p.unit_id for p in primaries}
        frontier, out = list(primaries), list[Passage]()
        for _ in range(self.s.link_depth):
            cands = []
            for src in frontier:
                for target in self.corpus.unit_links(src.unit_id):
                    if target in visited:
                        continue
                    meta = {"access": self.corpus.access(target), "jurisdiction": self.corpus.jurisdiction(target)}
                    if not self._permitted(meta, allowed, jurisdiction):
                        continue
                    info = self.enrich[target]
                    text = " ".join([" ".join(self.corpus.sections[target].heading_path), info["summary"], *info["themes"]])
                    overlap = len(q & set(tokenize(text)))
                    if overlap > 0:
                        cands.append((overlap, src.score, target, src))
            cands.sort(key=lambda c: (-c[0], -c[1]))
            frontier = []
            for _overlap, _score, target, src in cands:
                if target in visited or len(out) >= self.s.max_linked:
                    continue
                visited.add(target)
                p = self._passage(target, src.score * 0.5, "linked", [], via=src.unit_id)
                out.append(p)
                frontier.append(p)
        return out

    def _within_budget(self, passages: list[Passage]) -> list[Passage]:
        budget, kept = self.s.context_budget_chars, list[Passage]()
        for p in passages:
            if budget <= 0:
                break
            if len(p.text) > budget:
                if budget < 300:
                    break
                p.text = p.text[:budget].rsplit(" ", 1)[0] + " …"
            budget -= len(p.text)
            kept.append(p)
        return kept
