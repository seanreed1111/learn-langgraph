"""STEP 3 - INDEXING: two vector indexes in one store, both carrying rich metadata.

  kind="summary" : one entry per UNIT (H2 subtree or doc overview): "<heading path>\\n<summary>\\nThemes: ..."
                   -> answers "WHICH section is this question about?"            (stage A, coarse)
  kind="chunk"   : leaf-level text of every section, split to ~450 chars, each prefixed with its HEADING PATH
                   -> answers "WHICH passage inside it says so?"                  (stage B, fine)

Every entry carries metadata: doc_id, unit_id, section_id, heading_path, access, jurisdiction, themes.
Metadata powers (1) access control, (2) jurisdiction filtering, (3) theme boosting, (4) citations with deep links.

Same production properties as lesson 14: incremental per document, idempotent, persistent, embedding-fingerprinted.
Swap-out: replace InMemoryVectorStore with pgvector/Qdrant; keep the metadata schema and filter semantics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, override

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_prod.observability import logger
from westeros_rag.config import WSettings
from westeros_rag.parse import Corpus, strip_links


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass
class SyncStats:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    entries: int = 0

    @override
    def __str__(self) -> str:
        return f"added={len(self.added)} updated={len(self.updated)} removed={len(self.removed)} unchanged={len(self.unchanged)} entries={self.entries}"


def build_entries(doc_id: str, corpus: Corpus, enrich: dict[str, dict[str, Any]], settings: WSettings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    entries: list[Document] = []
    for unit in corpus.units(doc_id):
        meta = enrich[unit.section_id]
        base = {
            "doc_id": doc_id, "unit_id": unit.section_id, "access": corpus.access(unit.section_id),
            "jurisdiction": corpus.jurisdiction(unit.section_id), "themes": meta["themes"],
        }
        path = " > ".join(unit.heading_path)
        entries.append(Document(
            id=_sha(f"sum:{unit.section_id}:{meta['summary']}"),
            page_content=f"{path}\n{meta['summary']}\nThemes: {', '.join(meta['themes'])}",
            metadata={**base, "kind": "summary", "section_id": unit.section_id, "heading_path": path},
        ))
        for sec in corpus.unit_members(unit.section_id):
            if not sec.text:
                continue
            spath = " > ".join(sec.heading_path)
            for i, piece in enumerate(splitter.split_text(strip_links(sec.text))):
                content = f"{spath}\n{piece}"  # heading path prefix: coherent, self-describing chunks
                entries.append(Document(
                    id=_sha(f"chunk:{sec.section_id}:{i}:{content}"), page_content=content,
                    metadata={**base, "kind": "chunk", "section_id": sec.section_id, "heading_path": spath, "chunk_index": i},
                ))
    return entries


class VectorIndex:
    def __init__(self, settings: WSettings, embeddings, fingerprint: str) -> None:
        self.settings, self.embeddings, self.fingerprint = settings, embeddings, fingerprint
        self.manifest: dict[str, Any] = {"embedding": fingerprint, "docs": {}}
        self.store = InMemoryVectorStore(embeddings)
        self._load()

    @property
    def _manifest_path(self):
        return self.settings.index_path.with_suffix(".manifest.json")

    def _load(self) -> None:
        p = self.settings.index_path
        if not (p.exists() and self._manifest_path.exists()):
            return
        manifest = json.loads(self._manifest_path.read_text())
        if manifest.get("embedding") != self.fingerprint:
            logger.warning("embedding model changed: full rebuild")
            return
        self.store = InMemoryVectorStore.load(str(p), self.embeddings)
        self.manifest = manifest

    def _save(self) -> None:
        self.settings.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.dump(str(self.settings.index_path))
        self._manifest_path.write_text(json.dumps(self.manifest, indent=1))

    def sync(self, corpus: Corpus, enrich: dict[str, dict[str, Any]]) -> SyncStats:
        stats, known = SyncStats(), self.manifest["docs"]
        for doc_id in [d for d in known if d not in corpus.docs]:
            self.store.delete(ids=known.pop(doc_id)["ids"])
            stats.removed.append(doc_id)
        for doc_id, doc in corpus.docs.items():
            # signature covers the file AND its enrichment: re-summarizing a section must refresh its summary vector
            sig = _sha(doc.content_hash + json.dumps([enrich[u.section_id] for u in corpus.units(doc_id)], sort_keys=True))
            if known.get(doc_id, {}).get("sig") == sig:
                stats.unchanged.append(doc_id)
                continue
            if doc_id in known:
                self.store.delete(ids=known[doc_id]["ids"])  # drop stale entries BEFORE adding new ones
                stats.updated.append(doc_id)
            else:
                stats.added.append(doc_id)
            entries = build_entries(doc_id, corpus, enrich, self.settings)
            self.store.add_documents(entries, ids=[str(e.id) for e in entries])
            known[doc_id] = {"sig": sig, "ids": [e.id for e in entries]}
        stats.entries = sum(len(v["ids"]) for v in known.values())
        if stats.added or stats.updated or stats.removed or not self.settings.index_path.exists():
            self._save()
        return stats

    def entries(self, kind: str) -> list[Document]:
        return [Document(id=v["id"], page_content=v["text"], metadata=v["metadata"])
                for v in self.store.store.values() if v["metadata"].get("kind") == kind]
