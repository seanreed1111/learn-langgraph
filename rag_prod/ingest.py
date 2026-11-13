"""Ingestion: documents -> chunks -> vector index. The part tutorials skip and production lives or dies on.

Properties this module guarantees:
  * IDEMPOTENT      : running sync() twice changes nothing the second time.
  * INCREMENTAL     : only changed/new files are re-embedded (embedding costs money); deleted files are removed.
  * STABLE IDS      : chunk id = hash(source, position, content hash) -> safe upserts, easy debugging.
  * METADATA        : every chunk carries source/title/access/updated -> citations + access control.
  * MODEL-AWARE     : the embedding model fingerprint is stored; a change forces a full rebuild.
  * PERSISTENT      : the index survives restarts (here a JSON file; in prod: pgvector/Qdrant/etc.).

SWAP-OUT NOTE: InMemoryVectorStore keeps this lesson dependency-free. In production use a real
vector DB (Postgres+pgvector, Qdrant, Pinecone, OpenSearch...). Only `Index` changes - the graph
only needs "search with a metadata filter".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from rag_prod.config import Settings
from rag_prod.observability import logger


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Tiny `---\\nkey: value\\n---` parser (use python-frontmatter/yaml in real life)."""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            meta = {}
            for line in text[4:end].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            return meta, text[end + 4 :].lstrip("\n")
    return {}, text


def load_documents(kb_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        docs.append(
            Document(
                page_content=body,
                metadata={
                    "source": path.name,
                    "title": meta.get("title", path.stem),
                    # SECURE DEFAULT: a doc with no access label is NOT public.
                    "access": meta.get("access", "internal"),
                    "updated": meta.get("updated", ""),
                    "doc_hash": _sha(raw),
                },
            )
        )
    return docs


def chunk_document(doc: Document, settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = []
    for i, text in enumerate(splitter.split_text(doc.page_content)):
        meta = {**doc.metadata, "chunk_index": i}
        # Contextual header: a chunk like "Up to 5 days carry over" is meaningless alone.
        # Prefixing the title makes both embeddings and keyword search far better.
        content = f"[{doc.metadata['title']}]\n{text}"
        chunks.append(Document(id=_sha(f"{doc.metadata['source']}:{i}:{_sha(content)}"), page_content=content, metadata=meta))
    return chunks


@dataclass
class SyncStats:
    added_docs: list[str] = field(default_factory=list)
    updated_docs: list[str] = field(default_factory=list)
    removed_docs: list[str] = field(default_factory=list)
    unchanged_docs: list[str] = field(default_factory=list)
    chunks: int = 0

    def __str__(self) -> str:
        return (f"added={len(self.added_docs)} updated={len(self.updated_docs)} "
                f"removed={len(self.removed_docs)} unchanged={len(self.unchanged_docs)} chunks={self.chunks}")


class Index:
    def __init__(self, settings: Settings, embeddings, embedding_fingerprint: str) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.fingerprint = embedding_fingerprint
        self.manifest: dict = {"embedding": embedding_fingerprint, "docs": {}}
        self.store = InMemoryVectorStore(embeddings)
        self._load()

    # ---- persistence
    @property
    def _manifest_path(self) -> Path:
        return self.settings.index_path.with_suffix(".manifest.json")

    def _load(self) -> None:
        if not (self.settings.index_path.exists() and self._manifest_path.exists()):
            return
        manifest = json.loads(self._manifest_path.read_text())
        if manifest.get("embedding") != self.fingerprint:
            logger.warning("embedding model changed (%s -> %s): full rebuild",
                           manifest.get("embedding"), self.fingerprint)
            return  # start empty; sync() will re-embed everything
        self.store = InMemoryVectorStore.load(str(self.settings.index_path), self.embeddings)
        self.manifest = manifest

    def _save(self) -> None:
        self.settings.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.dump(str(self.settings.index_path))
        self._manifest_path.write_text(json.dumps(self.manifest, indent=2))

    # ---- sync
    def sync(self, kb_dir: Path | None = None) -> SyncStats:
        stats = SyncStats()
        docs = {d.metadata["source"]: d for d in load_documents(kb_dir or self.settings.kb_dir)}
        known = self.manifest["docs"]

        gone = [s for s in known if s not in docs]
        for source in gone:
            self.store.delete(ids=known.pop(source)["ids"])
            stats.removed_docs.append(source)

        for source, doc in docs.items():
            entry = known.get(source)
            if entry and entry["hash"] == doc.metadata["doc_hash"]:
                stats.unchanged_docs.append(source)
                continue
            if entry:  # changed -> drop old chunks first so stale text can never be retrieved
                self.store.delete(ids=entry["ids"])
                stats.updated_docs.append(source)
            else:
                stats.added_docs.append(source)
            chunks = chunk_document(doc, self.settings)
            self.store.add_documents(chunks, ids=[c.id for c in chunks])
            known[source] = {"hash": doc.metadata["doc_hash"], "ids": [c.id for c in chunks]}

        stats.chunks = sum(len(e["ids"]) for e in known.values())
        if stats.added_docs or stats.updated_docs or stats.removed_docs or not self.settings.index_path.exists():
            self._save()
        logger.info("index_sync", extra={"fields": {"stats": str(stats)}})
        return stats

    def all_documents(self) -> list[Document]:
        """Used by the keyword (BM25) retriever. With a real DB, use its native full-text search instead."""
        return [
            Document(id=v["id"], page_content=v["text"], metadata=v["metadata"])
            for v in self.store.store.values()
        ]
