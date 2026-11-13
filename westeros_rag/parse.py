"""STEP 1 - STRUCTURE EXTRACTION: markdown -> document tree + link graph.

Why parse structure instead of splitting text blindly?
  * Headings are the author's own semantic boundaries. A chunk that respects them is a coherent unit
    ("Capital Cases" under "When It May Be Demanded"), and its heading PATH is free context for retrieval.
  * Links are the author's own relevance edges. "See Attainder" is a retrieval hint no embedding can infer.
  * Section-level metadata (jurisdiction, access) can differ inside one document.

Output model
  ParsedDoc  -> Sections (H1..H6) arranged as a tree (parent/children), each with its OWN text and its outgoing Links.
  Corpus     -> all docs + helpers: resolve links, find the "unit" (H2 subtree) a section belongs to, backlinks.

A UNIT is the context granularity we hand to the LLM: an H2 section INCLUDING its H3+ subsections (the H1 intro
is a unit of its own). Chunks (retrieval granularity) are smaller; units (context granularity) are bigger.
That asymmetry is parent-child chunking.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rag_prod.ingest import parse_front_matter  # same tiny front-matter parser as lesson 14
from westeros_rag.config import KNOWN_ACCESS, REGIONS

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
META_COMMENT = re.compile(r"^<!--\s*([A-Za-z_]+)\s*:\s*(.+?)\s*-->$")


def slugify(text: str) -> str:
    """GitHub-style anchor: lowercase, drop punctuation, spaces -> hyphens. (So links work in any markdown viewer.)"""
    s = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"\s+", "-", s.strip())


def strip_links(text: str) -> str:
    """`[trial by combat](trial_by_combat.md#x)` -> `trial by combat`.

    Raw link syntax is poison for retrieval: URL fragments become tokens in embeddings and BM25 (`crimes_and_punishments`,
    `md`, `murder`...) and make unrelated sections look similar. Index and show the TEXT; keep the links as graph metadata
    (Section.links) - they are used for link expansion, not for matching."""
    return LINK.sub(lambda m: m.group(1), text)


@dataclass
class Link:
    text: str
    target_doc: str | None  # None = same document
    target_anchor: str | None  # None = whole document


@dataclass
class Section:
    section_id: str  # "<doc_id>#<slug>"
    doc_id: str
    slug: str
    title: str
    level: int
    heading_path: list[str]
    parent_id: str | None
    children: list[str] = field(default_factory=list)
    text: str = ""  # this section's OWN text (excludes subsections)
    links: list[Link] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)  # from <!-- key: value --> right after the heading


@dataclass
class ParsedDoc:
    doc_id: str
    path: Path
    meta: dict[str, str]
    content_hash: str
    sections: list[Section]  # document order

    @property
    def title(self) -> str:
        return self.meta.get("title") or self.sections[0].title


def parse_markdown(raw: str, path: Path) -> ParsedDoc:
    meta, body = parse_front_matter(raw)
    doc_id = meta.get("doc_id", path.stem)
    sections: list[Section] = []
    stack: list[Section] = []
    slugs: Counter = Counter()
    in_fence = False
    pending_meta_for: Section | None = None
    buf: list[str] = []

    def flush() -> None:
        if sections:
            sections[-1].text = "\n".join(buf).strip()
            for m in LINK.finditer(sections[-1].text):
                target = m.group(2)
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                doc_part, _, anchor = target.partition("#")
                sections[-1].links.append(
                    Link(m.group(1), doc_part[:-3] if doc_part.endswith(".md") else (doc_part or None), anchor or None)
                )
        buf.clear()

    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            slug = slugify(title)
            slugs[slug] += 1
            if slugs[slug] > 1:  # duplicate headings get -1, -2 like GitHub does
                slug = f"{slug}-{slugs[slug] - 1}"
            while stack and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1] if stack else None
            sec = Section(
                section_id=f"{doc_id}#{slug}", doc_id=doc_id, slug=slug, title=title, level=level,
                heading_path=[*(parent.heading_path if parent else []), title],
                parent_id=parent.section_id if parent else None,
            )
            if parent:
                parent.children.append(sec.section_id)
            sections.append(sec)
            stack.append(sec)
            pending_meta_for = sec
            continue
        if pending_meta_for is not None and line.strip():
            cm = META_COMMENT.match(line.strip())
            pending_meta_for_ref, pending_meta_for = pending_meta_for, None
            if cm:  # section-level metadata, e.g. <!-- jurisdiction: north -->
                pending_meta_for_ref.meta[cm.group(1)] = cm.group(2)
                continue
        buf.append(line)
    flush()
    return ParsedDoc(doc_id, path, meta, hashlib.sha256(raw.encode()).hexdigest()[:16], sections)


class Corpus:
    def __init__(self, docs: list[ParsedDoc]) -> None:
        self.docs = {d.doc_id: d for d in docs}
        self.sections = {s.section_id: s for d in docs for s in d.sections}
        self.duplicate_doc_ids = [i for i, c in Counter(d.doc_id for d in docs).items() if c > 1]

    @classmethod
    def load(cls, directory: Path) -> "Corpus":
        return cls([parse_markdown(p.read_text(encoding="utf-8"), p) for p in sorted(directory.glob("*.md"))])

    # ---- attributes with inheritance: section meta overrides doc front matter
    def access(self, section_id: str) -> str:
        return self.docs[self.sections[section_id].doc_id].meta.get("access", "internal")  # secure default

    def jurisdiction(self, section_id: str) -> str:
        s = self.sections[section_id]
        return s.meta.get("jurisdiction") or self.docs[s.doc_id].meta.get("jurisdiction", "all")

    # ---- units (context granularity)
    def unit_id(self, section_id: str) -> str:
        s = self.sections[section_id]
        while s.level > 2 and s.parent_id:
            s = self.sections[s.parent_id]
        return s.section_id

    def units(self, doc_id: str | None = None) -> list[Section]:
        docs = [self.docs[doc_id]] if doc_id else list(self.docs.values())
        return [s for d in docs for s in d.sections if s.level <= 2]

    def subtree(self, section_id: str) -> list[Section]:
        out = [self.sections[section_id]]
        for child in self.sections[section_id].children:
            out.extend(self.subtree(child))
        return out

    def unit_members(self, unit_id: str) -> list[Section]:
        """Sections that belong to a unit. An H1 unit is ONLY the intro (its H2s are units of their own);
        an H2 unit is the section plus all of its H3+ subsections."""
        root = self.sections[unit_id]
        return [root] if root.level == 1 else self.subtree(unit_id)

    def unit_text(self, unit_id: str) -> str:
        """The unit's own text plus every subsection, each introduced by its heading (keeps structure for the LLM)."""
        root = self.sections[unit_id]
        parts = [strip_links(root.text)] if root.text else []
        for s in self.unit_members(unit_id)[1:]:
            parts.append(f"{'#' * s.level} {s.title}\n{strip_links(s.text)}".strip())
        return "\n\n".join(parts).strip()

    # ---- link graph
    def resolve(self, link: Link, from_doc: str) -> str | None:
        doc_id = link.target_doc or from_doc
        doc = self.docs.get(doc_id)
        if not doc:
            return None
        if not link.target_anchor:
            return doc.sections[0].section_id
        sid = f"{doc_id}#{link.target_anchor}"
        return sid if sid in self.sections else None

    def unit_links(self, unit_id: str) -> list[str]:
        """Outgoing links from anywhere in the unit's subtree, resolved to OTHER units. Order preserved, deduped."""
        seen: dict[str, None] = {}
        for s in self.unit_members(unit_id):
            for link in s.links:
                target = self.resolve(link, s.doc_id)
                if target and self.unit_id(target) != unit_id:
                    seen.setdefault(self.unit_id(target), None)
        return list(seen)

    def backlinks(self) -> dict[str, set[str]]:
        inbound: dict[str, set[str]] = {sid: set() for sid in self.sections}
        for s in self.sections.values():
            for link in s.links:
                target = self.resolve(link, s.doc_id)
                if target:
                    inbound[target].add(s.section_id)
        return inbound


# ------------------------------------------------------------------ corpus linter (a QUALITY GATE for ingestion)
@dataclass
class Issue:
    kind: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.where}: {self.detail}"


def lint(corpus: Corpus) -> list[Issue]:
    """Bad inputs make bad RAG. Fail the ingest on structural problems instead of discovering them in production."""
    issues: list[Issue] = []
    for dup in corpus.duplicate_doc_ids:
        issues.append(Issue("duplicate_doc_id", dup, "two files share a doc_id"))
    inbound = corpus.backlinks()
    for doc in corpus.docs.values():
        where = doc.path.name
        for key in ("doc_id", "title", "access", "jurisdiction"):
            if key not in doc.meta:
                issues.append(Issue("missing_front_matter", where, f"missing '{key}'"))
        if doc.meta.get("doc_id", doc.path.stem) != doc.path.stem:
            issues.append(Issue("doc_id_mismatch", where, "doc_id must equal the file name"))
        if doc.meta.get("access") and doc.meta["access"] not in KNOWN_ACCESS:
            issues.append(Issue("unknown_access", where, doc.meta["access"]))
        j = doc.meta.get("jurisdiction")
        if j and j != "all" and j not in REGIONS:
            issues.append(Issue("unknown_jurisdiction", where, j))
        h1 = [s for s in doc.sections if s.level == 1]
        if len(h1) != 1:
            issues.append(Issue("h1_count", where, f"expected exactly one H1, found {len(h1)}"))
        prev = 0
        for s in doc.sections:
            if s.level > prev + 1 and prev:
                issues.append(Issue("heading_skip", s.section_id, f"H{prev} -> H{s.level}"))
            prev = s.level
            if not s.text and not s.children:
                issues.append(Issue("empty_section", s.section_id, "heading with no text"))
            if s.meta.get("jurisdiction") and s.meta["jurisdiction"] not in REGIONS:
                issues.append(Issue("unknown_jurisdiction", s.section_id, s.meta["jurisdiction"]))
            for link in s.links:
                if link.target_doc and link.target_doc not in corpus.docs:
                    issues.append(Issue("broken_link", s.section_id, f"unknown document '{link.target_doc}'"))
                elif corpus.resolve(link, s.doc_id) is None:
                    issues.append(Issue("broken_anchor", s.section_id, f"'{link.target_doc or ''}#{link.target_anchor}' not found"))
        doc_inbound = sum(len(inbound[s.section_id]) for s in doc.sections)
        # Public docs must be reachable. Restricted docs are EXPECTED to be unlinked from public ones:
        # a public page linking to a restricted page leaks its existence.
        if doc_inbound == 0 and doc.doc_id != "index" and doc.meta.get("access") == "public":
            issues.append(Issue("orphan_doc", where, "no other section links here"))
    return issues
