"""Offline TEST DOUBLES for RAG lessons (lessons 13 and 14).

These let the full RAG graphs run with no API key and no network:

  HashingEmbeddings  - bag-of-words vectors (lexical, NOT semantic). Good enough to make
                       retrieval deterministic in tests; swap for real embeddings in prod.
  OfflineChatModel   - a rule-based stand-in that looks at the PROMPT and answers the
                       way a well-behaved LLM would (route, grade, rewrite, answer, verify).

They implement the same duck-typed interface the graphs use (`invoke`, `bind_tools`,
`with_structured_output`, `batch`), so real models drop in with zero graph changes.
Never ship these - they exist so you can learn and TEST graph logic deterministically.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Any, override

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage

STOP = set(
    ["the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "what", "who", "whom", "how", "why", "when", "where", "which", "does", "do", "did", "can", "could", "would", "should", "about", "from", "into", "at", "by", "as", "not", "no", "any", "all", "some", "your", "our", "their", "his", "her", "you", "we", "they", "i", "me", "my", "say", "says", "said", "tell", "please", "there", "than", "then", "so", "if", "but", "also", "have", "has", "had", "more", "most", "very"]
)

# A tiny thesaurus the offline "rewriter" uses to expand vague questions into corpus vocabulary.
THESAURUS = {
    "cheat": "reward hacking specification gaming",
    "cheating": "reward hacking specification gaming",
    "game": "reward hacking specification gaming",
    "machine": "AI agent reward function",
    "machines": "AI agent reward function",
    "objective": "reward function misspecification",
    "objectives": "reward function misspecification",
    "making": "hallucination fabricated",
    "lie": "hallucination fabricated facts",
    "lies": "hallucination fabricated facts",
    "money": "refund reimbursement expense",
    "trip": "business travel expense",
    "trips": "business travel expense",
    "vacation": "paid time off PTO leave",
    "holiday": "paid time off PTO leave",
    "holidays": "paid time off PTO leave",
    "laptop": "equipment hardware stipend",
    "wfh": "remote work policy",
    "password": "security credentials authentication",
}


def stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def words(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2]


class HashingEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embeddings (L2-normalised)."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for w in words(text):
            vec[zlib.crc32(w.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# --------------------------------------------------------------------------- helpers
def _text_of(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("content", ""))
    if isinstance(msg, BaseMessage):
        return msg.text if hasattr(msg, "text") else str(msg.content)
    return str(msg)


def _prompt(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(_text_of(m) for m in messages)


def _between(text: str, start: str, end: str) -> str:
    m = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, re.S)
    return str(m.group(1)).strip() if m else ""


def _question_from(prompt: str) -> str:
    q = _between(prompt, "<question>", "</question>")
    if q:
        return q
    q = _between(prompt, "user question:", "\n")
    if q:
        return q
    q = _between(prompt, "Question:", "\n")
    return q


def _context_from(prompt: str) -> str:
    return _between(prompt, "<context>", "</context>")


def _strip_md(text: str) -> str:
    """Drop markdown headings and our '[Title]' chunk headers so fake answers read like sentences."""
    keep = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(("#", "---", "title:")) or (s.startswith("[") and s.endswith("]")):
            continue
        keep.append(s)
    return " ".join(keep)


def overlap_ratio(question: str, context: str) -> float:
    qw = set(words(question))
    if not qw:
        return 0.0
    return len(qw & set(words(context))) / len(qw)


def expand(question: str) -> str:
    """Add synonyms (a real LLM 'understands' that 'game' ~ 'gaming'); used only when ANSWERING."""
    extra = " ".join(THESAURUS[w] for w in re.findall(r"[a-z]+", question.lower()) if w in THESAURUS)
    return f"{question} {extra}".strip()


GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon))\b", re.I)


class _Structured:
    def __init__(self, schema: type) -> None:
        self.schema = schema

    def invoke(self, messages: Any, *a: Any, **k: Any) -> Any:
        p = _prompt(messages)
        name = self.schema.__name__
        q, ctx = _question_from(p), _context_from(p)
        if name in ("GradeDocuments", "DocGrade"):
            # Lesson 13 grades the raw question (its vague-question demo relies on that);
            # lesson 14's DocGrade gets the thesaurus, like a real LLM that knows synonyms.
            qx = expand(q) if name == "DocGrade" else q
            ok = overlap_ratio(qx, ctx) >= 0.5 or len(set(words(qx)) & set(words(ctx))) >= 3
            return self.schema(binary_score="yes" if ok else "no")
        if name == "RouteDecision":
            return self.schema(route="chitchat" if GREETING.match(q or p.splitlines()[-1]) else "retrieve")
        if name == "CitedAnswer":
            return self._cited_answer(p, q)
        if name == "Groundedness":
            ans = _between(p, "<answer>", "</answer>")
            ok = overlap_ratio(ans, ctx) >= 0.5
            return self.schema(grounded=ok, reason="all claims supported" if ok else "unsupported claims")
        raise NotImplementedError(f"OfflineChatModel has no handler for schema {name}")

    def batch(self, inputs: list[Any], *a: Any, **k: Any) -> list[Any]:
        return [self.invoke(i) for i in inputs]

    def _cited_answer(self, p: str, q: str) -> Any:
        # Context passages are numbered like "[1] (source) text". Pick best-matching sentences.
        ctx = _context_from(p)
        q = expand(q)
        passages = re.findall(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\Z)", ctx, re.S)
        scored = []
        for pid, text in passages:
            for sent in re.split(r"(?<=[.!?])\s+", _strip_md(text)):
                scored.append((overlap_ratio(q, sent), int(pid), sent.strip()))
        scored.sort(key=lambda t: -t[0])
        best = [s for s in scored[:2] if s[0] > 0]
        if not best:
            return self.schema(answer="I don't know based on the provided context.", citations=[])
        return self.schema(
            answer=" ".join(s[2] for s in best), citations=sorted({s[1] for s in best})
        )


class OfflineChatModel:
    """Rule-based fake chat model. See module docstring."""

    def __init__(self, tools: list[Any] | None = None) -> None:
        self.tools = tools or []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> OfflineChatModel:
        return OfflineChatModel(tools)

    def with_structured_output(self, schema: type, **kwargs: Any) -> _Structured:
        return _Structured(schema)

    def batch(self, inputs: list[Any], *a: Any, **k: Any) -> list[AIMessage]:
        return [self.invoke(i) for i in inputs]

    def invoke(self, messages: Any, *a: Any, **k: Any) -> AIMessage:
        p = _prompt(messages)
        if "Formulate an improved question" in p or "Rewrite the search query" in p:
            q = _between(p, "-------", "-------") or _between(p, "<question>", "</question>")
            extra = " ".join(THESAURUS[w] for w in re.findall(r"[a-z]+", q.lower()) if w in THESAURUS)
            return AIMessage(content=f"{q} {extra}".strip())
        if "standalone question" in p:
            return AIMessage(content=_between(p, "<question>", "</question>"))
        if "question-answering tasks" in p:
            q, ctx = expand(_question_from(p)), _context_from(p)
            sents = [s for s in re.split(r"(?<=[.!?])\s+", _strip_md(ctx)) if s.strip()]
            sents.sort(key=lambda s: -overlap_ratio(q, s))
            if not sents or overlap_ratio(q, sents[0]) == 0:
                return AIMessage(content="I do not know.")
            return AIMessage(content=" ".join(sents[:2]))
        if self.tools:  # generate_query_or_respond: decide to call the retriever tool or chat
            human = [m for m in (messages if isinstance(messages, list) else []) if getattr(m, "type", None) == "human"
                     or (isinstance(m, dict) and m.get("role") == "user")]
            last_q = _text_of(human[-1]) if human else p
            if GREETING.match(last_q):
                return AIMessage(content="Hello! Ask me anything about the documents.")
            arg = next(iter(self.tools[0].args))
            return AIMessage(
                content="", tool_calls=[{"name": self.tools[0].name, "args": {arg: last_q}, "id": "call_rag"}]
            )
        return AIMessage(content="Hello! How can I help?")
