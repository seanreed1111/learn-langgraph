"""GUARDRAILS: defence in depth around the model. No single layer is trusted.

 INPUT   1. validate     : empty / too long
         2. injection    : known prompt-injection phrasings are refused (cheap heuristic first; add an LLM classifier
                           or a hosted guard model as a second layer in production)
         3. scope        : off-topic questions are refused politely (done by the planner's `route`)
 RETRIEVAL 4. ACCESS     : permission + jurisdiction filters inside the search (see retrieval.py)
           5. sanitize   : retrieved text is UNTRUSTED. Neutralize delimiter break-outs and instruction-like lines
                           before it is placed inside the prompt
 OUTPUT  6. citations    : an answer must cite passages that were actually provided
         7. leak check   : the answer must not reproduce text from documents the caller may not read
         8. groundedness : (in graph.py) a verifier checks every claim against the cited passages
         9. abstain      : when any check fails the system says "I don't know", never improvises

Heuristics are NOT a complete defence against prompt injection. They remove the cheap attacks, the structural
controls (access filters, citations, verifier) limit the damage of the rest, and red-team evals measure both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from westeros_rag.parse import Corpus

_QUALIFIERS = r"(?:(?:all|any|the|your|my|previous|prior|above|earlier|these|those)\s+){1,4}"  # "all of your previous"...
INJECTION_PATTERNS = [
    rf"\b(?:ignore|disregard|forget|override)\s+{_QUALIFIERS}(?:instructions|rules|prompts?|guidelines)",
    r"(reveal|show|print|repeat|output) (me )?(your|the) (system|hidden|initial) (prompt|instructions)",
    r"you are now\b",
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
    r"pretend (that )?(you|there) (are|is|have|has) no (rules|restrictions|limits)",
    r"bypass (the )?(access|security|permission|guardrail)",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.I)

# lines inside RETRIEVED text that try to talk to the model
_INSTRUCTION_LINE = re.compile(r"^\s*(system|assistant|developer|user)\s*:|^\s*(ignore|disregard|forget)\b.*\b(instructions|rules|above)\b", re.I)
_DELIMITERS = re.compile(r"</?\s*(context|question|answer|history|text|heading)\s*>", re.I)


@dataclass
class GuardResult:
    ok: bool
    category: str = ""
    message: str = ""


def check_input(text: str, max_chars: int) -> GuardResult:
    text = text.strip()
    if not text:
        return GuardResult(False, "empty", "Please enter a question.")
    if len(text) > max_chars:
        return GuardResult(False, "too_long", f"Question is too long (max {max_chars} characters).")
    if _INJECTION.search(text):
        return GuardResult(False, "injection", "I can't act on instructions that try to change my rules. Ask a question about the laws of the Seven Kingdoms.")
    return GuardResult(True)


def sanitize_passage(text: str) -> tuple[str, int]:
    """Returns (safe_text, number_of_changes). Escape our own delimiters so a document can't close <context>
    and start writing instructions; drop lines that address the model."""
    changes = 0
    safe_lines = []
    for line in text.splitlines():
        if _INSTRUCTION_LINE.search(line):
            changes += 1
            continue
        safe_lines.append(line)
    safe = "\n".join(safe_lines)
    safe, n = _DELIMITERS.subn(lambda m: m.group(0).replace("<", "‹").replace(">", "›"), safe)
    return safe, changes + n


class LeakDetector:
    """Detects reproduction of text from documents the caller may NOT read, using word shingles.

    For a given access set: forbidden = shingles of unreadable docs MINUS shingles that also occur in readable docs
    (so shared phrases like cross-reference link text never cause false alarms)."""

    def __init__(self, corpus: Corpus, n: int = 6) -> None:
        self.n, self.corpus = n, corpus
        self._by_access: dict[str, set[str]] = {}
        for doc in corpus.docs.values():
            acc = doc.meta.get("access", "internal")
            text = " ".join(corpus.unit_text(s.section_id) for s in doc.sections if s.level <= 2)
            self._by_access.setdefault(acc, set()).update(self._shingles(text))
        self._cache: dict[frozenset, set[str]] = {}

    def _shingles(self, text: str) -> set[str]:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        return {" ".join(toks[i : i + self.n]) for i in range(max(len(toks) - self.n + 1, 0))}

    def forbidden_for(self, allowed: frozenset[str]) -> set[str]:
        if allowed not in self._cache:
            hidden = set().union(*[s for a, s in self._by_access.items() if a not in allowed]) if self._by_access else set()
            readable = set().union(*[s for a, s in self._by_access.items() if a in allowed]) if self._by_access else set()
            self._cache[allowed] = hidden - readable
        return self._cache[allowed]

    def leaks(self, answer: str, allowed) -> list[str]:
        forbidden = self.forbidden_for(frozenset(allowed))
        return sorted(self._shingles(answer) & forbidden)[:3]
