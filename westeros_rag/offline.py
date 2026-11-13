"""Offline TEST DOUBLE for this lesson's structured calls (section tagging, query planning, grading, answering).

Extends lessons/offline_llm.OfflineChatModel. Real chat models do all of this with prompts + structured output;
these rules exist so the whole pipeline - and its evals - run deterministically with no key and no network.
"""

from __future__ import annotations

import re
import sys

from westeros_rag.config import ROOT

sys.path.insert(0, str(ROOT / "lessons"))
from offline_llm import (  # noqa: E402
    GREETING, STOP, THESAURUS, OfflineChatModel, _between, _context_from, _prompt, _question_from, _strip_md,
    _Structured, expand, stem, words,
)

# theme -> keywords (used for BOTH section tagging and query theme detection in offline mode)
THEME_KEYWORDS = {
    "homicide": ["murder", "kinslaying", "kinslayer", "slay", "kill", "homicide", "blood price"],
    "theft": ["theft", "thief", "robbery", "poach", "steal", "stole"],
    "treason": ["treason", "traitor", "attainder", "forfeit", "extinct"],
    "trial": ["trial", "combat", "ordeal", "witness", "verdict", "appeal", "accus", "evidence", "champion"],
    "punishment": ["punish", "penalty", "flog", "hang", "behead", "maim", "fine", "exile", "outlaw", "sentence"],
    "oath": ["oath", "vow", "sworn", "forsworn", "perjury"],
    "hospitality": ["guest", "host", "bread and salt"],
    "inheritance": ["inherit", "succession", "heir", "lordship", "regent", "ward", "dower", "primogeniture"],
    "marriage": ["marriage", "betroth", "dowry", "annul", "wed", "dissol"],
    "bastardy": ["bastard", "legitim"],
    "mercy": ["pardon", "mercy", "commut", "clemency"],
    "faith": ["septon", "sept", "faith", "sanctuary", "heresy", "blasphemy", "high septon"],
    "war": ["war", "siege", "hostage", "ransom", "quarter", "sack", "banner", "noncombatant"],
    "commerce": ["contract", "debt", "toll", "bank", "coin", "guild", "merchant", "creditor"],
    "governance": ["decree", "king", "hand of the king", "master of laws", "lord paramount", "high justice", "low justice"],
    "night_watch": ["night's watch", "the wall", "the black", "deserter", "lord commander"],
    "regional_custom": ["north", "riverlands", "vale", "reach", "westerlands", "stormlands", "dorne", "dornish", "iron islands", "crownlands"],
}

REGION_WORDS = {
    "north": ["north", "northern", "northmen"], "riverlands": ["riverlands", "riverrun"], "vale": ["the vale", "eyrie"],
    "reach": ["the reach", "highgarden"], "westerlands": ["westerlands", "casterly"], "stormlands": ["stormlands", "marcher"],
    "dorne": ["dorne", "dornish", "sunspear"], "ironislands": ["iron islands", "ironborn", "ironmen"],
    "crownlands": ["crownlands"], "wall": ["the wall", "night's watch"],
}

OFF_TOPIC = re.compile(r"capital of|weather|stock price|recipe|\bengland\b|\bsue\b|landlord|real[- ]world|bitcoin|football", re.I)


LAW_SYNONYMS = {
    "inheritance": "inherit inherits succession heir",
    "penalty": "punished punishment",
    "grant": "grants granted",
    "carries": "carry carried",
}

_RAW = re.compile(r"[a-z0-9]+")


def _syn_forms(raw: str) -> set[str]:
    """All normalized forms that count as a match for a question word (itself + synonyms)."""
    phrase = " ".join([raw, THESAURUS.get(raw, ""), LAW_SYNONYMS.get(raw, "")])
    return {_norm(w) for w in words(phrase)}


def question_terms(q: str) -> list[tuple[str, set[str]]]:
    """[(canonical, {acceptable normalized forms})] for each content word of the ORIGINAL question."""
    out = []
    for raw in _RAW.findall(q.lower()):
        if raw in STOP or len(raw) <= 2:
            continue
        forms = _syn_forms(raw)
        if forms:
            out.append((_norm(stem(raw)), forms))
    return out


def coverage(q: str, tokens: set[str]) -> tuple[float, int]:
    """Fraction (and count) of the question's own words matched by `tokens`, directly or via synonyms.
    Synonyms widen what MATCHES but never inflate the denominator (the bug in my first grader)."""
    terms = question_terms(q)
    hits = sum(1 for _c, forms in terms if forms & tokens)
    return (hits / len(terms) if terms else 0.0), hits


def _norm(w: str) -> str:
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) > len(suf) + 3:
            w = w[: -len(suf)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def lw(text: str) -> list[str]:
    """Law-double tokenizer: base words + light stemming (sentenced/sentence, witnesses/witness)."""
    return [_norm(w) for w in words(text)]


def lexpand(question: str) -> str:
    extra = " ".join(LAW_SYNONYMS[w] for w in re.findall(r"[a-z]+", question.lower()) if w in LAW_SYNONYMS)
    return f"{expand(question)} {extra}".strip()


def score_themes(text: str, heading: str = "") -> list[str]:
    low, head = text.lower(), heading.lower()
    scored = []
    for theme, kws in THEME_KEYWORDS.items():
        s = sum(low.count(k) for k in kws) + 3 * sum(head.count(k) for k in kws)
        if s > 0:
            scored.append((s, theme))
    return [t for _, t in sorted(scored, key=lambda x: (-x[0], x[1]))[:3]]


def detect_region(question: str) -> str | None:
    low = question.lower()
    for region, kws in REGION_WORDS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", low) for k in kws):
            return region
    return None


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", _strip_md(text)) if s.strip()]


class _LawStructured(_Structured):
    def invoke(self, messages, *a, **k):  # noqa: ANN001
        p = _prompt(messages)
        name = self.schema.__name__
        if name == "SectionMeta":
            text = _between(p, "<text>", "</text>")
            heading = _between(p, "<heading>", "</heading>")
            sents = _sentences(text)
            summary = " ".join(sents[:2])[:320] or heading
            return self.schema(summary=summary, themes=score_themes(text, heading))
        if name == "QueryPlan":
            q = _between(p, "<question>", "</question>")
            if GREETING.match(q):
                route = "chitchat"
            elif OFF_TOPIC.search(q):
                route = "off_topic"
            else:
                route = "law"
            parts = [s.strip(" ,;?") for s in re.split(r",\s*and\s+|;\s*|\s+and also\s+", q) if len(s.split()) >= 4]
            subs = [s + "?" for s in parts] if len(parts) > 1 else [q]
            return self.schema(route=route, themes=score_themes(q), jurisdiction=detect_region(q), sub_questions=subs)
        if name == "DocGrade":
            ratio, hits = coverage(_question_from(p), set(lw(_context_from(p))))
            ok = ratio >= 0.5 or (hits >= 3 and ratio >= 0.3)
            return self.schema(binary_score="yes" if ok else "no")
        if name == "CitedAnswer":
            return self._answer(p)
        return super().invoke(messages, *a, **k)

    def _answer(self, p: str):
        q = _question_from(p)
        ctx = _context_from(p)
        passages = re.findall(r"\[(\d+)\] ([^\n]*)\n(.*?)(?=\n\[\d+\]|\Z)", ctx, re.S)
        # answer each sub-question separately (multi-topic questions), at most 3 sentences overall
        parts = [s for s in re.split(r",\s*and\s+|;\s*", q) if len(s.split()) >= 3] or [q]
        # IDF-like weighting INSIDE the retrieved set: a word that appears in one passage ("landed", "knight") says
        # more than one found in every passage ("blood", "price"). A real LLM does this implicitly.
        # a reader sees a sentence TOGETHER with its section heading ("Poaching and Forest Law"): include it
        sentences = [(int(pid), s) for pid, _h, t in passages for s in _sentences(t)]
        heading_of = {int(pid): h for pid, h, _t in passages}
        df: dict[str, int] = {}  # PASSAGE-level rarity: in how many passages does the word occur (text or heading)?
        for pid, h, t in passages:
            for w in set(lw(t)) | set(lw(h)):
                df[w] = df.get(w, 0) + 1

        def score(query: str, sent: str, pid: int = 0) -> float:
            toks = set(lw(sent)) | set(lw(heading_of.get(pid, "")))
            total = 0.0
            for _canon, forms in question_terms(query):
                hit = forms & toks
                if hit:  # a rare matched word is worth more than one that appears in every passage
                    total += 1.0 / min(df.get(h, 1) for h in hit)
            return total

        picked: list[tuple[int, str]] = []
        for part in parts:
            ranked = sorted(sentences, key=lambda ps: -score(part, ps[1], ps[0]))
            for pid, sent in ranked:
                if score(part, sent, pid) <= 0:
                    break
                if sent not in [s for _, s in picked]:  # never repeat a sentence that appears in two passages
                    picked.append((pid, sent))
                    break
        if len(picked) == 1:  # one-topic question: add the next best sentence for context, if it is close enough
            top = score(q, picked[0][1], picked[0][0])
            for pid, sent in sorted(sentences, key=lambda ps: -score(q, ps[1], ps[0])):
                if score(q, sent, pid) >= 0.6 * top and sent not in [s for _, s in picked]:
                    picked.append((pid, sent))
                    break
        if not picked:
            return self.schema(answer="I don't know based on the provided passages.", citations=[])
        picked = picked[:3]
        return self.schema(answer=" ".join(s for _, s in picked), citations=sorted({pid for pid, _ in picked}))


class OfflineLawModel(OfflineChatModel):
    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001
        return _LawStructured(schema)

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return OfflineLawModel()
