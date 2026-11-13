"""Settings, roles and the theme taxonomy. Every tunable lives here so evals can sweep it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Controlled vocabulary for section themes. LLM tagging is CONSTRAINED to this list (and validated):
# free-form tags drift ("trial by combat" / "combat trials" / "ordeals") and break metadata filtering.
THEMES = [
    "homicide", "theft", "treason", "trial", "punishment", "oath", "hospitality", "inheritance", "marriage",
    "bastardy", "mercy", "faith", "war", "commerce", "governance", "night_watch", "regional_custom",
]

REGIONS = ["north", "riverlands", "vale", "reach", "westerlands", "stormlands", "dorne", "ironislands", "crownlands", "wall"]


@dataclass(frozen=True)
class WSettings:
    corpus_dir: Path = ROOT / "data" / "westeros"
    index_path: Path = ROOT / ".westeros_index" / "index.json"
    enrich_cache: Path = ROOT / ".westeros_index" / "enrich.json"
    checkpoint_db: str = str(ROOT / ".westeros_index" / "checkpoints.sqlite")

    # chunking (leaf level)
    chunk_size: int = 450
    chunk_overlap: int = 60

    # retrieval
    # "flat" = search chunks (still hierarchical: heading-path chunks + parent expansion + links + themes)
    # "hier" = ALSO search section summaries and fuse. OPT-IN: on this corpus the ablation (evals/ablate.py) showed the
    # summary level did not improve recall. Turn it on only if YOUR ablation (real embeddings, LLM summaries) says so.
    strategy: str = "flat"
    top_sections: int = 6  # stage A: candidate sections from the summary index
    top_chunks: int = 12  # stage B: candidate chunks
    final_units: int = 4  # context units (H2 sections) handed to the LLM
    summary_weight: float = 0.5  # vote of the summary ranking vs the chunk ranking in unit-level RRF (1.0 = equal)
    use_parent: bool = True  # expand a matching chunk to its whole parent section
    use_links: bool = True  # follow cross-references one hop
    use_themes: bool = True  # boost sections whose theme tags match the query
    max_linked: int = 3
    link_depth: int = 1
    context_budget_chars: int = 7000
    theme_bonus: float = 0.004  # per matching theme, on the RRF scale (one rank step near the top is ~0.0003-0.0005)

    # agent budgets
    max_rewrites: int = 1
    max_generations: int = 2
    max_question_chars: int = 1500


@dataclass(frozen=True)
class LawContext:
    """Per-request identity from the auth layer. READ-ONLY; never produced by the LLM."""

    user_id: str
    access_levels: tuple[str, ...] = ("public",)


ROLES = {
    "commoner": LawContext("commoner", ("public",)),
    "noble": LawContext("noble", ("public", "noble")),
    "small_council": LawContext("small_council", ("public", "noble", "small_council")),
}
KNOWN_ACCESS = {"public", "noble", "small_council"}
