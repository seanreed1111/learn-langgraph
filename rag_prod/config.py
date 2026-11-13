"""Central, environment-driven configuration. No magic numbers scattered through the code:
every tunable lives here so it can be changed per environment and swept in evals."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    kb_dir: Path = ROOT / "data" / "kb"
    index_path: Path = ROOT / ".rag_index" / "index.json"
    checkpoint_db: str = str(ROOT / ".rag_index" / "checkpoints.sqlite")

    # chunking
    chunk_size: int = 600
    chunk_overlap: int = 100

    # retrieval
    top_k: int = 8  # candidates pulled from EACH retriever (vector + keyword)
    final_k: int = 4  # chunks kept after fusion

    # agent loop budgets - every loop is bounded
    max_rewrites: int = 2
    max_generations: int = 2

    # input / output guards
    max_question_chars: int = 2000

    # models
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        def env_int(name: str, default: int) -> int:
            return int(os.getenv(name, default))

        return cls(
            kb_dir=Path(os.getenv("RAG_KB_DIR", cls.kb_dir)),
            index_path=Path(os.getenv("RAG_INDEX_PATH", cls.index_path)),
            checkpoint_db=os.getenv("RAG_CHECKPOINT_DB", cls.checkpoint_db),
            chunk_size=env_int("RAG_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=env_int("RAG_CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=env_int("RAG_TOP_K", cls.top_k),
            final_k=env_int("RAG_FINAL_K", cls.final_k),
            max_rewrites=env_int("RAG_MAX_REWRITES", cls.max_rewrites),
            max_generations=env_int("RAG_MAX_GENERATIONS", cls.max_generations),
        )
