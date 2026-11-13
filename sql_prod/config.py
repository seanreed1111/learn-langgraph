"""Central, environment-driven configuration. Every limit the safety story depends on lives here, so it can
be changed per environment, asserted in tests and swept in evals (lesson 14 does the same for RAG)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".sql_prod"


@dataclass(frozen=True)
class Settings:
    # storage (demo: SQLite files. Production: your warehouse + Postgres for audit and checkpoints)
    db_path: Path = STATE_DIR / "shop.sqlite"
    audit_db: Path = STATE_DIR / "audit.sqlite"
    checkpoint_db: str = str(STATE_DIR / "checkpoints.sqlite")

    # loop budgets
    max_repairs: int = 2

    # blast-radius limits, enforced where they cannot be talked around (see db.py)
    max_rows: int = 100               # rows returned to a caller; the DB cursor stops at max_rows + 1
    max_write_rows: int = 5           # a write touching more rows than this is refused
    statement_timeout_s: float = 2.0  # wall-clock budget per statement (progress handler)
    max_value_bytes: int = 1_000_000  # SQLITE_LIMIT_LENGTH: caps any string/blob a query can build
    max_sql_chars: int = 4_000

    # approvals
    approval_ttl_s: float = 900.0     # an approval older than this is void: the data may have moved

    # input / API
    max_question_chars: int = 1_000
    rate_limit_per_minute: int = 30

    # models
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    @classmethod
    def from_env(cls) -> Settings:
        def env_int(name: str, default: int) -> int:
            return int(os.getenv(name, default))

        def env_float(name: str, default: float) -> float:
            return float(os.getenv(name, default))

        return cls(
            db_path=Path(os.getenv("SQL_DB_PATH", cls.db_path)),
            audit_db=Path(os.getenv("SQL_AUDIT_DB", cls.audit_db)),
            checkpoint_db=os.getenv("SQL_CHECKPOINT_DB", cls.checkpoint_db),
            max_repairs=env_int("SQL_MAX_REPAIRS", cls.max_repairs),
            max_rows=env_int("SQL_MAX_ROWS", cls.max_rows),
            max_write_rows=env_int("SQL_MAX_WRITE_ROWS", cls.max_write_rows),
            statement_timeout_s=env_float("SQL_STATEMENT_TIMEOUT_S", cls.statement_timeout_s),
            approval_ttl_s=env_float("SQL_APPROVAL_TTL_S", cls.approval_ttl_s),
            rate_limit_per_minute=env_int("SQL_RATE_LIMIT_PER_MINUTE", cls.rate_limit_per_minute),
        )
