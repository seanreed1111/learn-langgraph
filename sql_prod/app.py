"""Composition root: settings -> database -> writer -> audit -> graph. Everything that touches the outside
world is created here, once, and injected (the same pattern as rag_prod/app.py)."""

from __future__ import annotations

import time
from collections.abc import Callable

from langgraph.checkpoint.memory import InMemorySaver

from sql_prod import db
from sql_prod.audit import AuditLog
from sql_prod.config import Settings
from sql_prod.graph import build_graph
from sql_prod.llm import get_writer
from sql_prod.policy import SqlContext

# Demo identities. In production your auth layer authenticates (OIDC/JWT) and maps groups -> role.
# Never trust a client-supplied role.
USERS = {
    "vera": SqlContext("vera", "viewer"),
    "anna": SqlContext("anna", "analyst"),
    "sam": SqlContext("sam", "support"),
    "maya": SqlContext("maya", "manager"),
    "omar": SqlContext("omar", "manager"),  # a second approver, so managers can review each other
}


def ensure_demo_db(settings: Settings) -> None:
    """DEMO ONLY: create the sample database if it is missing. A real deployment points at an existing one."""
    if not settings.db_path.exists():
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        db.create_demo_db(settings.db_path)


def build_app(settings: Settings | None = None, checkpointer=None, writer=None, audit: AuditLog | None = None,
              clock: Callable[[], float] = time.time):
    """Returns (graph, audit)."""
    settings = settings or Settings.from_env()
    ensure_demo_db(settings)
    audit = audit or AuditLog(settings.audit_db)
    graph = build_graph(settings, writer or get_writer(settings), audit, checkpointer or InMemorySaver(), clock)
    return graph, audit
