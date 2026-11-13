"""Entrypoint for `langgraph dev` (see langgraph.json). The platform manages persistence, so we compile
WITHOUT a checkpointer. Pass the caller's identity as the run `context`, e.g. {"user_id": "sam", "role": "support"};
a run without one gets no access (fail closed)."""

from sql_prod.app import ensure_demo_db
from sql_prod.audit import AuditLog
from sql_prod.config import Settings
from sql_prod.graph import build_graph
from sql_prod.llm import get_writer

_settings = Settings.from_env()
ensure_demo_db(_settings)
graph = build_graph(_settings, get_writer(_settings), AuditLog(_settings.audit_db), checkpointer=None)
