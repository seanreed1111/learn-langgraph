"""Entrypoint for `langgraph dev` (see langgraph.json). Studio manages persistence, so no checkpointer."""

from pathlib import Path

from sql_agent.db import create_demo_db
from sql_agent.graph import build_graph
from sql_agent.llm import get_writer

_path = create_demo_db(Path(__file__).resolve().parent.parent / ".sql_demo.sqlite")
graph = build_graph(_path, get_writer(), checkpointer=None)
