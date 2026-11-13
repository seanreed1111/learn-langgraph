"""CLI: python -m sql_agent.cli ask "How many customers do we have?" [--approve]"""

from __future__ import annotations

import argparse
import tempfile
import uuid
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from sql_agent.db import create_demo_db
from sql_agent.graph import build_graph
from sql_agent.llm import get_writer


def main() -> None:
    ap = argparse.ArgumentParser(prog="sql_agent")
    ap.add_argument("question")
    ap.add_argument("--db", default=None, help="path to a demo db (default: fresh temp copy)")
    ap.add_argument("--approve", action="store_true", help="auto-approve a pending write")
    a = ap.parse_args()

    path = Path(a.db) if a.db else create_demo_db(Path(tempfile.mkdtemp()) / "demo.sqlite")
    graph = build_graph(path, get_writer())
    cfg: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
    out = graph.invoke({"question": a.question}, cfg)
    if "__interrupt__" in out:
        req = out["__interrupt__"][0].value
        print(f"APPROVAL NEEDED: {req['sql']}  (would change {req['rows_affected_preview']} row(s))")
        if not a.approve:
            print("not approved (pass --approve). Nothing was changed.")
            return
        out = graph.invoke(Command(resume=True), cfg)
    print(f"SQL    : {out.get('sql')}")
    print(f"STATUS : {out['status']}")
    print(f"ANSWER : {out['answer']}")


if __name__ == "__main__":
    main()
