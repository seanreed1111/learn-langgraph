"""CLI: python -m sql_prod.cli "Cancel order 6" --user sam [--approve-as maya] [--audit]

Each run uses a fresh temporary demo database, so you cannot damage anything. `--user` picks who is asking
(vera=viewer, anna=analyst, sam=support, maya=manager); try the same question as different people.
"""

from __future__ import annotations

import argparse
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from sql_prod.app import USERS, build_app
from sql_prod.config import Settings


def main() -> None:
    ap = argparse.ArgumentParser(prog="sql_prod")
    ap.add_argument("question")
    ap.add_argument("--user", default="anna", choices=sorted(USERS), help="who is asking (default: anna, an analyst)")
    ap.add_argument("--approve-as", default=None, choices=sorted(USERS), metavar="USER",
                    help="resume a pending write as this user (try maya, then try sam, or the same user as --user)")
    ap.add_argument("--audit", action="store_true", help="print the audit trail for this query")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    settings = replace(Settings.from_env(), db_path=tmp / "shop.sqlite", audit_db=tmp / "audit.sqlite")
    graph, audit = build_app(settings)
    query_id = uuid.uuid4().hex
    cfg: RunnableConfig = {"configurable": {"thread_id": query_id}}

    out = graph.invoke({"question": a.question}, cfg, context=USERS[a.user])
    if "__interrupt__" in out:
        req = out["__interrupt__"][0].value
        print(f"APPROVAL NEEDED (requested by {req['requested_by']}): {req['sql']}")
        print(f"  would change {req['rows_affected_preview']} row(s); fingerprint {req['sql_sha256'][:12]}...")
        if not a.approve_as:
            print("not approved (pass --approve-as maya). Nothing was changed.")
            return
        out = graph.invoke(Command(resume={"approve": True, "sql_sha256": req["sql_sha256"]}), cfg,
                           context=USERS[a.approve_as])
    print(f"SQL    : {out.get('sql')}")
    print(f"STATUS : {out['status']}")
    print(f"ANSWER : {out['answer']}")
    print(f"TRACE  : {' -> '.join(out['trace'])}")
    if a.audit:
        for e in audit.recent(50, query_id=query_id):
            print(f"AUDIT  : {e['ts']} {e['event']:<18} {e['user_id']:<6} {e['detail']}")


if __name__ == "__main__":
    main()
