"""FastAPI service around the SQL analyst graph.

    uvicorn sql_prod.api:app --port 8001
    curl -s localhost:8001/ask -H "X-API-Key: key-sam" -H "content-type: application/json" \
         -d '{"question": "Cancel order 6"}'                     # -> status "pending_approval" + query_id
    curl -s localhost:8001/queries/<query_id>/decision -H "X-API-Key: key-maya" -H "content-type: application/json" \
         -d '{"approve": true, "sql_sha256": "<from the pending response>"}'

Production concerns handled here (lesson 14's list, adapted to a tool that can change data):
  * AUTHN -> CONTEXT: the API key resolves to a `SqlContext` (user + role). A client never picks its own role.
  * TWO PEOPLE FOR A WRITE: the requester gets a `query_id`; a DIFFERENT user whose role can approve submits
    the decision. The service checks this (403) and the graph checks it again (defence in depth).
  * APPROVE WHAT YOU SAW: the decision must carry the `sql_sha256` from the pending response. Changed
    statement -> different hash -> the graph refuses.
  * OWNERSHIP: a query id is a capability, not an authorization. Reading a query needs to be its requester or
    an approver; anyone else gets 404 (not 403, so ids cannot be probed).
  * DURABLE APPROVALS: a SqliteSaver holds pending approvals, so a restart does not lose them. Use Postgres
    when you run several replicas.
  * RATE LIMITING per user (429 + Retry-After); input validation; clean 502s with a request id; /healthz vs
    /readyz; an /audit endpoint for the roles allowed to read it.
Missing on purpose (see PRODUCTION_CHECKLIST.md): OIDC/JWT, async I/O, a metrics endpoint, a secrets manager.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi import Path as PathParam
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from rag_prod.observability import logger, setup_logging
from sql_prod import db
from sql_prod.app import USERS, build_app
from sql_prod.config import Settings
from sql_prod.policy import SqlContext

# Demo only: load from a secrets manager / identity provider in production.
API_KEYS = {f"key-{name}": name for name in USERS}

QueryId = Annotated[str, PathParam(pattern=r"^[0-9a-f]{32}$")]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class DecisionRequest(BaseModel):
    approve: bool
    sql_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class Approval(BaseModel):
    requested_by: str
    sql: str
    sql_sha256: str
    rows_affected_preview: int
    expires_at: float


class QueryResponse(BaseModel):
    request_id: str
    query_id: str
    status: str
    answer: str
    sql: str | None = None
    columns: list[str] = []
    rows: list[list[Any]] = []
    truncated: bool = False
    approval: Approval | None = None


class RateLimiter:
    """Sliding-window limiter, per key, in memory. One process only: use Redis (or your gateway) for replicas."""

    def __init__(self, limit_per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._limit, self._clock = limit_per_minute, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> float | None:
        """None if the call is allowed (and counted); otherwise seconds until it would be."""
        now = self._clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= 60.0:
                hits.popleft()
            if len(hits) >= self._limit:
                return max(0.0, 60.0 - (now - hits[0]))
            hits.append(now)
            return None


def authenticate(api_key: str | None) -> SqlContext:
    user = API_KEYS.get(api_key or "")
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return USERS[user]


def create_app(settings: Settings | None = None, writer=None, clock: Callable[[], float] = time.time,
               limiter: RateLimiter | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    limiter = limiter or RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
            app.state.graph, app.state.audit = build_app(settings, saver, writer, clock=clock)
            logger.info("startup", extra={"fields": {"service": "sql_prod"}})
            yield

    app = FastAPI(title="SQL analyst", lifespan=lifespan)

    def _config(query_id: str, ctx: SqlContext | None = None, request_id: str = "") -> dict[str, Any]:
        cfg: dict[str, Any] = {"configurable": {"thread_id": query_id}, "recursion_limit": 40}
        if ctx is not None:
            cfg["metadata"] = {"user_id": ctx.user_id, "role": ctx.role, "request_id": request_id}
            cfg["tags"] = ["sql-api"]
        return cfg

    def _pending(snapshot) -> dict[str, Any] | None:
        for task in snapshot.tasks:
            if task.interrupts:
                return task.interrupts[0].value
        return None

    def _response(request_id: str, query_id: str, values: dict[str, Any], pending: dict[str, Any] | None) -> QueryResponse:
        approval = None
        status = values.get("status", "")
        if pending is not None:
            status = "pending_approval"
            approval = Approval(requested_by=pending["requested_by"], sql=pending["sql"], sql_sha256=pending["sql_sha256"],
                                rows_affected_preview=pending["rows_affected_preview"], expires_at=pending["expires_at"])
        return QueryResponse(
            request_id=request_id, query_id=query_id, status=status,
            answer=values.get("answer") or ("Waiting for approval." if pending is not None else ""),
            sql=values.get("sql") or None, columns=values.get("columns", []), rows=values.get("rows", []),
            truncated=values.get("truncated", False), approval=approval)

    def _load(query_id: str):
        snapshot = app.state.graph.get_state(_config(query_id))
        return snapshot if snapshot.values else None

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        if getattr(app.state, "graph", None) is None or not db.ping(settings.db_path):
            raise HTTPException(status_code=503, detail="not ready")
        return {"ready": True}

    @app.post("/ask", response_model=QueryResponse)
    def ask(req: AskRequest, x_api_key: str | None = Header(default=None)) -> QueryResponse:
        ctx = authenticate(x_api_key)
        wait = limiter.retry_after(ctx.user_id)
        if wait is not None:
            raise HTTPException(status_code=429, detail="rate limit exceeded", headers={"Retry-After": str(int(wait) + 1)})
        request_id, query_id = uuid.uuid4().hex, uuid.uuid4().hex
        try:
            out = app.state.graph.invoke({"question": req.question}, _config(query_id, ctx, request_id), context=ctx)
        except Exception:  # details go to logs, never to the client
            logger.exception("ask_failed", extra={"fields": {"request_id": request_id, "query_id": query_id}})
            raise HTTPException(status_code=502, detail=f"upstream error (request_id={request_id})") from None
        pending = out["__interrupt__"][0].value if "__interrupt__" in out else None
        return _response(request_id, query_id, out, pending)

    @app.get("/queries/{query_id}", response_model=QueryResponse)
    def get_query(query_id: QueryId, x_api_key: str | None = Header(default=None)) -> QueryResponse:
        ctx = authenticate(x_api_key)
        snapshot = _load(query_id)
        mine = snapshot is not None and snapshot.values.get("requested_by") == ctx.user_id
        if snapshot is None or not (mine or ctx.policy.can_approve):
            raise HTTPException(status_code=404, detail="no such query")  # 404, not 403: do not confirm the id exists
        return _response(uuid.uuid4().hex, query_id, snapshot.values, _pending(snapshot))

    @app.post("/queries/{query_id}/decision", response_model=QueryResponse)
    def decide(query_id: QueryId, req: DecisionRequest, x_api_key: str | None = Header(default=None)) -> QueryResponse:
        ctx = authenticate(x_api_key)
        if not ctx.policy.can_approve:
            raise HTTPException(status_code=403, detail="your role cannot approve writes")
        snapshot = _load(query_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="no such query")
        if _pending(snapshot) is None:
            raise HTTPException(status_code=409, detail="this query is not waiting for approval")  # also: no double-apply
        if snapshot.values.get("requested_by") == ctx.user_id:
            raise HTTPException(status_code=403, detail="you cannot approve your own request")
        request_id = uuid.uuid4().hex
        try:  # the resume carries the APPROVER's context: that is who the graph audits and checks
            out = app.state.graph.invoke(Command(resume={"approve": req.approve, "sql_sha256": req.sql_sha256}),
                                         _config(query_id, ctx, request_id), context=ctx)
        except Exception:
            logger.exception("decision_failed", extra={"fields": {"request_id": request_id, "query_id": query_id}})
            raise HTTPException(status_code=502, detail=f"upstream error (request_id={request_id})") from None
        return _response(request_id, query_id, out, None)

    @app.get("/audit")
    def audit(limit: Annotated[int, Query(ge=1, le=500)] = 50, query_id: str | None = None,
              x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        ctx = authenticate(x_api_key)
        if not ctx.policy.can_audit:
            raise HTTPException(status_code=403, detail="your role cannot read the audit log")
        return {"events": app.state.audit.recent(limit, query_id=query_id)}

    return app


app = create_app()  # the ASGI entrypoint: `uvicorn sql_prod.api:app`
