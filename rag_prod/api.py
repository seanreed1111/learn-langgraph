"""FastAPI service around the RAG graph.

    uvicorn rag_prod.api:app --port 8000
    curl -s localhost:8000/chat -H "X-API-Key: key-alice" -H "content-type: application/json" \
         -d '{"question": "How many PTO days do I get?"}'

Production concerns handled here (and which tutorials skip):
  * AUTHN -> context: the API key resolves to a RagContext (user + access levels). The client can
    never choose its own permissions. (Real systems: OIDC/JWT at a gateway; map groups -> access.)
  * THREAD ISOLATION: thread ids are namespaced `<user>:<client id>` so one user cannot read or
    continue another user's conversation by guessing an id.
  * INPUT VALIDATION: pydantic limits on question and thread id.
  * DURABLE MEMORY: SqliteSaver here; use PostgresSaver when you run >1 replica.
  * ERRORS: provider failures become a clean 502 (details go to logs, not to the client).
  * STREAMING: Server-Sent Events of node progress then the final answer.
  * HEALTH: /healthz (process up) vs /readyz (index loaded) for Kubernetes-style probes.
  * TRACING: user/request metadata attached to every run (visible in LangSmith when enabled).
Missing on purpose (see PRODUCTION_CHECKLIST.md): rate limiting, async I/O, metrics endpoint, secrets manager.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langgraph.checkpoint.sqlite import SqliteSaver

from rag_prod.app import USERS, build_app
from rag_prod.config import Settings
from rag_prod.graph import RagContext
from rag_prod.observability import logger, setup_logging

# Demo only: load from a secrets manager / identity provider in production.
API_KEYS = {f"key-{name}": name for name in USERS}


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ChatResponse(BaseModel):
    request_id: str
    thread_id: str
    status: str
    answer: str
    citations: list[dict]


def authenticate(api_key: str | None) -> RagContext:
    user = API_KEYS.get(api_key or "")
    if not user:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return USERS[user]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
            app.state.graph, app.state.index, stats = build_app(settings, saver)
            logger.info("startup", extra={"fields": {"index": str(stats)}})
            yield

    app = FastAPI(title="Agentic RAG", lifespan=lifespan)

    def _inputs(req: ChatRequest, ctx: RagContext):
        client_thread = req.thread_id or uuid.uuid4().hex[:16]
        request_id = uuid.uuid4().hex
        config = {
            "configurable": {"thread_id": f"{ctx.user_id}:{client_thread}"},  # namespaced per user
            "recursion_limit": 40,  # hard stop for runaway loops (we also have per-loop budgets)
            "metadata": {"user_id": ctx.user_id, "request_id": request_id},
            "tags": ["rag-api"],
        }
        return client_thread, request_id, config, {"messages": [{"role": "user", "content": req.question}]}

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/readyz")
    def readyz() -> dict:
        ready = getattr(app.state, "graph", None) is not None
        if not ready:
            raise HTTPException(status_code=503, detail="not ready")
        return {"ready": True, "chunks": len(app.state.index.store.store)}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, x_api_key: str | None = Header(default=None)) -> ChatResponse:
        ctx = authenticate(x_api_key)
        thread, request_id, config, payload = _inputs(req, ctx)
        try:
            out = app.state.graph.invoke(payload, config, context=ctx)
        except Exception:
            logger.exception("chat_failed", extra={"fields": {"request_id": request_id}})
            raise HTTPException(status_code=502, detail=f"upstream error (request_id={request_id})")
        return ChatResponse(request_id=request_id, thread_id=thread, status=out["status"],
                            answer=out["answer"], citations=out["citations"])

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest, x_api_key: str | None = Header(default=None)) -> StreamingResponse:
        ctx = authenticate(x_api_key)  # authenticate BEFORE opening the stream
        thread, request_id, config, payload = _inputs(req, ctx)

        def sse(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        def events():
            try:
                for chunk in app.state.graph.stream(payload, config, context=ctx, stream_mode="updates", version="v2"):
                    if chunk["type"] == "updates":
                        for node in chunk["data"]:
                            yield sse({"event": "node", "name": node})
                final = app.state.graph.get_state(config).values
                yield sse({"event": "final", "request_id": request_id, "thread_id": thread, "status": final["status"],
                           "answer": final["answer"], "citations": final["citations"]})
            except Exception:
                logger.exception("stream_failed", extra={"fields": {"request_id": request_id}})
                yield sse({"event": "error", "request_id": request_id})

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()  # the ASGI entrypoint: `uvicorn rag_prod.api:app`
