"""FastAPI service for the legal-compendium RAG (same production patterns as rag_prod/api.py, plus section endpoints).

    uvicorn westeros_rag.api:app --port 8001
    curl -s localhost:8001/chat -H "X-API-Key: key-noble" -H "content-type: application/json" \
         -d '{"question": "How much blood price is owed for killing a landed knight?"}'

Endpoints
  POST /chat                  grounded answer + deep-link citations + trace/flags
  GET  /tree                  outline of the documents THIS caller may see (never lists restricted docs)
  GET  /sections/{doc_id}/{slug}   full text of a cited section ("doc.md#slug" -> /sections/doc/slug) - ACL-checked
  GET  /healthz, /readyz

Security notes: identity comes from the API key (demo) -> LawContext. /sections returns 404 (not 403) for restricted
sections so the existence of hidden documents is not revealed. Thread ids are namespaced per user.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from langgraph.checkpoint.sqlite import SqliteSaver

from rag_prod.observability import logger, setup_logging
from westeros_rag.config import ROLES, LawContext, WSettings
from westeros_rag.graph import DISCLAIMER
from westeros_rag.pipeline import build_pipeline

API_KEYS = {f"key-{role}": role for role in ROLES}  # demo only: use an IdP / secret manager in production


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ChatResponse(BaseModel):
    request_id: str
    thread_id: str
    status: str
    answer: str
    citations: list[dict]
    flags: list[str]
    disclaimer: str = DISCLAIMER


def authenticate(api_key: str | None) -> LawContext:
    role = API_KEYS.get(api_key or "")
    if not role:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return ROLES[role]


def create_app(settings: WSettings | None = None) -> FastAPI:
    settings = settings or WSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
            app.state.pipe = build_pipeline(settings, checkpointer=saver)
            logger.info("startup", extra={"fields": {"index": str(app.state.pipe.stats)}})
            yield

    app = FastAPI(title="Westeros Law RAG", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/readyz")
    def readyz() -> dict:
        if getattr(app.state, "pipe", None) is None:
            raise HTTPException(status_code=503, detail="not ready")
        return {"ready": True, "documents": len(app.state.pipe.corpus.docs)}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, x_api_key: str | None = Header(default=None)) -> ChatResponse:
        ctx = authenticate(x_api_key)
        thread, request_id = req.thread_id or uuid.uuid4().hex[:16], uuid.uuid4().hex
        config = {"configurable": {"thread_id": f"{ctx.user_id}:{thread}"}, "recursion_limit": 40,
                  "metadata": {"user_id": ctx.user_id, "request_id": request_id}, "tags": ["westeros-api"]}
        try:
            out = app.state.pipe.graph.invoke({"messages": [{"role": "user", "content": req.question}]}, config, context=ctx)
        except Exception:
            logger.exception("chat_failed", extra={"fields": {"request_id": request_id}})
            raise HTTPException(status_code=502, detail=f"upstream error (request_id={request_id})")
        return ChatResponse(request_id=request_id, thread_id=thread, status=out["status"], answer=out["answer"],
                            citations=out["citations"], flags=out.get("flags", []))

    @app.get("/tree")
    def tree(x_api_key: str | None = Header(default=None)) -> list[dict]:
        ctx, corpus = authenticate(x_api_key), app.state.pipe.corpus
        return [
            {"doc_id": d.doc_id, "title": d.title, "sections": [s.section_id for s in d.sections if s.level == 2]}
            for d in corpus.docs.values() if d.meta.get("access", "internal") in ctx.access_levels
        ]

    @app.get("/sections/{doc_id}/{slug}")  # NOT "{doc}#{slug}": a URL #fragment is never sent to the server
    def section(doc_id: str, slug: str, x_api_key: str | None = Header(default=None)) -> dict:
        ctx, corpus = authenticate(x_api_key), app.state.pipe.corpus
        unit_id = f"{doc_id}#{slug}"
        if unit_id not in corpus.sections or corpus.access(unit_id) not in ctx.access_levels:
            raise HTTPException(status_code=404, detail="section not found")  # 404, not 403: don't confirm existence
        u = corpus.unit_id(unit_id)
        return {"unit_id": u, "heading_path": corpus.sections[u].heading_path, "text": corpus.unit_text(u),
                "links_to": corpus.unit_links(u), "summary": app.state.pipe.enrich[u]["summary"]}

    return app


app = create_app()
