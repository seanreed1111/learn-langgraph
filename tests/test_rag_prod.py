"""Tests for the production RAG package. All offline + deterministic.

The testing pyramid for agents:
  1. pure functions (ingest, retrieval, ACL)         - fast, exhaustive
  2. graph logic with FAKE retrievers/models         - loops, budgets, abstention, state resets
  3. evals on a labelled dataset                     - quality regression gate
  4. API contract (auth, isolation)                  - HTTP layer
"""

import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lessons"))

for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
    os.environ[_k] = ""

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from offline_llm import HashingEmbeddings, OfflineChatModel  # noqa: E402

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from rag_prod.api import create_app  # noqa: E402
from rag_prod.app import USERS, build_app  # noqa: E402
from rag_prod.config import Settings  # noqa: E402
from rag_prod.evals.run_evals import run_all, thresholds_ok  # noqa: E402
from rag_prod.graph import Groundedness, build_graph  # noqa: E402
from rag_prod.ingest import Index, load_documents  # noqa: E402
from rag_prod.retrieval import HybridRetriever  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), index_path=tmp_path / "idx.json", checkpoint_db=str(tmp_path / "ck.sqlite"))


@pytest.fixture
def kb(tmp_path):
    dst = tmp_path / "kb"
    shutil.copytree(ROOT / "data" / "kb", dst)
    return dst


def ask(graph, question, user="alice", thread="t"):
    return graph.invoke({"messages": [{"role": "user", "content": question}]},
                        {"configurable": {"thread_id": thread}}, context=USERS[user])


# ------------------------------------------------------------------ 1. ingestion
def test_ingest_is_idempotent_and_incremental(settings, kb):
    idx = Index(settings, HashingEmbeddings(), "fp")
    first = idx.sync(kb)
    assert len(first.added_docs) == 8 and first.chunks > 0
    again = idx.sync(kb)
    assert not (again.added_docs or again.updated_docs or again.removed_docs)  # nothing re-embedded

    (kb / "pto_policy.md").write_text((kb / "pto_policy.md").read_text() + "\nNew rule.\n", encoding="utf-8")
    (kb / "remote_work.md").unlink()
    third = idx.sync(kb)
    assert third.updated_docs == ["pto_policy.md"] and third.removed_docs == ["remote_work.md"]
    assert all(d.metadata["source"] != "remote_work.md" for d in idx.all_documents())  # no stale chunks


def test_index_persists_and_rebuilds_on_embedding_change(settings, kb):
    Index(settings, HashingEmbeddings(), "fp-A").sync(kb)
    reloaded = Index(settings, HashingEmbeddings(), "fp-A")
    assert len(reloaded.sync(kb).unchanged_docs) == 8  # loaded from disk, nothing re-embedded
    changed = Index(settings, HashingEmbeddings(), "fp-B")  # different embedding model
    assert len(changed.sync(kb).added_docs) == 8  # forced full rebuild


def test_missing_access_label_defaults_to_internal_not_public(tmp_path):
    (tmp_path / "x.md").write_text("# no front matter\nsecret-ish", encoding="utf-8")
    assert load_documents(tmp_path)[0].metadata["access"] == "internal"


# ------------------------------------------------------------------ 2. access control
@pytest.mark.parametrize("levels", [{"public"}, {"public", "internal"}, {"hr"}, set()])
def test_retriever_never_returns_unpermitted_chunks(settings, kb, levels):
    idx = Index(settings, HashingEmbeddings(), "fp")
    idx.sync(kb)
    retr = HybridRetriever(idx, settings)
    for q in ["salary", "incident sev1", "pto days", "password", "pricing plan"]:
        assert all(d.metadata["access"] in levels for d in retr.search(q, levels))


# ------------------------------------------------------------------ 3. graph logic with fakes
class StubRetriever:
    """Returns junk until the query contains 'reimbursement' - forces the rewrite loop."""

    def __init__(self):
        self.queries = []

    def search(self, query, allowed):
        self.queries.append(query)
        meta = {"source": "expense_policy.md", "title": "Expense Policy", "access": "internal"}
        if "reimbursement" in query:
            return [Document(id="1", page_content="Expense reimbursement happens in payroll. Business travel expense.", metadata=meta)]
        return [Document(id="2", page_content="Completely unrelated sentence about diffusion video models.", metadata=meta)]


def make_graph(settings, model=None, retriever=None):
    return build_graph(settings, model or OfflineChatModel(), retriever or StubRetriever(), InMemorySaver())


def test_rewrite_loop_recovers(settings):
    retr = StubRetriever()
    out = ask(make_graph(settings, retriever=retr), "How do I get money back for a trip?")
    assert out["status"] == "answered" and out["rewrites"] == 1
    assert "reimbursement" in retr.queries[-1]  # the REWRITTEN query is what got searched


def test_rewrite_budget_is_enforced(settings):
    class Never(StubRetriever):
        def search(self, query, allowed):
            self.queries.append(query)
            return [Document(id="x", page_content="nothing relevant at all here", metadata={"source": "a.md", "title": "A", "access": "public"})]

    retr = Never()
    out = ask(make_graph(settings, retriever=retr), "What is quantum gravity?")
    assert out["status"] == "abstained" and out["rewrites"] == settings.max_rewrites
    assert len(retr.queries) == settings.max_rewrites + 1  # bounded: initial + N rewrites


def test_ungrounded_answers_are_not_returned(settings):
    class Ungrounded(OfflineChatModel):
        def with_structured_output(self, schema, **kw):
            if schema is Groundedness:
                return type("S", (), {"invoke": lambda s, m, *a, **k: Groundedness(grounded=False, reason="made up")})()
            return super().with_structured_output(schema)

    retr = StubRetriever()
    out = ask(make_graph(settings, model=Ungrounded(), retriever=retr), "reimbursement for travel expense?")
    assert out["status"] == "abstained"
    assert out["trace"].count("generate") == settings.max_generations  # retried, then gave up


def test_hallucinated_citation_numbers_are_dropped(settings):
    class BadCites(OfflineChatModel):
        def with_structured_output(self, schema, **kw):
            real = super().with_structured_output(schema)
            if schema.__name__ == "CitedAnswer":
                return type("S", (), {"invoke": lambda s, m, *a, **k: schema(answer="Totally made up.", citations=[99])})()
            return real

    out = ask(make_graph(settings, model=BadCites(), retriever=StubRetriever()), "reimbursement for travel expense?")
    assert out["status"] == "abstained"  # an answer citing nonexistent passages is not trusted


@pytest.mark.parametrize("text,fragment", [("   ", "enter a question"), ("x" * 5000, "too long")])
def test_guard_rejects_bad_input_without_calling_the_model(settings, text, fragment):
    class Boom(OfflineChatModel):
        def invoke(self, *a, **k):
            raise AssertionError("model must not be called")

        def with_structured_output(self, schema, **k):
            raise_ = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("model must not be called"))  # noqa: E731
            return type("S", (), {"invoke": raise_, "batch": raise_})()

    out = ask(make_graph(settings, model=Boom()), text)
    assert out["status"] == "rejected" and fragment in out["answer"]


def test_per_turn_state_resets_between_turns_on_same_thread(settings):
    class Never(StubRetriever):
        pass

    graph = make_graph(settings, retriever=Never())
    first = ask(graph, "What is quantum gravity?", thread="same")  # burns the rewrite budget
    assert first["rewrites"] == settings.max_rewrites
    second = ask(graph, "How do I get money back for a trip?", thread="same")
    assert second["trace"][0] == "guard_input" and second["rewrites"] == 1  # NOT 2+1: counters were reset
    assert second["status"] == "answered"


# ------------------------------------------------------------------ 4. evals as a CI gate
def test_eval_dataset_meets_thresholds(settings):
    graph, _, _ = build_app(settings)
    results, summary = run_all(graph)
    failures = [(r.id, r.problems) for r in results if not r.ok]
    assert thresholds_ok(summary), (summary, failures)
    assert summary["acl_leaks"] == 0


# ------------------------------------------------------------------ 5. API contract
@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def post(client, user, question, **extra):
    return client.post("/chat", json={"question": question, **extra}, headers={"X-API-Key": f"key-{user}"})


def test_api_requires_valid_key(client):
    assert client.post("/chat", json={"question": "hi"}).status_code == 401
    assert client.post("/chat", json={"question": "hi"}, headers={"X-API-Key": "nope"}).status_code == 401


def test_api_health_and_ready(client):
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/readyz").json()["ready"] is True


def test_api_answers_with_citations_and_enforces_acl(client):
    ok = post(client, "alice", "How many PTO days do I accrue per year?").json()
    assert ok["status"] == "answered" and ok["citations"][0]["source"] == "pto_policy.md"
    denied = post(client, "alice", "What salary does an L4 engineer earn?").json()
    assert denied["status"] == "abstained" and "135,000" not in denied["answer"]


def test_api_threads_are_isolated_between_users(client):
    a = post(client, "alice", "How many PTO days do I accrue per year?", thread_id="shared").json()
    b = post(client, "bob", "hello!", thread_id="shared").json()
    assert a["thread_id"] == b["thread_id"] == "shared"  # same client-visible id...
    # ...but different server-side threads: bob's state is untouched by alice's history.
    state_b = client.app.state.graph.get_state({"configurable": {"thread_id": "bob:shared"}}).values
    state_a = client.app.state.graph.get_state({"configurable": {"thread_id": "alice:shared"}}).values
    assert len(state_b["messages"]) == 2 and len(state_a["messages"]) == 2
    assert state_a["messages"][0].content != state_b["messages"][0].content


def test_api_validates_input_and_streams(client):
    assert post(client, "alice", "").status_code == 422
    assert post(client, "alice", "q", thread_id="bad id!").status_code == 422
    with client.stream("POST", "/chat/stream", json={"question": "How many PTO days do I accrue per year?"},
                       headers={"X-API-Key": "key-alice"}) as r:
        body = "".join(r.iter_text())
    assert '"event": "node"' in body and '"event": "final"' in body
