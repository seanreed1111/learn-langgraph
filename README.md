# LangGraph: Zero to Hero

A hands-on, runnable path from "what's a graph?" to production-grade agents. Every lesson is a
single Python file with the explanation in the docstring, working code, and exercises.
**All lessons run offline** (scripted fake LLM) - add an API key and they use a real model.

Verified against: `langgraph 1.2.21`, `langchain 1.5`, `langchain-core 1.8`, Python 3.14 (requires 3.12+).

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U langgraph langchain langchain-anthropic langgraph-checkpoint-sqlite python-dotenv pytest
copy .env.example .env      # optional: add a key to use a real LLM
pytest -q                   # should be all green
```

## The path

| # | Lesson | You learn | Job-relevance |
|---|--------|-----------|---------------|
| 01 | `lessons/01_hello_graph.py` | State, nodes, edges, compile, invoke | Foundation |
| 02 | `lessons/02_state_and_reducers.py` | Reducers, `add_messages`, input/output schemas | Asked in every interview |
| 03 | `lessons/03_routing_and_loops.py` | Conditional edges, `Command`, cycles, recursion limit | Core agent control flow |
| 04 | `lessons/04_parallelism_and_send.py` | Fan-out/in, `Send` map-reduce | Latency, scale |
| 05 | `lessons/05_agent_from_scratch.py` | **ReAct agent by hand** | The big one |
| 06 | `lessons/06_prebuilt_agents.py` | `ToolNode`, `tools_condition`, `create_agent` | Speed + knowing when to drop down |
| 07 | `lessons/07_persistence_and_time_travel.py` | Checkpointers, threads, history, forking | Memory, debugging, durability |
| 08 | `lessons/08_human_in_the_loop.py` | `interrupt`, `Command(resume=)` | Why companies choose LangGraph |
| 09 | `lessons/09_streaming.py` | values/updates/messages/custom, async | UX in real products |
| 10 | `lessons/10_long_term_memory_and_runtime.py` | Store, namespaces, `Runtime`/context | Personalization |
| 11 | `lessons/11_subgraphs_and_multi_agent.py` | Subgraphs, supervisor, handoffs | Multi-agent systems |
| 12 | `lessons/12_production_reliability.py` | Retry, cache, tool errors, testing, deploy | Senior-level signal |
| 13 | `lessons/13_agentic_rag.py` | **Official agentic-RAG tutorial, to completion**: retriever tool, grader, rewrite loop | Most requested agent type |
| 14 | `lessons/14_production_rag.md` + `rag_prod/` | **Production RAG**: ingestion, hybrid search, ACLs, citations, groundedness, evals, API | What "senior" looks like |
| 15 | `lessons/15_westeros_rag.md` + `westeros_rag/` + `data/westeros/` | **Hierarchical + link-aware RAG** over a 22-doc linked legal corpus: parsing, lint gate, enrichment, retrieval, guardrails, evals + ablation, API | RAG over real document structure |
| 16 | `lessons/16_functional_api.py` | `@entrypoint`/`@task`, `previous`, `entrypoint.final`, replay rules, Graph vs Functional | Asked as "when would you not use StateGraph?" |
| 17 | `lessons/17_durable_execution.py` | Crash + resume, pending writes, durability modes, idempotent side effects, retry policy | Senior-level reliability |
| 18 | `lessons/18_pregel_runtime.py` | Super-steps, channels, why parallel nodes don't see each other, raw `Pregel` | The "how does it really work" answer |
| 19 | `lessons/19_gotchas.md` + `19_gotchas.py` | **Silent failures & anti-patterns**: in-place mutation, history resend, router typos, fan-in, 10007 recursion default, swallowed interrupts, shared thread_ids, model-supplied identity. Every claim is a runnable, tested demo | What separates intermediate from advanced |
| 20 | `lessons/20_sql_agent.md` + `sql_agent/` | **SQL agent with layered safety**: guard, read-only DB, rolled-back write preview, approval, repair loop, execution-accuracy evals | Official tutorial, made production-safe |
| ★ | `capstone/support_agent.py` | Everything together + approval gate | Portfolio piece |

Run any lesson: `python lessons/05_agent_from_scratch.py`

RAG extras (`pip install langchain-text-splitters numpy fastapi uvicorn httpx beautifulsoup4 requests`):
```powershell
python lessons/13_agentic_rag.py            # tutorial, offline by default (USE_WEB=1 fetches the docs' blog posts)
python -m rag_prod.evals.run_evals          # lesson 14 quality gate
uvicorn rag_prod.api:app --port 8000        # lesson 14 service (see lessons/14_production_rag.md)
```
Lesson 15 (linked markdown corpus):
```powershell
python -m westeros_rag.cli lint ; python -m westeros_rag.cli ingest
python -m westeros_rag.cli ask "Can a pardon bring back a traitor's lands?" --role commoner
python -m westeros_rag.evals.run_evals ; python -m westeros_rag.evals.ablate
uvicorn westeros_rag.api:app --port 8001
```
Lesson 20 (SQL agent):
```powershell
python -m sql_agent.cli "Which customers are from the UK?"
python -m sql_agent.cli "Cancel order 6" --approve
python -m sql_agent.evals.run_evals
```
Lessons 16-19 are plain scripts (19 also has a narrative, and `pytest tests/test_gotchas.py` pins its claims): `python lessons/16_functional_api.py`.

Also see `PRODUCTION_CHECKLIST.md`.

## How to study (suggested 3 weeks)

- **Week 1 - mechanics:** lessons 01-05. Do *every* exercise. Re-type lesson 05 from memory.
- **Week 2 - stateful agents:** lessons 06-10. Put your API key in `.env` and rerun 05/06/08 with a real model; watch tool calls in the output.
- **Week 3 - production:** lessons 11-12 + capstone. Then do the "Portfolio projects" below and read `INTERVIEW.md` aloud.

## Concepts in one page

- **Graph = State + Nodes + Edges.** Nodes return partial updates; reducers merge them.
- **Super-steps:** nodes scheduled together run in parallel; updates apply at the step boundary.
- **Checkpointer** = snapshot per step -> memory, pause/resume, fault tolerance, time travel.
- **`interrupt()`** pauses; **`Command(resume=...)`** continues. The node restarts from the top.
- **Store** = cross-thread long-term memory. **Runtime context** = per-run read-only config.
- **Graph API vs Functional API:** this tutorial teaches the Graph API (explicit, visual, most used). The Functional API (`@entrypoint`/`@task`) gives the same persistence/HITL features with plain Python control flow - skim <https://docs.langchain.com/oss/python/langgraph/functional-api.md> once you're comfortable.

## Portfolio projects (what gets you hired)

Build 2-3 of these, push to GitHub with a README, tests, and a diagram (`graph.get_graph().draw_mermaid()`):

1. **RAG agent with query rewriting + grading loop** (retrieve -> grade docs -> rewrite query or answer). Docs: "Build a custom RAG agent".
2. **SQL analyst agent** with read-only guardrails and human approval for anything non-SELECT. Docs: "Build a custom SQL agent".
3. **Supervisor multi-agent research team** (researcher, writer, reviewer) with a critique loop and a hard budget on steps/tokens.
4. **Support/ops agent with HITL** (extend the capstone: Postgres checkpointer, FastAPI + SSE streaming, LangSmith traces).
5. **An eval harness** - a dataset of conversations + assertions on tool calls/outcomes (LangSmith evals or plain pytest). Few candidates have this; it stands out.

## Skills beyond LangGraph that employers expect

LangChain basics (models, tools, structured output) - prompt engineering - RAG & vector DBs - **evals/observability (LangSmith)** - FastAPI/async Python - Postgres/Docker - cost & latency control - guardrails/security (prompt injection, tool permissions).

## Official docs map

- Overview: <https://docs.langchain.com/oss/python/langgraph/overview>
- Graph API: <https://docs.langchain.com/oss/python/langgraph/graph-api.md>
- Persistence: <https://docs.langchain.com/oss/python/langgraph/persistence.md>
- Interrupts: <https://docs.langchain.com/oss/python/langgraph/interrupts.md>
- Streaming: <https://docs.langchain.com/oss/python/langgraph/streaming.md>
- Stores/Memory: <https://docs.langchain.com/oss/python/langgraph/stores.md>
- Subgraphs: <https://docs.langchain.com/oss/python/langgraph/use-subgraphs.md>
- Workflows & agents: <https://docs.langchain.com/oss/python/langgraph/workflows-agents.md>
- Thinking in LangGraph: <https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph.md>
- Deploy: <https://docs.langchain.com/oss/python/langgraph/deploy.md>
- Full index: <https://langchain-ai.github.io/langgraph/llms.txt>

> API note: LangGraph moves fast. If something errors after an upgrade, check the docs' changelog first. Examples here were run against the versions listed above.
