# Lessons: files, tests and how to run them

Per-lesson plans (objectives, exercises, check-your-understanding answers) are in the main
[README](../README.md#curriculum-at-a-glance). This table maps each lesson to the executable code and the
tests that prove it, so you can read the behavior as an assertion.

| # | Run it | Tests that cover it |
|---|---|---|
| 01 Hello graph | `python lessons/01_hello_graph.py` | `test_graphs.py::test_nodes_are_plain_functions`, `test_smoke.py` |
| 02 State and reducers | `python lessons/02_state_and_reducers.py` | `test_graphs.py` (reducers), `test_smoke.py` |
| 03 Routing and loops | `python lessons/03_routing_and_loops.py` | `test_smoke.py` |
| 04 Parallelism and Send | `python lessons/04_parallelism_and_send.py` | `test_graphs.py::test_map_reduce`, `test_smoke.py` |
| 05 ReAct from scratch | `python lessons/05_agent_from_scratch.py` | `test_graphs.py::test_agent_calls_tool_then_answers`, `test_smoke.py` |
| 06 Prebuilt agents | `python lessons/06_prebuilt_agents.py` | `test_smoke.py` |
| 07 Persistence | `python lessons/07_persistence_and_time_travel.py` | `test_smoke.py` |
| 08 Human-in-the-loop | `python lessons/08_human_in_the_loop.py` | `test_graphs.py::test_interrupt_and_resume`, `test_smoke.py` |
| 09 Streaming | `python lessons/09_streaming.py` | `test_smoke.py` |
| 10 Long-term memory | `python lessons/10_long_term_memory_and_runtime.py` | `test_smoke.py` |
| 11 Multi-agent | `python lessons/11_subgraphs_and_multi_agent.py` | `test_smoke.py` |
| 12 Reliability | `python lessons/12_production_reliability.py` | `test_smoke.py` |
| 13 Agentic RAG | `python lessons/13_agentic_rag.py` | `test_graphs.py::test_agentic_rag_lesson_paths`, `test_smoke.py` |
| 14 Production RAG | `uvicorn rag_prod.api:app`; `python -m rag_prod.evals.run_evals` | `test_rag_prod.py`, `test_smoke.py` (eval gate) |
| 15 Hierarchical RAG | `python -m westeros_rag.cli ask "..."` | `test_westeros.py`, `test_smoke.py` (lint CLI, eval gate) |
| 16 Functional API | `python lessons/16_functional_api.py` | `test_lessons_16_18.py`, `test_smoke.py` |
| 17 Durable execution | `python lessons/17_durable_execution.py` | `test_lessons_16_18.py`, `test_smoke.py` |
| 18 Pregel runtime | `python lessons/18_pregel_runtime.py` | `test_lessons_16_18.py`, `test_smoke.py` |
| 19 Gotchas | `python lessons/19_gotchas.py` | `test_gotchas.py`, `test_smoke.py` |
| 20 SQL agent | `python -m sql_agent.cli "How many customers do we have?"` | `test_sql_agent.py`, `test_smoke.py` (CLI, eval gate) |
| 21 LangSmith tracing | `python lessons/21_langsmith_tracing.py` | `test_langsmith_lessons.py`, `test_smoke.py` |
| 22 LangSmith evaluation | `python lessons/22_langsmith_evaluation.py` | `test_langsmith_lessons.py`, `test_smoke.py` |
| 23 LangSmith production loop | `python lessons/23_langsmith_production_loop.py` | `test_langsmith_lessons.py`, `test_smoke.py` |
| ★ Capstone | `python capstone/support_agent.py` | `test_graphs.py::test_capstone_approval_gate`, `test_smoke.py` |

`common.py`, `offline_llm.py`, `trace_tools.py` (fake LangSmith server, run recorder, mini experiment harness) and `demo_agent.py` (the deterministic agent for lessons 21-23) are shared helpers (scripted fake chat model, offline embeddings); the model
id and provider selection live in `common.py` only.
