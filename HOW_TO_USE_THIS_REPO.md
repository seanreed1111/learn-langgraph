# How to use this repo

A practical guide to getting the most out of the LangGraph zero-to-hero curriculum: how to set it up, how to
work through the lessons, how to run the apps, and how to change things safely. The [README](README.md) has the
full lesson plans (objectives, exercises, check-your-understanding questions); this page is the operating manual.

## Contents

- [The shape of the repo](#the-shape-of-the-repo)
- [Set up in five minutes](#set-up-in-five-minutes)
- [How to work through a lesson](#how-to-work-through-a-lesson)
- [Which path should I take?](#which-path-should-i-take)
- [Offline by default, real model on demand](#offline-by-default-real-model-on-demand)
- [Running the three apps](#running-the-three-apps)
- [Visualizing and debugging graphs](#visualizing-and-debugging-graphs)
- [Running the checks](#running-the-checks)
- [Making changes safely](#making-changes-safely)
- [Troubleshooting](#troubleshooting)
- [Where to read next](#where-to-read-next)

## The shape of the repo

| You want to... | Go to |
|---|---|
| Learn the concepts in order | `lessons/` (run each file) and the lesson plans in the [README](README.md#curriculum-at-a-glance) |
| See which test proves which lesson | [`lessons/README.md`](lessons/README.md) |
| Study a production-style service | `rag_prod/` (lesson 14), `westeros_rag/` (lesson 15), `sql_agent/` (lesson 20) |
| Build something end to end | `capstone/support_agent.py` |
| Prepare for interviews | [`INTERVIEW.md`](INTERVIEW.md) |
| Trace, evaluate and monitor an agent | lessons 21-23 (LangSmith track; offline, no account needed) |
| Check what is left before shipping an agent | [`PRODUCTION_CHECKLIST.md`](PRODUCTION_CHECKLIST.md) |
| Use a real model instead of the fakes | [`docs/real-models.md`](docs/real-models.md) |
| Change the code or upgrade dependencies | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

Lessons 01-12, 16-18 and 21-23 are single Python files: the explanation is in the docstring, the code runs, and
exercises sit at the bottom. Lessons 13-15, 19 and 20 are bigger: a narrative plus runnable code.

## Set up in five minutes

You need Python 3.12 or newer. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"      # exact-pinned dependencies + pytest, ruff, ty
pytest -q                    # everything should pass, offline, with no API key
python lessons/01_hello_graph.py
```

If the tests pass and lesson 01 prints a greeting, you are ready. No API key is needed for anything below
unless you choose to add one.

Optional extras:

| Extra | Install | Why |
|---|---|---|
| `openai` | `pip install -e ".[openai]"` | real embeddings for the RAG apps (Anthropic has no embeddings API) |
| `postgres` | `pip install -e ".[postgres]"` | the Postgres checkpointer from the production lessons |

## How to work through a lesson

The same loop works for every lesson:

1. **Read the plan** in the [README](README.md#curriculum-at-a-glance): prerequisites, objectives, what you will see.
2. **Read the file's docstring**, then **run it**: `python lessons/05_agent_from_scratch.py`. Compare the output
   to "what you should see".
3. **Change something and break it.** Rename a state key, remove a reducer, lower the recursion limit. The
   error message is part of the lesson.
4. **Do the exercises** at the bottom of the file or in the lesson plan. Do not skip them; they are the exit criteria.
5. **Answer the check-your-understanding questions** before opening the collapsed answers.
6. **Read the test** that covers the lesson (see [`lessons/README.md`](lessons/README.md)). Tests show the
   expected behavior as assertions, which is often clearer than prose.

Prerequisites are cumulative, except Track D (lessons 16-19), which you can take any time after lesson 08.

## Which path should I take?

| Goal | Path |
|---|---|
| **From scratch, three weeks** | Week 1: lessons 01-05 (do every exercise, re-type 05 from memory). Week 2: 06-10 with a real model for 05, 06 and 08. Week 3: 11-12, the capstone, then `INTERVIEW.md`. |
| **Two more weeks to "advanced"** | Week 4: RAG, lessons 13-15 (run the evals, read an ablation table). Week 5: lessons 16-19, then 20, then the LangSmith track (21-23). |
| **Interview in a week** | Lessons 05, 08, 07, 17, 18, 19, then `INTERVIEW.md` and its whiteboard drills. |
| **I only need to ship an agent** | 05, 07, 08, 12, then the capstone and `PRODUCTION_CHECKLIST.md`. |
| **I already know LangGraph** | Skim 17-19 for durable execution, the Pregel runtime and the gotchas, then read `rag_prod/` and `sql_agent/` as reference designs. |

## Offline by default, real model on demand

Every lesson runs against scripted fake models and deterministic embeddings, so it is free, fast and repeatable.
That proves the *graph mechanics*; it does not measure model quality.

To use a real model, copy the example env file and add a key:

```powershell
copy .env.example .env      # then set ANTHROPIC_API_KEY=...
python lessons/05_agent_from_scratch.py
```

| You set | You get |
|---|---|
| nothing | offline fakes |
| `ANTHROPIC_API_KEY` | the default model, `anthropic:claude-sonnet-5-5` |
| `ANTHROPIC_API_KEY` and `LG_MODEL=anthropic:claude-opus-5-5` | a different Claude model |
| another provider's key **and** `LG_MODEL=<provider>:<id>` | that provider's model |

The model id lives in exactly one place (`lessons/common.py`). Expect different behavior with a real model:
nondeterministic output, real cost, and failure modes the fakes cannot produce. See
[`docs/real-models.md`](docs/real-models.md) before you start spending money, and turn on tracing
(`LANGSMITH_TRACING=true` plus a key) to watch the calls.

## Running the three apps

All three run offline out of the box.

### Lesson 14: production RAG (`rag_prod/`)

```powershell
python -m rag_prod.evals.run_evals                       # quality gate; exit code 1 on regression
uvicorn rag_prod.api:app --port 8000                     # the API
```

```powershell
curl.exe -s localhost:8000/chat -H "X-API-Key: key-alice" -H "content-type: application/json" `
  -d '{\"question\": \"How many PTO days do I get?\"}'
```

The demo users are `anon`, `alice`, `bob` and `cara`; their API keys are `key-<name>`, and each sees only the
documents their access labels allow (try the same question as `anon` and as `bob`). Other endpoints:
`/chat/stream` (server-sent events), `/healthz`, `/readyz`. Or run it in a container with
`docker compose up --build`.

### Lesson 15: hierarchical RAG (`westeros_rag/`)

```powershell
python -m westeros_rag.cli lint                           # validate the corpus
python -m westeros_rag.cli ingest                         # build the index
python -m westeros_rag.cli tree --role commoner           # what this role may see
python -m westeros_rag.cli ask "Can a pardon restore a traitor's lands?" --role noble
python -m westeros_rag.cli passages "trial by combat" --role commoner
python -m westeros_rag.evals.run_evals                    # quality gate
python -m westeros_rag.evals.ablate                       # what each retrieval feature buys you
uvicorn westeros_rag.api:app --port 8001                  # the API
```

Roles are `commoner`, `noble` and `small_council`. Try the same question under each role.

### Lesson 20: SQL agent (`sql_agent/`)

```powershell
python -m sql_agent.cli "How many customers do we have?"
python -m sql_agent.cli "Cancel order 6"                  # asks for approval; changes nothing
python -m sql_agent.cli "Cancel order 6" --approve        # applies the write
python -m sql_agent.evals.run_evals                       # quality + safety gate
```

Each run uses a fresh temporary demo database, so you cannot damage anything.

## Visualizing and debugging graphs

- **Print the diagram.** Every compiled graph can draw itself: `print(graph.get_graph().draw_mermaid())`, then
  paste it into <https://mermaid.live>.
- **LangGraph Studio.** `langgraph.json` registers three graphs (`rag`, `westeros`, `sql`). With the LangGraph
  CLI installed, `langgraph dev` opens them in Studio so you can step through nodes and edit state.
- **Inspect state.** With a checkpointer, `graph.get_state(config)` and `graph.get_state_history(config)` show
  every step (lesson 07). This is the fastest way to debug a graph that took the wrong branch.
- **Read the errors.** LangGraph's compile-time and runtime messages are specific; lesson 19 covers the silent
  failures that have no error at all.

## Running the checks

```powershell
pytest -q                          # offline unit + smoke tests (runs every lesson as a script; about a minute)
pytest -q tests/test_graphs.py     # just one file while you iterate
pytest -q -k interrupt             # just tests matching a name
pytest -q --cov                    # with coverage
ruff check .                       # lint
ty check                           # type check (Astral's ty, every rule an error)
pytest -q -m live                  # OPT-IN: calls a real model, needs a key, costs cents
```

With [Hatch](https://hatch.pypa.io) installed, `hatch run check` runs lint, types and tests with coverage, and is
what the GitHub CI workflow runs on Python 3.12, 3.13 and 3.14 (manual trigger only). Offline tests never touch the network, even if you have keys in `.env`.

## Making changes safely

- **Fork a lesson, don't edit the original.** Copy `lessons/05_agent_from_scratch.py` to a scratch file and
  experiment there; the originals are pinned by tests.
- **Add a test with every change.** Nodes are plain functions, so the cheapest test is calling one directly.
  Use the scripted fake model for whole-graph tests (see `tests/test_graphs.py`).
- **Type checks:** prefer fixing a `ty` diagnostic to ignoring it. LangGraph's `StateGraph(State)` and
  `Command[Literal[...]]` idioms are the known exceptions and carry a targeted `# ty: ignore[rule]`.
- **Upgrading dependencies:** versions are exact pins. Bump them in `pyproject.toml`, reinstall, then run all the
  checks. A failure in `tests/test_gotchas.py` means a lesson's claim about LangGraph is stale, so update the
  lesson text too. The full procedure is in [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **Changing the model:** edit `DEFAULT_MODEL` in `lessons/common.py`, or set `LG_MODEL` in `.env` for yourself.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError` when running a lesson | The venv is not active, or you skipped `pip install -e ".[dev]"`. |
| Lessons still print "(offline fake model)" after adding a key | `.env` is not in the repo root, the key line is empty, or you set only a non-Anthropic key without `LG_MODEL`. |
| Auth error from a real model | The key does not match the provider in `LG_MODEL` (the default is Anthropic). |
| `langchain_openai` import error in the RAG apps | You set `OPENAI_API_KEY` but did not install the extra: `pip install -e ".[openai]"`. |
| The RAG index looks stale or wrong after changing embeddings | Delete `.rag_index/` (or `.westeros_index/`); it rebuilds on the next run. Changing the embedding model forces this on purpose. |
| A test fails only after a dependency upgrade | Run `pytest -q tests/test_gotchas.py` first; it pins the library behavior the lessons describe. |
| `pytest` is slow | `tests/test_smoke.py` runs every lesson in a fresh interpreter. Run `pytest -q --ignore=tests/test_smoke.py` while iterating, and the full suite before you commit. |
| `docker compose up` cannot find a key | Keys come from your `.env` or shell, never from the image. Without one, the API runs offline. |

## Where to read next

- [README](README.md): the full lesson plans, concepts in one page, portfolio project ideas, and the official docs map.
- [`INTERVIEW.md`](INTERVIEW.md): questions, answers and whiteboard drills with lesson pointers.
- [`PRODUCTION_CHECKLIST.md`](PRODUCTION_CHECKLIST.md): what is done and what remains before shipping.
- [`docs/real-models.md`](docs/real-models.md): what changes when you leave the offline fakes.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): the development loop and the dependency upgrade procedure.
