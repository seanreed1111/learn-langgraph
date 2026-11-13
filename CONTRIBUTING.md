# Contributing / development loop

## One-time setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

The project uses **Hatch + Hatchling** for packaging. If you have Hatch installed, the same checks are
available as scripts (they build an isolated env from the `dev` extra):

| Task | Plain | Hatch |
|---|---|---|
| tests (offline) | `pytest -q` | `hatch run test` |
| tests + coverage | `pytest -q --cov` | `hatch run cov` |
| lint | `ruff check .` | `hatch run lint` |
| auto-fix lint | `ruff check --fix .` | `hatch run fmt` |
| types | `ty check` | `hatch run typecheck` |
| everything CI runs | | `hatch run check` |
| real-model smoke tests (opt-in) | `pytest -q -m live` | `hatch run live` |

CI (`.github/workflows/ci.yml`) runs lint, types and tests with coverage on Python 3.12, 3.13 and 3.14. It is manual only (`workflow_dispatch`): start it from the Actions tab. `hatch run check` runs the same checks locally.

## Type checking: `ty`, strictest settings, no mypy

Types are checked with Astral's [`ty`](https://docs.astral.sh/ty/). `[tool.ty.rules]` sets `all = "error"`, and the
strict analysis switches are on (`respect-type-ignore-comments = false`, `strict-equality-semantics`,
`strict-generic-narrowing`, `return-type-inference = "always"`). Do not add mypy.

LangGraph's `StateGraph(State)` and `Command[Literal[...]]` idioms are not understood by `ty` yet, so those
lines carry `# ty: ignore[invalid-argument-type]` or `[invalid-explicit-return]`. Prefer fixing a diagnostic
to ignoring it. An ignore that is no longer needed is itself an error, so they clean up as `ty` improves.

## Dependencies are exact pins

`pyproject.toml` pins every direct dependency with `==`. `tests/test_repo_integrity.py` fails if the installed
packages drift from the pins. To upgrade:

1. Bump the pins in `pyproject.toml` (check PyPI for current versions).
2. `pip install -e ".[dev]"`, then run `pytest -q`, `ruff check .`, `ty check`.
3. **A failure in `tests/test_gotchas.py` means a lesson's claim about LangGraph behavior is stale.** Update the
   lesson text, not just the test.
4. Run `pytest -q -m live` with a real key if you changed `langchain-anthropic` or the model id.

## Model ids

The default model id lives in exactly one place: `DEFAULT_MODEL` in `lessons/common.py` (a test enforces this).
See [docs/real-models.md](docs/real-models.md).

## Commits

Describe the change; do not add AI attribution trailers.
