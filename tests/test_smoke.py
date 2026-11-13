"""Smoke tests: run every lesson and CLI exactly as a learner would (`python lessons/NN_x.py`).

Unit tests import functions; these run the real `__main__` blocks in a fresh interpreter, offline, so a
lesson that imports fine but crashes when run (or an API a dependency bump removed) is caught here.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LESSON_FILES = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "lessons").glob("[0-9][0-9]_*.py"))
SCRIPTS = [*LESSON_FILES, "capstone/support_agent.py"]
# Python code that must exist as lessons (guards against a lesson file being renamed or dropped).
EXPECTED_LESSONS = {f"{n:02d}" for n in [*range(1, 14), 16, 17, 18, 19]}


def _run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in {"LG_MODEL", "LG_TEMPERATURE", "LANGSMITH_API_KEY"}}
    env.update(ANTHROPIC_API_KEY="", OPENAI_API_KEY="", GOOGLE_API_KEY="", LANGSMITH_TRACING="false",
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=timeout, check=False, encoding="utf-8")


def test_every_expected_lesson_file_exists():
    assert {Path(f).name[:2] for f in LESSON_FILES} >= EXPECTED_LESSONS


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs_offline(script: str):
    done = _run([script])
    assert done.returncode == 0, f"{script} failed:\n{done.stderr[-1500:]}"
    assert done.stdout.strip(), f"{script} printed nothing"


def test_sql_agent_cli_read_question():
    done = _run(["-m", "sql_agent.cli", "How many customers do we have?"])
    assert done.returncode == 0, done.stderr[-1500:]
    assert "STATUS" in done.stdout


def test_sql_agent_cli_write_needs_approval():
    done = _run(["-m", "sql_agent.cli", "Cancel order 6"])
    assert done.returncode == 0, done.stderr[-1500:]
    assert "APPROVAL NEEDED" in done.stdout and "Nothing was changed" in done.stdout


def test_westeros_cli_lint_passes():
    done = _run(["-m", "westeros_rag.cli", "lint"])
    assert done.returncode == 0, done.stderr[-1500:]


@pytest.mark.parametrize("module", ["rag_prod.evals.run_evals", "sql_agent.evals.run_evals", "westeros_rag.evals.run_evals"])
def test_eval_gates_pass_offline(module: str):
    """The eval runners exit non-zero when a quality threshold regresses: that is the CI gate."""
    done = _run(["-m", module], timeout=300)
    assert done.returncode == 0, f"{module} gate failed:\n{done.stdout[-1500:]}\n{done.stderr[-800:]}"
