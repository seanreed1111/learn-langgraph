"""Opt-in smoke tests against a REAL model. Everything else in this repo runs offline.

    set ANTHROPIC_API_KEY=...      (or put it in .env)
    hatch run live                 # == pytest -m live

They cost a few cents and are non-deterministic, so assertions are deliberately loose. They exist to
prove the real-model path (init_chat_model, tool calling, structured output) still works after you
bump a dependency or a model id, which no fake model can tell you.
"""

import common
import pytest
from dotenv import load_dotenv

pytestmark = pytest.mark.live

load_dotenv()
if not common.have_real_llm():
    pytest.skip("no usable API key / LG_MODEL: live tests are opt-in", allow_module_level=True)


def test_plain_completion():
    msg = common.make_real_chat_model().invoke("Reply with exactly one word: pong")
    assert "pong" in str(msg.content).lower()


def test_tool_calling():
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"sunny in {city}"

    model = common.make_real_chat_model().bind_tools([get_weather])
    msg = model.invoke("What is the weather in Paris? Use the tool.")
    assert [c["name"] for c in msg.tool_calls] == ["get_weather"]


def test_structured_output_sql_draft():
    from sql_agent.llm import SQLDraft

    draft = common.make_real_chat_model().with_structured_output(SQLDraft).invoke(
        "Write one SQLite query that counts rows in a table named customers."
    )
    assert "select" in draft.sql.lower()

