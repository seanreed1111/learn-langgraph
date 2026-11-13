"""The model configuration lives in ONE place (lessons/common.py). These tests pin its contract."""

import re
from pathlib import Path
from typing import Any

import common
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = re.compile(r"claude-(?:fable|opus|sonnet|haiku|mythos)-\d")


def test_default_is_a_current_claude_5_model():
    assert common.DEFAULT_MODEL == "anthropic:claude-sonnet-5-5"


def test_offline_when_no_keys():
    assert common.have_real_llm() is False


def test_anthropic_key_alone_is_enough(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert common.have_real_llm() is True
    assert common.model_name() == common.DEFAULT_MODEL


@pytest.mark.parametrize("key", ["OPENAI_API_KEY", "GOOGLE_API_KEY"])
def test_other_provider_key_without_lg_model_stays_offline(monkeypatch: pytest.MonkeyPatch, key: str):
    """Regression: an OpenAI/Google key used to flip us 'online' but still ask for the Anthropic default."""
    monkeypatch.setenv(key, "k")
    assert common.have_real_llm() is False


def test_other_provider_key_with_lg_model_is_online(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("LG_MODEL", "openai:some-model")
    assert common.have_real_llm() is True
    assert common.model_name() == "openai:some-model"


def test_make_real_chat_model_passes_name_and_leaves_sampling_to_provider(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    def fake_init(name: str, **kwargs: Any) -> str:
        seen.update(name=name, kwargs=kwargs)
        return "model"

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init)
    monkeypatch.setenv("LG_MODEL", "anthropic:claude-haiku-5-5")
    assert common.make_real_chat_model(timeout=5) == "model"
    assert seen == {"name": "anthropic:claude-haiku-5-5", "kwargs": {"timeout": 5}}  # no temperature


def test_temperature_is_opt_in(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}
    monkeypatch.setattr("langchain.chat_models.init_chat_model", lambda name, **kw: seen.update(kw))
    monkeypatch.setenv("LG_TEMPERATURE", "0.2")
    common.make_real_chat_model()
    assert seen == {"temperature": 0.2}


def test_model_ids_are_defined_only_in_common():
    """Single source of truth: no other module may hardcode a Claude model id."""
    offenders = []
    for folder in ("lessons", "rag_prod", "westeros_rag", "sql_agent", "capstone"):
        for path in (ROOT / folder).rglob("*.py"):
            if path.name == "common.py" and folder == "lessons":
                continue
            if MODEL_ID.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
