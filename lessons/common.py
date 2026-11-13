"""Shared helpers for every lesson.

Key idea: lessons run OFFLINE by default using a scripted fake chat model, so you can
learn the *graph mechanics* without spending money or needing an API key.
Put a key in `.env` (see .env.example) and the same code uses a real LLM.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

load_dotenv()

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"


def have_real_llm() -> bool:
    return bool(
        os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


class ScriptedChatModel(GenericFakeChatModel):
    """A fake chat model that replays a fixed list of AIMessages.

    `bind_tools` is a no-op so it can stand in for a tool-calling model.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self


def tool_call_msg(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """Build an AIMessage that *asks* for a tool call (what a real LLM would emit)."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def get_model(script: list[AIMessage] | None = None):
    """Return a real chat model if a key is configured, else a scripted fake.

    `script` is only used offline: the sequence of AI replies the fake will give.
    """
    if have_real_llm():
        from langchain.chat_models import init_chat_model

        return init_chat_model(os.getenv("LG_MODEL", DEFAULT_MODEL), temperature=0)
    script = script or [AIMessage(content="(offline fake model) Hello from the fake LLM!")]
    return ScriptedChatModel(messages=iter(script))


def banner(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")
