"""Shared test setup: import paths, and every test starts OFFLINE.

Keys in your real environment or .env must never make a unit test hit the network, so each test gets
blank provider keys and no LG_MODEL override. Opt-in live tests (-m live) clear this themselves.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "lessons", ROOT / "capstone"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PROVIDER_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")


@pytest.fixture(autouse=True)
def _offline_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return
    for key in PROVIDER_KEYS:
        monkeypatch.setenv(key, "")
    monkeypatch.delenv("LG_MODEL", raising=False)
    monkeypatch.delenv("LG_TEMPERATURE", raising=False)
