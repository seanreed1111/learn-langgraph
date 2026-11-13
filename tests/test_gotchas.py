"""Lesson 19: each test pins an OBSERVED langgraph behaviour the lesson teaches.

If one fails after an upgrade, the lesson text is stale (or the gotcha was fixed): read the
changelog, then update lessons/19_gotchas.md. That is the point of testing library behaviour.
"""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lessons"))

import pytest  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

g = importlib.import_module("19_gotchas")


def test_in_place_mutation_diverges_from_checkpoint():
    r = g.gotcha_mutation()
    assert r["bad"]["returned"] == ["x"]            # looks fine to the caller...
    assert r["bad"]["caller_input_after"] == ["x"]  # ...and mutated the caller's own input...
    assert r["bad"]["saved_checkpoint"] == []       # ...but the checkpoint never saw it
    assert r["good"]["saved_checkpoint"] == ["x"] and r["good"]["caller_input_after"] == []


def test_resending_history_duplicates_but_sending_new_turn_does_not():
    r = g.gotcha_resend_history()
    assert r["bad"].count("hi") == 2
    assert r["good"] == ["hi", "reply1", "more", "reply3"]


def test_plain_list_loses_history():
    assert g.gotcha_overwrite_messages() == ["hi"]


def test_typos_are_silent_and_non_dict_is_loud():
    r = g.gotcha_typo_keys()
    assert r["typo_result"]["out"] == ""        # typo'd key dropped without error
    assert r["extra_input_key"] == {"q": "a", "out": ""}
    assert r["non_dict"] == "InvalidUpdateError"


def test_update_key_guard_catches_the_typo():
    class S(TypedDict):
        out: str

    assert g.assert_update_keys({"out": "x"}, S) == {"out": "x"}
    with pytest.raises(AssertionError, match="ouput"):
        g.assert_update_keys({"ouput": "x"}, S)


def test_command_plus_static_edge_runs_both():
    assert g.gotcha_command_and_edge() == ["a", "b", "c"]


def test_router_typo_ends_quietly_with_only_a_log_warning():
    r = g.gotcha_router_typo()
    assert r == {"out": "", "warned": True, "with_path_list": "KeyError"}


def test_separate_fan_in_edges_run_join_twice():
    assert g.gotcha_fan_in() == {"separate_edges": 2, "list_edge": 1}


def test_default_recursion_limit_is_huge_so_set_your_own():
    r = g.gotcha_recursion_default()
    assert r["default"] > 1000 and r["explicit"].startswith("GraphRecursionError")


def test_try_except_swallows_interrupt():
    assert g.gotcha_swallowed_interrupt() == {"paused": False, "out": "auto-approved"}


def test_fresh_input_on_paused_thread_replaces_state_but_keeps_pause():
    r = g.gotcha_abandoned_interrupt()
    assert r["state_q"] == "refund $5000" and r["still_paused"]
    assert r["pending_question"] == "approve?"  # the human is approving a DIFFERENT request now


def test_unserializable_state_fails_only_with_a_checkpointer():
    r = g.gotcha_unserializable_state()
    assert "msgpack" in r["error"] and r["works_without_checkpointer"]


def test_shared_thread_id_leaks_between_users():
    assert g.gotcha_shared_thread() == {"shared_sees_alice": True, "namespaced_sees_alice": False}


def test_runtime_injected_identity_cannot_be_overridden_by_the_model():
    r = g.gotcha_tool_identity()
    assert r["schema_bad"] == ["user_id"] and r["schema_good"] == []
    assert r["bad"] == "orders of victim" and r["good"] == "orders of alice"


def test_compile_is_cheap_so_dont_cache_graphs_out_of_fear():
    assert g.gotcha_compile_cost_ms() < 50
