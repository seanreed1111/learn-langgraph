"""Offline helpers for the LangSmith lessons (21-23).

Two tools, both fully offline:

* `FakeLangSmith` - a stand-in for the LangSmith *server*. It is a `requests.Session` that records every
  HTTP call the REAL `langsmith.Client` makes and answers 200. So `@traceable`, LangGraph's tracer,
  `hide_inputs`, projects, tags and metadata all run through the real client code path; you just read the
  payloads here instead of in the web UI.
* `RunRecorder` - a LangChain callback tracer that collects the run tree in memory. Simpler than the fake
  server when you only want to look at structure (names, parents, tags, metadata, errors, latency).

Plus a tiny experiment harness (`run_experiment`) with the same shape as `langsmith.evaluate`, because the
real `evaluate()` always talks to the network.
"""

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, override

import requests
from langchain_core.tracers.base import BaseTracer
from langsmith import Client

# --------------------------------------------------------------------------------------------------
# Are we talking to real LangSmith?
# --------------------------------------------------------------------------------------------------


def langsmith_configured() -> bool:
    """True only when tracing is switched on AND a key exists (both are needed for anything to be sent)."""
    on = (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").lower() == "true"
    return on and bool(os.getenv("LANGSMITH_API_KEY"))


# --------------------------------------------------------------------------------------------------
# A fake LangSmith server
# --------------------------------------------------------------------------------------------------


@dataclass
class Call:
    method: str
    path: str
    body: dict[str, Any] | None


class FakeLangSmith(requests.Session):
    """Records the calls the real client makes. Never touches the network."""

    HOST = "https://fake-langsmith.invalid"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Call] = []

    @override
    def request(self, method: str | bytes, url: str | bytes, *args: Any, **kwargs: Any) -> requests.Response:
        raw = kwargs.get("data") if kwargs.get("data") is not None else kwargs.get("json")
        body: dict[str, Any] | None = None
        if isinstance(raw, dict):
            body = raw
        elif isinstance(raw, (str, bytes)):
            try:
                loaded = json.loads(raw)
                body = loaded if isinstance(loaded, dict) else None
            except ValueError:
                body = None
        path = str(url).removeprefix(self.HOST)
        self.calls.append(Call(str(method), path, body))
        resp = requests.Response()
        resp.status_code = 200
        resp._content = b"{}"
        resp.url = str(url)
        return resp

    def client(self, **kwargs: Any) -> Client:
        """A real `langsmith.Client` wired to this fake server. Extra kwargs (hide_inputs=...) pass through."""
        return Client(api_url=self.HOST, api_key="offline-demo-key", session=self, auto_batch_tracing=False,
                      info={}, **kwargs)

    # --- reading what was "sent" -------------------------------------------------------------
    def runs(self) -> list[dict[str, Any]]:
        """One merged dict per run: the POST (start) with the PATCH (end) fields layered on top."""
        merged: dict[str, dict[str, Any]] = {}
        for call in self.calls:
            if call.body is None or not call.path.startswith("/runs"):
                continue
            run_id = str(call.body.get("id") or call.path.rsplit("/", 1)[-1])
            merged.setdefault(run_id, {}).update({k: v for k, v in call.body.items() if v is not None})
        return list(merged.values())

    def tree(self) -> str:
        return format_tree(self.runs(), name="name", parent="parent_run_id", id_="id", extra=_extra_summary)


def _extra_summary(run: Mapping[str, Any]) -> str:
    bits = [str(run.get("run_type", ""))]
    if run.get("error"):
        bits.append("ERROR")
    return " ".join(b for b in bits if b)


# --------------------------------------------------------------------------------------------------
# An in-memory run collector
# --------------------------------------------------------------------------------------------------


class RunRecorder(BaseTracer):
    """Collects every run (flat list; use `parent_run_id` to rebuild the tree)."""

    def __init__(self) -> None:
        super().__init__()
        self.runs: list[Any] = []

    @override
    def _persist_run(self, run: Any) -> None:
        return None

    @override
    def _on_run_create(self, run: Any) -> None:
        self.runs.append(run)

    # --- queries -----------------------------------------------------------------------------
    @property
    def roots(self) -> list[Any]:
        return [r for r in self.runs if r.parent_run_id is None]

    def by_type(self, run_type: str) -> list[Any]:
        return [r for r in self.runs if r.run_type == run_type]

    def node_names(self) -> list[str]:
        """Graph nodes in the order they started (a node run is named after its `langgraph_node`)."""
        return [r.name for r in self.runs if r.metadata.get("langgraph_node") == r.name]

    def tree(self) -> str:
        return format_tree(self.runs, name="name", parent="parent_run_id", id_="id", extra=_run_summary)


def _run_summary(run: Any) -> str:
    bits = [str(run.run_type)]
    if getattr(run, "error", None):
        bits.append("ERROR")
    ms = latency_ms(run)
    if ms is not None:
        bits.append(f"{ms:.0f}ms")
    return " ".join(bits)


def latency_ms(run: Any) -> float | None:
    if getattr(run, "end_time", None) is None or getattr(run, "start_time", None) is None:
        return None
    return float((run.end_time - run.start_time).total_seconds() * 1000)


def format_tree(runs: Sequence[Any], *, name: str, parent: str, id_: str,
                extra: Callable[[Any], str]) -> str:
    """Indented text tree for either dict runs (fake server) or Run objects (recorder)."""

    def get(run: Any, key: str) -> Any:
        return run.get(key) if isinstance(run, Mapping) else getattr(run, key, None)

    ids = {str(get(r, id_)) for r in runs}
    children: dict[str | None, list[Any]] = {}
    for run in runs:
        p = get(run, parent)
        key = str(p) if p is not None and str(p) in ids else None
        children.setdefault(key, []).append(run)

    lines: list[str] = []

    def walk(run: Any, depth: int) -> None:
        lines.append(f"{'  ' * depth}{get(run, name)}  [{extra(run)}]")
        for child in children.get(str(get(run, id_)), []):
            walk(child, depth + 1)

    for root in children.get(None, []):
        walk(root, 0)
    return "\n".join(lines)


def _usage_of(message: Any) -> Mapping[str, Any]:
    """`usage_metadata` from a message object, or from its serialized form (runs store outputs as dicts)."""
    if isinstance(message, Mapping):
        return message.get("kwargs", {}).get("usage_metadata") or message.get("usage_metadata") or {}
    return getattr(message, "usage_metadata", None) or {}


def token_totals(recorder: RunRecorder) -> dict[str, int]:
    """Sum `usage_metadata` over LLM runs (providers fill it in; the fakes only when scripted to)."""
    totals = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
    for run in recorder.by_type("llm"):
        totals["llm_calls"] += 1
        for gens in (run.outputs or {}).get("generations", []):
            for gen in gens:
                usage = _usage_of(gen.get("message"))
                totals["input_tokens"] += int(usage.get("input_tokens", 0))
                totals["output_tokens"] += int(usage.get("output_tokens", 0))
    return totals


# --------------------------------------------------------------------------------------------------
# A miniature `evaluate()`
# --------------------------------------------------------------------------------------------------

Evaluator = Callable[..., Mapping[str, Any]]


@dataclass
class ExperimentRow:
    inputs: dict[str, Any]
    reference: dict[str, Any]
    outputs: dict[str, Any]
    scores: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ExperimentResult:
    rows: list[ExperimentRow]

    def mean(self, key: str) -> float:
        values = [r.scores[key] for r in self.rows if key in r.scores]
        return statistics.fmean(values) if values else 0.0

    def means(self) -> dict[str, float]:
        keys = sorted({k for r in self.rows for k in r.scores})
        return {k: round(self.mean(k), 3) for k in keys}

    def failures(self, key: str) -> list[ExperimentRow]:
        return [r for r in self.rows if r.scores.get(key, 0.0) < 1.0]


def run_experiment(target: Callable[[dict[str, Any]], dict[str, Any]], examples: Sequence[Mapping[str, Any]],
                   evaluators: Sequence[Evaluator]) -> ExperimentResult:
    """Same contract as `langsmith.evaluate(target, data=..., evaluators=...)`, minus the network.

    * `examples`: [{"inputs": {...}, "outputs": {...reference...}}, ...]
    * `target(inputs) -> outputs`
    * each evaluator takes any of `inputs`, `outputs`, `reference_outputs` (by name) and returns
      {"key": str, "score": number|bool}. A bool score becomes 1.0 / 0.0.
    A target that raises is recorded as an error row with every score 0, never silently dropped.
    """
    import inspect

    rows: list[ExperimentRow] = []
    for ex in examples:
        inputs, reference = dict(ex["inputs"]), dict(ex.get("outputs") or {})
        row = ExperimentRow(inputs=inputs, reference=reference, outputs={})
        try:
            row.outputs = target(inputs)
        except Exception as exc:
            row.error = f"{type(exc).__name__}: {exc}"
        for ev in evaluators:
            if row.error is not None:
                row.scores[getattr(ev, "__name__", "evaluator")] = 0.0
                continue
            available = {"inputs": inputs, "outputs": row.outputs, "reference_outputs": reference}
            wanted = inspect.signature(ev).parameters
            result = ev(**{k: v for k, v in available.items() if k in wanted})
            row.scores[str(result["key"])] = float(result["score"])
        rows.append(row)
    return ExperimentResult(rows)
