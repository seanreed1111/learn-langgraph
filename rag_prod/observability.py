"""Structured logging + per-node timing. Production agents need to answer:
'which node was slow / failing for which conversation?' Pair this with LangSmith tracing
(LANGSMITH_TRACING=true) for full prompt/response traces."""

from __future__ import annotations

import functools
import json
import logging
import time
from typing import Any, Callable

logger = logging.getLogger("rag")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False


def timed(node_name: str) -> Callable:
    """Decorator for graph nodes: logs duration, thread id and success/failure."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)  # keeps the signature so LangGraph can still inject `runtime`/`config`
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            thread_id = None
            try:
                from langgraph.config import get_config

                thread_id = get_config().get("configurable", {}).get("thread_id")
            except Exception:  # noqa: BLE001 - observability must never break the request
                pass
            try:
                result = fn(*args, **kwargs)
                outcome = "ok"
                return result
            except Exception:
                outcome = "error"
                raise
            finally:
                logger.info(
                    "node_done",
                    extra={"fields": {
                        "node": node_name, "outcome": outcome, "thread_id": thread_id,
                        "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                    }},
                )

        return wrapper

    return deco
