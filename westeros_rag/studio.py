"""Entrypoint for `langgraph dev` / LangSmith Studio (see langgraph.json). The platform manages persistence itself."""

from westeros_rag.pipeline import build_pipeline

graph = build_pipeline().graph.builder.compile()  # same graph, no custom checkpointer
