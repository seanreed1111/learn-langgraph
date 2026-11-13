FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY . .
RUN pip install -e ".[openai]"

# never run as root
RUN useradd -m app && chown -R app /app
USER app

# Secrets (ANTHROPIC_API_KEY, OPENAI_API_KEY, LANGSMITH_API_KEY) come from the environment / secret manager, NOT the image.
# The demo index/checkpoints live in /app/.rag_index: mount a volume, or (better) use Postgres + pgvector
# so several replicas share the same index and conversation state.
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1
CMD ["uvicorn", "rag_prod.api:app", "--host", "0.0.0.0", "--port", "8000"]
