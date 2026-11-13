FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app
WORKDIR /app

# Layer 1 (cached until pyproject.toml changes): the exact pinned runtime deps, incl. the openai extra.
COPY pyproject.toml ./
RUN python -c "import tomllib, pathlib; p = tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']; pathlib.Path('/tmp/req.txt').write_text(chr(10).join(p['dependencies'] + p['optional-dependencies']['openai']))" \
 && pip install -r /tmp/req.txt

# Layer 2: application code and the demo corpus only (see .dockerignore).
COPY rag_prod ./rag_prod
COPY sql_agent ./sql_agent
COPY sql_prod ./sql_prod
COPY lessons ./lessons
COPY data ./data

# never run as root
RUN useradd -m app && mkdir -p /app/.rag_index /app/.sql_prod && chown -R app /app
USER app

# Secrets (ANTHROPIC_API_KEY, OPENAI_API_KEY, LANGSMITH_API_KEY) come from the environment / secret manager, NOT the image.
# The demo index/checkpoints live in /app/.rag_index: mount a volume, or (better) use Postgres + pgvector
# so several replicas share the same index and conversation state.
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1
# The default service is the RAG API. The SQL analyst (lesson 24) uses the same image:
#   docker run -p 8001:8001 <image> uvicorn sql_prod.api:app --host 0.0.0.0 --port 8001
CMD ["uvicorn", "rag_prod.api:app", "--host", "0.0.0.0", "--port", "8000"]
