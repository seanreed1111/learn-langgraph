# Production checklist for an agentic RAG service

Status key: [x] done in `rag_prod/`, [ ] you still need it.

## Data & indexing
- [x] Idempotent, incremental ingestion; stale chunk removal
- [x] Embedding-model fingerprint -> forced rebuild on change
- [x] Metadata on every chunk (source, title, access, updated)
- [ ] Scheduled/event-driven re-sync (webhook, cron) + alert on ingest failure
- [ ] Document parsing for PDFs/HTML/tables (Unstructured, Docling, etc.) with quality checks
- [ ] Freshness/versioning policy; deletion requests (GDPR) propagate to the index
- [ ] Real vector DB (pgvector/Qdrant/OpenSearch) with backups

## Retrieval
- [x] Hybrid (vector + BM25) with RRF
- [ ] Reranker (cross-encoder or LLM)
- [ ] Metadata filters beyond ACL (product, version, region, date)
- [ ] Tune chunk size / overlap / top-k against evals

## Security & privacy
- [x] ACL enforced inside retrieval, fail-closed, secure default label
- [x] Identity from auth context, never from the model or request body
- [x] Per-user thread namespacing
- [x] Retrieved text treated as data in prompts (delimiters + instructions)
- [ ] Real authN/Z (OIDC/JWT), service-to-service auth
- [ ] Prompt-injection red-team suite in CI; output filtering
- [ ] PII detection/redaction in logs, traces and checkpoints; retention policy
- [ ] Secrets in a secret manager; least-privilege DB credentials

## Agent behaviour
- [x] Every loop bounded (rewrites, generations, recursion_limit)
- [x] Abstain instead of guessing; generic abstain text
- [x] Structured output, validated citations, groundedness check
- [x] Per-turn state reset; follow-up question condensing
- [ ] Human-in-the-loop for high-risk answers (`interrupt`)
- [ ] Per-user token/cost budgets and rate limits
- [ ] Conversation length management (trim/summarize history)

## Reliability
- [x] Retries on transient errors only; model timeouts
- [x] Clean 502s; no internals leaked to clients
- [ ] Circuit breaker / fallback model on provider outage
- [ ] Async endpoints + connection pooling for concurrency
- [ ] Postgres checkpointer, migrations, load test
- [ ] Graceful shutdown; readiness depends on dependencies (DB, index)

## Quality & operations
- [x] Offline unit tests with fakes; eval dataset with thresholds
- [x] Per-node timing logs; path trace; run metadata for LangSmith
- [ ] Evals in CI on every PR; LLM-as-judge for faithfulness/helpfulness
- [ ] Online feedback (thumbs) -> new eval cases; sampling + human review of traces
- [ ] Dashboards/alerts: latency p95, error rate, abstain rate, cost per request, groundedness failures
- [ ] Prompt/model versioning with canary or shadow rollout and rollback
- [ ] Runbook: "answers got worse" / "provider down" / "index corrupted"
