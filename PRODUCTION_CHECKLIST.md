# Production checklist for an agentic RAG service

Status key: [x] done in `rag_prod/` (RAG sections) or `sql_prod/` (the text-to-SQL section at the end), [ ] you still need it.

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

## Text-to-SQL agent (lesson 24, `sql_prod/`)
- [x] Parser-based guard (allow-list over the tree); the re-serialised statement is the one that runs
- [x] Per-role table AND column policy enforced by the database (authorizer here; GRANTs/RLS in Postgres)
- [x] Model sees only the schema its role may read; unknown role or no identity = no access
- [x] Statement timeout, value-size limit, row cap applied as rows leave the database
- [x] Error classification: permission/timeout/size/constraint are final; only real SQL errors are repaired
- [x] Writes: rolled-back preview, row limit, foreign keys on, human approval, drift check in one transaction
- [x] Approval needs a different authorised person, inside a window, for the exact statement; runs under the requester's privileges
- [x] Append-only audit log (metadata, never rows); every graph path ends in one audit node
- [x] Per-connection-per-operation database access; per-user rate limit; durable approvals; ownership checks (404)
- [x] Evals: execution accuracy, safety (DB hash + audit), access with leak count = 0, write gating incl. drift
- [ ] Row-level security / tenant isolation (the policy is per role, not per row)
- [ ] PII masking of allowed columns; DLP on answers
- [ ] `EXPLAIN` cost gate; per-user query and token budgets
- [ ] Postgres: role per policy, `default_transaction_read_only`, pool, `PostgresSaver`, audit in a separate store
- [ ] Approval worklist, notifications, expiry sweeper, escalation
- [ ] OIDC/JWT instead of demo API keys; secrets manager
- [ ] Red-team suite against the parser guard (fuzzing), run in CI
