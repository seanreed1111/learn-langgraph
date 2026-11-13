# Using a real model

Everything in this repo runs offline against scripted fakes. That is deliberate (deterministic, free, fast CI),
but fakes only prove the *graph mechanics*. This page covers what changes the moment a real model is behind it.

## Turning it on

```powershell
copy .env.example .env      # then set ANTHROPIC_API_KEY=...
python lessons/05_agent_from_scratch.py
```

| You set | You get |
|---|---|
| nothing | offline fakes |
| `ANTHROPIC_API_KEY` | `anthropic:claude-sonnet-5-5` (the default, defined once in `lessons/common.py`) |
| `ANTHROPIC_API_KEY` + `LG_MODEL=anthropic:claude-opus-5-5` | any other Claude model |
| `OPENAI_API_KEY` or `GOOGLE_API_KEY` **and** `LG_MODEL=<provider>:<id>` | that provider's model |
| `OPENAI_API_KEY` or `GOOGLE_API_KEY` **without** `LG_MODEL` | still offline: the default is an Anthropic model, so it would fail |

Current Claude model ids (as of 2026-11-12, from the
[models overview](https://platform.claude.com/docs/en/models/overview)): `claude-fable-5-2` (top tier),
`claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`. Check the page again before pinning a new id; ids
are dateless, and each model has a published "not sooner than" retirement date.

Embeddings are separate: Anthropic has no embeddings endpoint. The RAG apps use `OpenAIEmbeddings`
(`pip install -e ".[openai]"`, default `text-embedding-3-small`, override with `RAG_EMBEDDING_MODEL`) when
`OPENAI_API_KEY` is set, otherwise offline hashing embeddings. Changing the embedding model changes the index
fingerprint and forces a rebuild, on purpose.

## What changes

1. **Nondeterminism.** The same input can take a different path. Never assert on exact wording; assert on
   *structure*: which tool was called, which node ran, that a refund exists or does not.
2. **Sampling parameters are not set by default.** Models differ in which sampling parameters they accept, so
   the repo leaves them at the provider default. Set `LG_TEMPERATURE=0` to opt in, and check your model's docs
   if the provider rejects it.
3. **Cost and latency are now real.** Every loop iteration is a billed call. Lesson 12 (retries, caching) and
   the loop budgets in lessons 03 and 13 stop being exercises. Watch the trace, not just the answer.
4. **Evals become measurements.** Offline, the eval gates prove the rails (ACL, citations, abstention). With a
   real model they measure the model, so thresholds need tuning and a failure may be noise: run twice
   before concluding anything, and compare trends, not single runs.
5. **Failure modes appear that fakes cannot produce:** malformed tool arguments, refusals, a tool called when
   it should not be, long outputs, provider timeouts and rate limits. `rag_prod` already sets a timeout and
   bounded retries; lesson 12 shows per-node `RetryPolicy`.
6. **Tracing pays for itself.** Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to see every prompt,
   tool call and token count. Lessons 21-23 cover tracing, evaluation and the production loop, all runnable
   offline.

## Proving the real path still works

The opt-in live tests exercise the pieces fakes cannot: `init_chat_model`, tool calling and structured output.
They cost a few cents, so they never run by default or in CI:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
pytest -q -m live
```

Run them after bumping a dependency or a model id. They are intentionally loose: they check that the plumbing
works, not that the model is clever.
