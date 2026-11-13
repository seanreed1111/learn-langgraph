# LESSON 15 - Hierarchical, link-aware agentic RAG over a linked legal corpus

Lessons 13-14 retrieved from flat files. Real knowledge bases are **trees of sections with cross-references**: legal codes,
API docs, runbooks, wikis, contracts, policies. This lesson builds a complete app for that shape of data, from
parsing to guardrails to evals, over a corpus you can read in an hour: the laws of the Seven Kingdoms.

```powershell
python -m westeros_rag.cli lint                 # structural quality gate for the corpus
python -m westeros_rag.cli ingest               # parse -> lint -> enrich -> index (incremental, cached)
python -m westeros_rag.cli tree --role commoner # the outline THIS role may see
python -m westeros_rag.cli passages "Can a pardon bring back a traitor's lands?"   # retrieval only: debug what was fetched
python -m westeros_rag.cli ask "How much blood price is owed for killing a landed knight?" --role noble
python -m westeros_rag.evals.run_evals          # retrieval + end-to-end evals, CI gate (exit 1 below thresholds)
python -m westeros_rag.evals.ablate             # what is each technique actually worth?
uvicorn westeros_rag.api:app --port 8001        # the service
pytest -q tests/test_westeros.py                # 52 tests
```

All of it runs **offline** (rule-based test doubles in `westeros_rag/offline.py`); add a key to `.env` and the same code uses a real model.

---------------------------------------------------------------------------------------------------

## 1. The corpus (`data/westeros/`, 22 documents, 190 sections, 181 links)

An original, written-for-this-course legal compendium of the Seven Kingdoms: crown law and lordly justice, courts and procedure,
crimes and punishments, treason and attainder, trial by combat, trial by seven, the Night's Watch, guest right, royal mercy,
succession, bastards, marriage, the Faith and sanctuary, war and siege, the Kingsguard, commerce and debt, regional customs
(North, Riverlands, Vale, Reach, Westerlands, Stormlands, Dorne, Iron Islands, Crownlands), a glossary, and a hub page.
Two documents are **restricted** on purpose, so access control has something to protect.

> *Provenance:* the wiki page you linked refused automated fetching (HTTP 403), so I did not copy from it. The corpus is original text
> covering the same topics, extended into a fuller code (articles, exceptions, regional variants, restricted decrees). It is fiction.

**Authoring conventions** (these are the "contract" between writers and the parser; the linter enforces them):

| Convention | Why |
|---|---|
| Front matter: `doc_id`, `title`, `domain`, `jurisdiction`, `access` | metadata for filtering, citations, ACLs. `doc_id` must equal the file name |
| Exactly one `# H1`; `##` articles; `###` clauses; no skipped levels | the heading tree **is** the structure the retriever exploits |
| Links are relative: `[text](other.md#anchor)` or `(#anchor)` | the author's own relevance edges. Anchors = GitHub-style slugs |
| `<!-- jurisdiction: north -->` right under a heading | section-level metadata inside a multi-topic document |
| Restricted docs are **not linked from public docs** | a public link to a secret page leaks that it exists |

---------------------------------------------------------------------------------------------------

## 2. Your pipeline, mapped to code (and judged by evidence)

The outline you pasted (structure extraction, hierarchical chunking, theme indexing, multi-level retrieval, summaries) is a good
**menu of techniques**. A menu is not a recommendation: every item costs complexity, so each one is implemented *switchable* and measured.

| Pipeline step | Where | Verdict on this corpus (from `ablate.py`) |
|---|---|---|
| Convert to markdown, keep heading levels | `parse.py` | essential: it is what makes everything else possible |
| Build a document tree (sections, themes) | `parse.py` `Corpus`, `Section` | essential |
| Split by headers first, then recursively | `index.py` `build_entries` | essential: chunks inherit a **heading-path header** |
| Parent-child chunks | chunks (retrieval) vs **units** (context) | context +24% chars; gives the LLM whole, coherent sections |
| Tag chunks with themes; filter/boost | `enrich.py`, `retrieval.py` | **soft boost only**: neutral here; never a hard filter (a wrong tag would hide the answer) |
| Store embeddings + metadata | `index.py` | essential: powers ACLs, jurisdiction filters, citations |
| Multi-level retrieval (top section, then subsections) | `retrieval.py` stage A, `strategy="hier"` | **did not help here** (recall 0.971 -> 0.941). **Opt-in.** See section 6 |
| Summaries per section | `enrich.py` | useful as metadata and for orientation; weak as a retrieval signal here |
| Route via the document tree / follow structure | **link expansion** | rescues recall when first-pass misses (0.853 -> 0.941 in the `hier` rows); costs +50% context in default mode |
| Combine summaries for multi-topic queries | planner splits sub-questions; one search each | works; graded per sub-question (see 4.6) |

---------------------------------------------------------------------------------------------------

## 3. Architecture

```
 data/westeros/*.md
        |  parse.py       markdown -> tree of Sections + link graph          (STRUCTURE EXTRACTION)
        |  lint()         broken links/anchors, heading skips, orphans...    (QUALITY GATE - fails the ingest)
        |  enrich.py      per-unit summary + themes, cached by content hash  (ENRICHMENT, LLM, validated)
        |  index.py       "summary" entries + "chunk" entries + metadata     (INDEX, incremental, persistent)
        v
 retrieval.py   chunks (hybrid vector+BM25) -> units (RRF) -> parent expansion -> link expansion -> budget
        |         filters BEFORE the model sees anything: access level, jurisdiction
        v
 graph.py       guard_input -> condense -> analyze(plan) -> retrieve -> grade -> generate -> verify -> output_guard
        |
 api.py / cli.py
```

Vocabulary: **section** = any heading and its own text. **Unit** = the context granularity: an H2 *with its H3 subsections* (an H1 is
only its intro). **Chunk** = ~450 characters of one section's text. We **search small, show big**: chunks find the place, units are
what the LLM reads.

---------------------------------------------------------------------------------------------------

## 4. Stage by stage

### 4.1 Parsing (`parse.py`): structure is data
- A line-based parser builds `Section(section_id="doc#slug", level, heading_path, parent_id, children, text, links, meta)`.
  `text` is the section's **own** text only; hierarchy lives in the tree, not duplicated in strings.
- Handles: front matter, GitHub slugs (duplicate headings get `-1`), fenced code (a `# comment` in code is not a heading),
  section-metadata comments, relative links.
- **`strip_links()`**: `[trial by combat](trial_by_combat.md#x)` -> `trial by combat` for everything that is embedded, keyword-indexed
  or shown to a model. The links stay in `Section.links` as graph edges. *Not doing this was a real bug I hit:* URL fragments
  (`crimes_and_punishments`, `md`) became tokens, made unrelated sections look alike, and cost ~6 points of recall (0.912 -> 0.971 when fixed).
- `Corpus` offers `unit_id()`, `unit_text()`, `unit_links()`, `resolve()` (link -> section), `backlinks()`.

### 4.2 The lint gate (`lint()`): fail before you index
Checks: duplicate/mismatched `doc_id`, missing front matter, unknown access/jurisdiction, exactly-one-H1, heading skips, empty sections,
**broken links**, **broken anchors**, orphan public docs. `build_pipeline()` raises `CorpusError` instead of indexing a broken corpus.
Run it in CI on every docs change: it is the cheapest quality control in the whole system.

### 4.3 Enrichment (`enrich.py`): summaries and themes, with discipline
- One structured LLM call per unit: `SectionMeta(summary, themes)`. The document overview is a **hierarchical summary** built from
  the intro plus the children's summaries.
- **Constrained vocabulary:** themes must come from `THEMES`; anything else is dropped (free-form tags drift and break filters).
  Summaries are length-capped. LLM output is *validated*, never trusted.
- **Cached** by `sha(prompt version | model id | heading | text)`: the second ingest made **0** LLM calls (first run: 134). Bump
  `PROMPT_VERSION` when you change the prompt, or you will reuse stale results.

### 4.4 Indexing (`index.py`): two kinds of entry, rich metadata
- `kind="summary"`: one per unit (`heading path + summary + themes`). `kind="chunk"`: each section's text, **prefixed with its heading path**
  (`Trial by Combat > When It May Be Demanded > Capital Cases`), so a chunk like "The judge may not refuse" is self-describing.
- Metadata on every entry: `doc_id, unit_id, section_id, heading_path, access, jurisdiction, themes`.
- Incremental per document (signature = file hash + its enrichment), stale entries deleted before new ones are added, persisted, embedding
  fingerprint forces a rebuild if you change models (same properties as lesson 14).

### 4.5 Retrieval (`retrieval.py`): the heart
1. **Filters first**: `access in allowed` and (if the user names a region) `jurisdiction in {"all", region}`. Fail closed on an empty permission set.
2. **Stage B**: hybrid vector + BM25 over chunks -> ordered list of units.
3. **Stage A (opt-in)**: the same over summaries; fused with stage B by **RRF at unit level**, weight `summary_weight`.
4. **Theme boost**: small additive bonus per matching theme (soft).
5. **Parent expansion**: return the whole unit, not the 450-char chunk.
6. **Link expansion**: follow the unit's cross-references up to `link_depth`, ranked by relevance to the query, **visited-set (cycle-safe)**,
   access + jurisdiction checked (no leaking through links), capped by `max_linked`. Linked passages are labelled `kind="linked"` with `via=`.
7. **Context budget**: trim to a character budget; primary evidence first.

### 4.6 The graph (`graph.py`)
- **`analyze`** is one structured call returning a *plan*: `route` (law / chitchat / off_topic), `themes`, `jurisdiction`, `sub_questions`.
  The plan is **validated** (themes in vocabulary, jurisdiction in the region list).
- **One retrieval per sub-question**, merged. Each passage remembers *which sub-question found it*, and is **graded against that one**.
  (Grading a poaching passage against "penalty for poaching, **and who can grant a pardon**" wrongly rejects it. My first version did this.)
- **Grading sees the heading path**: an H2's own heading ("Poaching and Forest Law") is not in its text, but it is the most informative line.
- **Citations are deep links** (`royal_mercy.md#limits-of-the-pardon`) to sections that were actually provided; invented numbers are dropped.
- `verify` (groundedness vs the cited passages), then `output_guard` (leak check), then `finalize`. Any failure -> **abstain**.
- Per-turn fields are reset in `guard_input` (the lesson-14 multi-turn gotcha).

### 4.7 Guardrails (`guardrails.py`): defence in depth
| Layer | Control | Test |
|---|---|---|
| Input | length/empty; **prompt-injection patterns** (refused before any model call) | `test_injection_phrasings_are_refused` |
| Scope | planner `route=off_topic` -> polite refusal | `test_off_topic_and_chitchat_do_not_retrieve` |
| Retrieval | **access + jurisdiction filters**, also enforced on linked passages | `test_retriever_never_returns_unpermitted_sections_even_through_links` |
| Retrieved text | **sanitize**: escape `</context>`-style delimiters, drop lines that address the model | `test_sanitize_...`, `test_poisoned_document_cannot_hijack_the_answer` |
| Output | citations must be provided passages; **leak detector** (shingles of documents the caller cannot read) | `test_leak_detector_...` |
| Output | groundedness verifier; abstain on failure | graph tests |
| API | per-user thread namespacing; `/sections` returns an identical 404 for "hidden" and "missing" (no existence oracle) | `test_api_sections_and_tree_respect_access` |

Honest limit: regex injection detection removes the *cheap* attacks only. The structural controls (filters, citations, verifier,
leak check) limit the damage of the rest, and red-team cases in the evals measure both.

**Bugs the tests caught while building this lesson:** an injection regex that missed "disregard *your prior* instructions"; a disclaimer I
defined but never attached; and an API route using `/sections/doc#slug` (a URL `#fragment` never reaches a server, so the "ACL 404" assertion was passing for the
wrong reason until I made it compare against a successful call).

---------------------------------------------------------------------------------------------------

## 5. Evals: three layers, localize the failure

`python -m westeros_rag.evals.run_evals` (current offline numbers):

| Layer | What it measures | Result |
|---|---|---|
| Retrieval (no LLM, 19 cases) | section **recall**, **primary recall**, **MRR**, forbidden-section leaks; sliced by tag (`multi_hop`, `regional`, `acl`, `vocabulary`...) | recall 0.971, MRR 0.811, leaks 0 |
| End-to-end (16 cases) | status, cited sections, key facts, guardrail flags, no restricted text/citations | 15/16 pass (0.938), leaks 0, injection blocked 100% |
| Ablation | each technique switched off in turn | table below |

**Why layers?** Look at the one failing e2e case, `two-witnesses`. Retrieval evals show the right section *was* retrieved; the end-to-end
failure is a **generation** error (the answerer cited a distractor passage). Without the retrieval layer you would waste a day tuning
chunking for a problem that lives in the prompt/model. The gate (`THRESHOLDS`) allows 0.9 so one known failure does not block CI, but it stays visible.

```
configuration                   recall  primary    MRR  ctx chars
1 chunks only                    0.971    0.971  0.811       1540
2 + parent expansion             0.971    0.971  0.811       1913
3 + theme boost                  0.971    0.971  0.811       1913
4 + link expansion (DEFAULT)     0.971    0.971  0.811       2875
5 DEFAULT + summary stage        0.941    0.853  0.804       2287
6 summary stage, no links        0.853    0.853  0.804       1434
```

### 5.1 What this experiment taught (more valuable than the final numbers)
1. **The summary stage did not earn its keep here.** My first implementation *added* raw scores from two levels (different scales, double counting)
   and dropped recall to 0.765. Switching to **rank fusion (RRF)** raised it to 0.824, **weighting the coarser signal down** helped a little, and
   even the best weight (0.0, i.e. ignore summaries) only *matched* flat search. The diagnosis was visible in the data: the right section's
   summary didn't mention the "has confessed" exception, which lives in a subsection. **Summaries capture topic, not specifics.**
   So the default is `strategy="flat"` and the summary stage is opt-in.
2. **Link expansion is insurance, not a free win.** In the default config recall is already 0.971, so links add nothing but +50% context. In the
   `hier` rows, where first-pass search misses sections, links lift recall 0.853 -> 0.941. Keep it where answers span cross-referenced
   sections, cap it (`max_linked`), and let *your* ablation decide.
3. **Parent expansion trades context for completeness**: +24% characters, which recall@k cannot reward, so it needs an answer-quality eval (LLM judge) to justify.
4. **Preprocessing beat architecture.** Stripping link syntax was worth more than any retrieval-stage change.

### 5.2 Caveats you must carry to your own project
- Offline embeddings are lexical, summaries are extractive, and answers come from a rule-based double. **Absolute numbers will not transfer;
  conclusions about summaries might even flip** with a real model writing abstractive summaries and real embeddings. Re-run `ablate.py` on your data.
- 19 retrieval cases is small, and the sweep tuned on the same cases it reports. Use a **dev/held-out split** and grow the set from real questions
  and thumbs-down feedback.
- Deterministic `must_contain` checks are brittle with real models (paraphrase). Move to LLM-as-judge for faithfulness/completeness
  (LangSmith evaluators) once the structural checks are green.

---------------------------------------------------------------------------------------------------

## 6. Moving this to production
1. **Real models**: `ANTHROPIC_API_KEY` (+ `OPENAI_API_KEY` and `pip install langchain-openai` for embeddings). Delete `.westeros_index/`, run `ingest`, then `run_evals` and `ablate`. Use a **small model** for `analyze`/`grade`/`verify`/enrichment and a stronger one for `generate`.
2. **Vector store**: Postgres + pgvector (metadata in `jsonb`, native full-text search instead of the in-memory BM25) or Qdrant/OpenSearch. Only `VectorIndex` changes; keep the metadata schema and the filter-before-search semantics.
3. **Link graph**: for a few thousand sections, an adjacency table in Postgres is plenty. Reach for a graph DB only when you need multi-hop traversals with edge types.
4. **Ingestion as a pipeline**: trigger on docs changes (webhook/CI), run `lint` first, make enrichment idempotent and cached, alert on failures, and version the index (blue/green swap) so a bad ingest can be rolled back.
5. **Real parsers** for PDF/HTML/Word -> markdown (keep heading levels; verify with the same linter), plus a human spot-check on a sample.
6. **Reranker** (cross-encoder or LLM scoring) over the top ~20 units, measured with the same ablation.
7. **Operations**: everything in `PRODUCTION_CHECKLIST.md` still applies (rate limits, token budgets, PII policy, tracing, dashboards, canary prompts).
   `langgraph.json` registers this graph for `langgraph dev`/Studio via `westeros_rag/studio.py` (written, imported OK; not run in Studio here).

---------------------------------------------------------------------------------------------------

## 7. Exercises
1. Add a new document `sealed_pardons.md` with `access: small_council`. Extend the lint gate to flag any *public* document that links to a restricted one, and add retrieval and end-to-end ACL eval cases for it.
2. Turn on `strategy="hier"`, replace the offline summaries with *abstractive* ones from a real model, and re-run `ablate.py`. Did the verdict change?
3. Add a `reranker` stage after unit fusion and measure MRR change.
4. Add a **contradiction check**: when two retrieved sections disagree by region (e.g. Dorne vs default inheritance), make `generate` state both and cite both. Write an eval that requires it.
5. Replace the character budget with a **token budget** (use the model's tokenizer) and show how recall/latency move as it shrinks.
6. Add a *temporal* dimension: `updated:` front matter + a "law as of year X" filter. Which parts of parsing, indexing and retrieval change?
7. Red-team: write 10 injection attempts (including ones in other languages and encoded). Which pass `check_input`? What stops the rest?

## 8. Interview talking points
- "Why chunk by headings?" -> headings are the author's semantic boundaries; the heading path is free context; units give parents for context while chunks give precision.
- "How do you use links?" -> they are curated relevance edges: expand along them with a visited set, access checks, relevance ranking and a budget.
- "How do you know hierarchical retrieval helps?" -> ablation on labelled cases; I measured it and on a small corpus it did not, so it's opt-in.
- "How do you stop leaking restricted documents?" -> filter at retrieval (also through links), leak detector on output, no-oracle 404s, tests and evals at 0 leaks.
- "A user asks two questions in one message." -> planner splits, retrieve and grade per sub-question, merge.
- "What breaks when docs change?" -> incremental sync by hash + enrichment signature, stale-entry deletion, cache keyed by prompt version, lint gate in CI.
