# Plan: RAG pipeline portfolio project

**Complexity**: Large (~5 weeks part-time) · **Approved**: 2026-10-10 · **Status (2026-10-11)**: Phases 1–4 done, Phase 5 next

## Summary
Question answering over the Kubernetes docs with hybrid retrieval (pgvector + Postgres FTS,
RRF fusion, cross-encoder rerank), cited answers with refusal, and an eval harness whose
ablation table is the centrepiece of the README.

## Decisions
| Decision | Choice |
|---|---|
| Corpus | `kubernetes/website` `content/en/docs`, pinned release tag, CC BY 4.0 |
| Embeddings | Local `BAAI/bge-small-en-v1.5` (384d, fastembed); OpenAI `text-embedding-3-small` kept as ablation row
| Reranker | Local `bge-reranker-v2-m3` (sentence-transformers), Cohere-swappable |
| LLM | Claude via Anthropic SDK behind a `Generator` protocol |
| UI | Next.js (Open Design, Linear-inspired) + FastAPI SSE backend; Markdown uploads first |

## Patterns to Mirror
| Category | Source | Pattern |
|---|---|---|
| Naming | `src/rag/cli.py` | kebab-case commands, snake_case single-concern modules |
| Errors (lib) | `src/rag/ingest/chunking.py`, `src/rag/db.py` | `ValueError` with offending value |
| Errors (CLI) | `src/rag/cli.py` `init_db_cmd` | catch specific exc → red message + fix hint → `typer.Exit(1)` |
| Data | `src/rag/ingest/chunking.py`, `src/rag/db.py` | frozen dataclasses, raw psycopg, SQL in `schema.sql` |
| Config | `src/rag/config.py` | `Settings`, `RAG_*` env, pydantic validation |
| Tests | `tests/test_cli.py` | `tests/test_<module>.py`, AAA, `CliRunner`, `monkeypatch` |

## Phases

### Phase 1 — Corpus + ingestion (M)
- [x] `scripts/fetch_corpus.sh`: sparse clone into `data/`, SHA recorded in `CORPUS_REF`
- [x] `ingest/loaders.py`: frontmatter, shortcodes, HTML→text, `generated` flag → `Document`
- [x] Corpus profiled: 1,652 docs, 17,863 chunks; generated = 55% of chunks (kept, tagged)
- [x] `embed/`: `Embedder` protocol (`embed` + `embed_query`); `LocalEmbedder` (default), `OpenAIEmbedder`, `FakeEmbedder`, `make_embedder`
- [x] `ingest/pipeline.py`: incremental by `content_hash` + pipeline fingerprint, prune, one transaction per doc
- [x] CLI `rag ingest <path>`: added/updated/skipped/deleted counts, progress bar
- [x] Integration tests (`@pytest.mark.integration`) against throwaway `rag_test` DB
- [x] Full ingest: 1,652 docs, 17,730 chunks, all `loader2`; re-run skips all 1,652 in 0.0s

### Phase 2 — Retrieval (M)
- [x] dense (cosine, ef_search + iterative scan), keyword (OR tsquery + `ts_rank_cd`)
- [x] `fusion.py` RRF (k=60), pure + unit-tested
- [x] `Reranker` protocol: cross-encoder via fastembed (MiniLM-L-6 default, bge-v2-m3 configurable); `rag search --rerank`
- [x] `RetrieverConfig(mode, k, candidates, include_generated, rrf_k, rerank, max_per_source, rewrite)`; CLI `rag search`
- [x] Fixed heading-anchor noise; `LOADER_VERSION` in fingerprint; full re-ingest done

### Phase 3 — Cited generation (M)
- [x] numbered-context prompt, `[n]` citation rule, exact refusal sentence
- [x] Claude streaming wrapper (`claude-sonnet-5-5`, key via SecretStr setting)
- [x] citation parser + validator (cited ⊆ retrieved)
- [x] refusal on empty retrieval + model refusal; CLI `rag ask`
- [x] score-threshold refusal: measured in Phase 4, not adopted
- [x] first real Claude run: 2/3 correct, refusal works; failure #1 (CPU sort) = retrieval, logged in docs ch. 5

### Phase 4 — Evaluation (H)
- [x] `evals/golden.jsonl` `{id, question, relevant_sources[], reference_answer, tags[], origin}`
- [x] 25 hand-written questions (pooled corrections after first run)
- [x] 97 Nemotron drafts → 69 accepted after review (Claude, reasons logged) → 94 golden questions
- [x] hit@k, recall@k, MRR, nDCG@10 (page relevance, chunk ranks)
- [x] LLM-judge faithfulness/correctness, citation stats, 15 unanswerable questions (`rag eval-answers`)
- [x] answer evals (Nemotron answers, Sonnet judge): faithfulness 0.84, correct 0.79+0.16 partial, 14/15 refusals
- [x] refusal threshold measured: rejected (false premises score high)
- [x] fix #1 max chunks per page: max 1/page Hit@5 0.80 → 0.88
- [x] fix #2 query rewriting: +0.01 Hit@5 overall (helps hand-written only) → opt-in `--rewrite`
- [x] `rag ask` defaults from n=94: hybrid + rerank + max 1/page
- [x] `rag eval [--config]` → `evals/results/<ts>.json` + Markdown table (first results in README)
- [x] CI: lint + unit + pgvector integration tests

### Phase 5 — API + UI + deploy (M) — "glass-box RAG" demo (agreed 2026-10-10, after Phase 4)
- Decided 2026-10-11: Next.js UI + FastAPI SSE; Markdown uploads first, other formats later
- [x] UI shell in `web/`, ported from the Open Design prototype
- [x] Production v1 (k8s only): API + guards, privacy/legal, Docker + Railway + Vercel runbook (`.claude/plans/production-v1.plan.md`)
- [x] Pipeline emits structured progress events (stage, progress, timing) usable by CLI and API
- [ ] `collection` column + retrieval filter: uploads never mix with the k8s index or evals
- [ ] Loaders: PDF (pypdf), URL (trafilatura, SSRF-safe: block private IPs, size/time limits), pasted text
- [x] FastAPI `/ask` streaming with retrieval steps (SSE) — upload/ingest endpoint still to do
- [ ] UI ingest view: load → clean → chunk → embed (progress) → store, chunk previews
- [x] UI ask view: query embed, dense/keyword lists, RRF, rerank before/after, streamed answer
- [x] Clickable [n] citations open the exact passage; invented citations kept as plain text
- [ ] Prebuilt "Kubernetes docs" collection + link to eval results (interview story)
- [x] Limits: per-IP rate limit, daily LLM cap, concurrency cap, input size — upload limits/TTL with uploads
- [ ] Judge output via structured output (tool calling) when the generator interface changes
- [ ] Dockerfile, deploy (e.g. Fly.io + Neon), secrets in the platform; latency/cost logs
- [ ] docs/07-serving.md chapter

### Phase 6 — Polish (L)
- [x] GitHub Actions: lint, unit, integration (pgvector service), green on main
- [ ] CI retrieval-quality gate on a small committed fixture corpus
- [x] README: ablation + answer tables, commands, configuration (kept in sync)
- [ ] README: demo GIF, architecture diagram, failure-case gallery (5 documented in docs ch. 5–6)

## Validation
```bash
uv run ruff format --check . && uv run ruff check --no-cache . && uv run pytest -m "not integration" --cov=rag
docker compose up -d && uv run pytest -m integration
```

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Golden set biased toward dense retrieval | High | hand-written questions, tags, full review |
| Shortcode noise hurts BM25 | Medium | loader fixtures, `rag chunk --preview` review |
| Torch bloats deploy image | Medium | optional dep group / Cohere in prod |
| Demo abuse burns API budget | Medium | rate limit, spend cap, answer cache |
| Upstream corpus drift | Low | pin tag, record in eval results |

## Acceptance
- [ ] All phases complete, coverage ≥ 80%
- [ ] Ablation table with real numbers in README
- [ ] Public demo live
