# Plan: RAG pipeline portfolio project

**Complexity**: Large (~5 weeks part-time) · **Approved**: 2026-10-10

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
| UI | FastAPI + small Next.js page |

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
- [x] `RetrieverConfig(mode, k, candidates, include_generated, rrf_k)`; CLI `rag search`
- [x] Fixed heading-anchor noise; `LOADER_VERSION` in fingerprint; full re-ingest running

### Phase 3 — Cited generation (M)
- [x] numbered-context prompt, `[n]` citation rule, exact refusal sentence
- [x] Claude streaming wrapper (`claude-sonnet-5-5`, key via SecretStr setting)
- [x] citation parser + validator (cited ⊆ retrieved)
- [x] refusal on empty retrieval + model refusal; CLI `rag ask`
- [ ] score-threshold refusal: calibrate in Phase 4
- [x] first real Claude run: 2/3 correct, refusal works; failure #1 (CPU sort) = retrieval, logged in docs ch. 5

### Phase 4 — Evaluation (H)
- [x] `evals/golden.jsonl` `{id, question, relevant_sources[], reference_answer, tags[], origin}`
- [x] 25 hand-written questions (pooled corrections after first run)
- [ ] LLM-drafted questions (Nemotron free, retry/backoff) + human review → ~100 total
- [x] hit@k, recall@k, MRR, nDCG@10 (page relevance, chunk ranks)
- [ ] LLM-judge faithfulness/relevance; citation precision; refusal threshold calibration
- [x] `rag eval [--config]` → `evals/results/<ts>.json` + Markdown table (first results in README)
- [ ] fixes to measure: query rewriting, max chunks per page
- [ ] CI subset gate (fail on >5pt recall drop)

### Phase 5 — API + UI + deploy (M)
- [ ] FastAPI `/search`, `/ask` (streaming), validation, per-IP rate limit, daily cost cap
- [ ] Next.js page: answer, clickable citations, chunk panel with scores
- [ ] Dockerfile, Fly.io + Neon, secrets via Fly; per-query latency/cost logs

### Phase 6 — Polish (L)
- [ ] GitHub Actions: lint, tests, eval subset
- [ ] README: ablation table, cost/latency, 3–5 failure cases, GIF, diagram

## Validation
```bash
uv run ruff check . && uv run pytest -m "not integration" --cov=rag
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
