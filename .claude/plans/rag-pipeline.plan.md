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
| Embeddings | OpenAI `text-embedding-3-small` (1536d); local `bge-m3` later as ablation row |
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
- [ ] `scripts/fetch_corpus.sh`: sparse clone, pinned tag, into `data/`
- [ ] `ingest/loaders.py`: frontmatter (title, weight), strip Hugo shortcodes → `Document(source, url, title, section, text)`
- [ ] `embed/`: `Embedder` protocol; `OpenAIEmbedder` (batch, retry), `FakeEmbedder` (tests)
- [ ] `ingest/pipeline.py`: incremental by `content_hash`, delete vanished docs, one transaction per doc
- [ ] CLI `rag ingest <path>`: added/updated/skipped/deleted counts
- [ ] Integration tests (`@pytest.mark.integration`) against compose DB

### Phase 2 — Retrieval (M)
- [ ] dense (cosine), keyword (`websearch_to_tsquery` + `ts_rank_cd`)
- [ ] `fusion.py` RRF (k=60), pure + unit-tested
- [ ] `Reranker` protocol: cross-encoder + passthrough
- [ ] `RetrieverConfig(mode, rerank, k)`; CLI `rag search`

### Phase 3 — Cited generation (M)
- [ ] numbered-context prompt, `[n]` citation rule
- [ ] Claude streaming wrapper
- [ ] citation parser + validator (cited ⊆ retrieved)
- [ ] refusal under rerank-score threshold; CLI `rag ask`

### Phase 4 — Evaluation (H)
- [ ] `evals/golden.jsonl` `{id, question, reference_answer, relevant_sources[], tags[]}`
- [ ] LLM-drafted questions + human review + 20–30 hand-written
- [ ] recall@k, MRR, nDCG@10; LLM-judge faithfulness/relevance; citation precision
- [ ] `rag eval --config`, `rag eval --ablation` → `evals/results/<ts>.json` + Markdown table
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
