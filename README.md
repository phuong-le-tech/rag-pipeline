# rag-pipeline

Production-style Retrieval-Augmented Generation over a real document corpus:
structure-aware chunking, hybrid search (pgvector + Postgres full-text, fused with RRF),
cross-encoder reranking, answers with inline citations — and an eval harness that
measures every change.

> Status: scaffolding. See [Roadmap](#roadmap).

**New to RAG?** The [docs](docs/README.md) explain every stage (concept, code, trade-offs,
experiments) and grow with the project.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                      # Python 3.12 venv + deps
docker compose up -d         # Postgres 17 + pgvector on localhost:5433
uv run rag init-db           # create tables and indexes
uv run rag chunk path/to/docs --preview   # inspect chunking on a folder of .md/.txt
uv run pytest                # tests
```

Configuration comes from environment variables (or a local `.env`), prefix `RAG_`:

| Variable | Default |
|---|---|
| `RAG_DATABASE_URL` | `postgresql://rag:rag@localhost:5433/rag` |
| `RAG_EMBEDDING_BACKEND` | `local` (`openai`, `fake`) |
| `RAG_EMBEDDING_DIM` | `384` (1536 for OpenAI) |
| `RAG_CHUNK_MAX_CHARS` | `1500` |
| `RAG_LLM_BACKEND` | `anthropic` (`openrouter`, `fake`) |
| `RAG_OPENROUTER_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` |

Embeddings run locally by default (no key). `ANTHROPIC_API_KEY` is needed for generation
(Phase 3); `OPENAI_API_KEY` only if you switch `RAG_EMBEDDING_BACKEND=openai`. Never commit `.env`.

## Architecture

```
docs ──► loaders ──► chunker ──► embedder ──► Postgres (pgvector + tsvector)
                                                   │
question ──► dense search ─┐                       │
            BM25 search ───┴► RRF fusion ──► reranker ──► LLM ──► answer + [n] citations
```

## Layout

```
src/rag/
  config.py          settings (pydantic-settings, RAG_ prefix)
  db.py              connection + schema bootstrap
  schema.sql         documents / chunks tables, HNSW + GIN indexes
  cli.py             `rag` command
  ingest/chunking.py heading-aware Markdown chunker
tests/               pytest
evals/               golden set + eval runs (results are gitignored)
data/                raw corpus (gitignored)
```

## Roadmap

- [x] Week 0 — scaffold, schema, chunker
- [ ] Week 1 — loaders (md/html/pdf), embeddings, incremental ingestion
- [ ] Week 2 — hybrid retrieval, RRF, reranking, cited generation
- [ ] Week 3 — golden set, retrieval + answer metrics, ablation table
- [ ] Week 4 — UI, deploy, failure analysis

## Results

Retrieval on 94 reviewed questions (25 hand-written, 69 LLM-drafted; corpus `b2e60d8`, local
`bge-small-en-v1.5`, reranker `ms-marco-MiniLM-L-6-v2`, 2026-10-10). Method, per-group results
and caveats in [docs/06-evaluation.md](docs/06-evaluation.md).

| Config | Hit@1 | Hit@5 | Recall@5 | MRR | nDCG@10 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.543 | 0.766 | 0.725 | 0.647 | 0.662 | 9 |
| keyword | 0.234 | 0.500 | 0.456 | 0.358 | 0.396 | 44 |
| hybrid | 0.511 | 0.766 | 0.697 | 0.609 | 0.620 | 58 |
| hybrid+rerank | 0.628 | 0.894 | 0.821 | 0.742 | 0.748 | 1448 |
| **hybrid+rerank, max 1 chunk/page** (default) | 0.628 | **0.915** | 0.851 | 0.749 | 0.763 | 1314 |
| hybrid+rerank, LLM query rewrite, max 1/page | 0.638 | 0.926 | 0.862 | 0.762 | 0.786 | 1422 |

Key findings: reranking is the robust win (+0.13 Hit@5); capping one chunk per page is a free
gain; LLM-drafted eval questions score ~0.15 higher than hand-written ones, so results are
always reported by question origin; query rewriting helps hand-written questions but not
others, so it is opt-in (`rag ask --rewrite`).

**Answer quality** (default retrieval, answers by Nemotron 3 Super via OpenRouter, judged by
Claude Sonnet, 94 answerable + 15 unanswerable questions):

| Metric | Value |
|---|---:|
| Correct or partly correct vs reference | 0.95 (0.79 + 0.16) |
| Faithfulness (claims supported by the cited passages) | 0.84 |
| Answers with an invented citation | 0.00 |
| Refused on unanswerable / false-premise questions | 14 / 15 |
| Refused although answerable (retrieval misses) | 0.07 |
