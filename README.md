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

Retrieval on 25 hand-written questions (corpus `b2e60d8`, local `bge-small-en-v1.5`, reranker
`ms-marco-MiniLM-L-6-v2`, 2026-10-10). Full method and caveats in
[docs/06-evaluation.md](docs/06-evaluation.md).

| Config | Hit@1 | Hit@5 | Recall@5 | MRR | nDCG@10 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.520 | 0.600 | 0.467 | 0.576 | 0.502 | 8 |
| keyword | 0.320 | 0.480 | 0.313 | 0.380 | 0.323 | 30 |
| hybrid | 0.480 | 0.640 | 0.460 | 0.547 | 0.483 | 35 |
| **hybrid+rerank** | 0.480 | **0.800** | **0.547** | **0.613** | **0.548** | 1281 |
| hybrid, no generated | 0.520 | 0.680 | 0.547 | 0.562 | 0.519 | 27 |
| hybrid+rerank, no generated | 0.520 | 0.760 | 0.547 | 0.603 | 0.530 | 1309 |

With 25 questions one question = 0.04, so only large gaps are conclusive: dense ≫ keyword, and
reranking lifts Hit@5 from 0.64 to 0.80. The golden set is being extended to ~100 questions.
