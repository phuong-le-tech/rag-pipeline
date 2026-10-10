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

_Ablation table goes here once evals run._
