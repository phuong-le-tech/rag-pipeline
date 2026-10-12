# rag-pipeline

[![ci](https://github.com/phuong-le-tech/rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/phuong-le-tech/rag-pipeline/actions/workflows/ci.yml)

Retrieval-Augmented Generation over the Kubernetes documentation (1,652 pages, 17,730 chunks),
built without a framework so every stage is visible and measured: structure-aware chunking,
hybrid search (pgvector + Postgres full-text, fused with RRF), cross-encoder reranking, answers
with validated `[n]` citations and refusal, and an evaluation harness behind every default.

> Status: Phases 1–4 complete (ingestion, retrieval, cited generation, evaluation).
> Phase 5: the "glass-box" web UI ([`web/`](web/)) runs on the real pipeline through a FastAPI
> service ([`src/rag/api/`](src/rag/api/)) that streams every stage; deployable on Railway +
> Vercel (see [Deploy](#deploy)). Uploads come later. See [Roadmap](#roadmap).

**New to RAG?** The [docs](docs/README.md) explain every stage (concept, code, trade-offs,
experiments) and record every measured decision.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Docker. Embeddings and reranking run locally (no
key); answering needs an LLM key (Anthropic, or OpenRouter for free models).

```bash
uv sync                                   # Python 3.12 venv + deps
docker compose up -d                      # Postgres 17 + pgvector on localhost:5433
uv run rag init-db                        # create tables and indexes
scripts/fetch_corpus.sh                   # Kubernetes docs -> data/k8s-docs (commit SHA recorded)
RAG_EMBEDDING_THREADS=6 uv run rag ingest data/k8s-docs   # first run ~35-60 min on CPU, re-runs seconds
uv run rag search "how do I roll back a deployment" --rerank
uv run rag ask "How do I see which pods use the most CPU?"
uv run rag eval                           # retrieval metrics on evals/golden.jsonl
uv run pytest                             # tests (DB tests skip if Postgres is down)
```

Put keys in a local `.env` (gitignored, never commit it): `ANTHROPIC_API_KEY=...` for Claude, or
`OPENROUTER_API_KEY=...` (default backend: a free OpenRouter model), or `ANTHROPIC_API_KEY=...`
plus `RAG_LLM_BACKEND=anthropic` for Claude.

### Commands

| Command | What it does |
|---|---|
| `rag init-db` | Create the pgvector extension, tables and indexes (idempotent) |
| `rag chunk PATH [--preview]` | Show how documents are chunked (no DB) |
| `rag ingest CORPUS_ROOT [--no-prune]` | Incremental load → chunk → embed → store |
| `rag search QUERY [--mode dense\|keyword\|hybrid] [--rerank] [-k N] [--no-generated]` | Ranked passages, no LLM |
| `rag ask QUESTION [-k N] [--rewrite] [--no-rerank]` | Cited answer, streamed |
| `rag eval [--config NAME] [--limit N]` | Retrieval metrics per configuration, split by question origin |
| `rag eval-answers [--config NAME] [--judge-backend B]` | Faithfulness, correctness, citations, refusals (LLM judge) |
| `rag draft-questions --n N --seed S` | LLM-drafted golden questions (resumable) |
| `rag review-drafts` | Human review of drafts: y / e / n / q (resumable) |

### API + web UI

```bash
uv run uvicorn rag.api.app:main --factory --port 8000   # needs the DB and the index
cd web && npm install && npm run dev                     # http://localhost:3000 (Node 22+)
```

The API streams each stage as server-sent events (`stage`, `context`, `token`, `done`, `error`)
and enforces a per-IP rate limit, a daily LLM-call budget and a concurrency cap. The UI reads
`NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

### Configuration

Environment variables or `.env`, prefix `RAG_` (keys have no prefix):

| Variable | Default |
|---|---|
| `RAG_DATABASE_URL` | `postgresql://rag:rag@localhost:5433/rag` |
| `RAG_CHUNK_MAX_CHARS` | `1500` |
| `RAG_EMBEDDING_BACKEND` | `local` (`openai`, `fake`) |
| `RAG_EMBEDDING_DIM` | `384` (1536 for OpenAI) |
| `RAG_EMBEDDING_THREADS` | unset (ONNX decides; 6 was fastest on an M4) |
| `RAG_RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` |
| `RAG_LLM_BACKEND` | `openrouter` (`anthropic`, `fake`) |
| `RAG_LLM_MODEL` | `claude-sonnet-5-5` (Anthropic backend) |
| `RAG_OPENROUTER_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` |
| `RAG_LLM_TIMEOUT_S` | `60` |
| `RAG_CORS_ORIGINS` | `http://localhost:3000` (comma-separated origins allowed to call the API) |
| `RAG_RATE_LIMIT_PER_MINUTE` | `10` (questions per client IP) |
| `RAG_DAILY_LLM_LIMIT` | `300` (LLM calls per UTC day; retrieval keeps working past it) |
| `RAG_MAX_CONCURRENT_ASKS` | `2` (extra requests get HTTP 503 "busy") |
| `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` / `OPENAI_API_KEY` | read from env or `.env` |
| `NEXT_PUBLIC_API_URL` (web) | `http://localhost:8000`; inlined at build time |
| `NEXT_PUBLIC_PUBLISHER_NAME`, `NEXT_PUBLIC_CONTACT_EMAIL` (web) | shown on `/legal` and `/privacy` |

## Architecture

```
OFFLINE   docs ─► loader (frontmatter, Hugo shortcodes, HTML → text) ─► heading-aware chunker
               ─► embedder (bge-small, local) ─► Postgres: pgvector HNSW + weighted tsvector
               (incremental: skipped when content hash + pipeline fingerprint match)

ONLINE    question ─► [optional LLM rewrite] ─► dense search ─┐
                                              keyword search ─┴► RRF fusion ─► cross-encoder rerank
               ─► max 1 chunk per page ─► top 5 ─► grounded prompt ─► LLM ─► answer + validated [n]

EVAL      golden set (94 reviewed questions) ─► hit@k / recall@k / MRR / nDCG per config
          unanswerable set (15) + LLM judge ─► faithfulness, correctness, citations, refusals
```

## Layout

```
src/rag/
  config.py            settings (pydantic-settings, RAG_ prefix, SecretStr keys)
  db.py, schema.sql    connection + schema (documents, chunks, HNSW + GIN indexes)
  cli.py               the `rag` command
  api/                 app.py (FastAPI, SSE stages + answer), guards.py (rate limit, daily budget)
  ingest/              loaders.py (Hugo → Document), chunking.py, pipeline.py (incremental)
  embed/               Embedder protocol: local (fastembed), OpenAI, fake
  retrieval/           search.py (dense, keyword), fusion.py (RRF, per-page cap),
                       rerank.py (cross-encoder), rewrite.py (LLM query rewrite), retriever.py
  generate/            prompt.py, citations.py, generator.py (Claude, OpenAI-compatible), answer.py
  evals/               metrics.py, golden.py, runner.py, drafting.py, review.py, judge.py, answers.py
evals/                 golden.jsonl, unanswerable.jsonl, draft review audit trail (results gitignored)
docs/                  learning docs: one chapter per stage, decision log, glossary
web/                   Next.js glass-box UI (components/, lib/api.ts SSE client, /privacy, /legal)
Dockerfile, railway.json   API image (models baked in) and Railway deploy config
scripts/               fetch_corpus.sh, publish_index.sh (copy the local index to a remote DB)
data/                  corpus and logs (gitignored)
```

## Roadmap

- [x] Phase 1: corpus, loader, chunker, local embeddings, incremental ingestion
- [x] Phase 2: dense, keyword, hybrid (RRF), cross-encoder reranking, `rag search`
- [x] Phase 3: grounded prompt, validated citations, refusal, Claude / OpenRouter, `rag ask`
- [x] Phase 4: golden set (94 reviewed questions), retrieval ablations, answer-level LLM judge,
      measured defaults, CI
- [ ] Phase 5: glass-box web UI: upload a file, URL or text, watch chunking and embedding live,
      ask with visible retrieval steps and clickable citations; deploy
      (done: real pipeline over FastAPI + SSE, privacy/legal pages, deploy config; next: uploads)
- [ ] Phase 6: polish: demo GIF, diagrams, failure-analysis write-up

## Deploy

The API and its Postgres run on Railway, the UI on Vercel. You run these steps; keys never go
through the repo.

1. **Railway Postgres**: new project → add a Postgres service from the `pgvector/pgvector:pg17`
   image (Railway's pgvector template). Copy its *public* connection URL.
2. **Publish the index** from your machine (local DB running): 
   `scripts/publish_index.sh "<public DATABASE_URL>"` → prints 1,652 docs / 17,730 chunks.
3. **Railway API**: add a service from this GitHub repo (it uses `railway.json` + `Dockerfile`).
   Variables: `RAG_DATABASE_URL=${{Postgres.DATABASE_URL}}` (private URL), `OPENROUTER_API_KEY`,
   `RAG_CORS_ORIGINS=https://<your-vercel-domain>`, `RAG_EMBEDDING_THREADS=4`. Give it ≥ 2 vCPU
   and 2 GB RAM (rerank takes ~1.5 s at 4 threads, ~4 s at 2). Keep **one replica**: the guards
   live in memory. Generate a public domain.
4. **Vercel**: import the repo, root directory `web`. Env: `NEXT_PUBLIC_API_URL=https://<api-domain>`,
   `NEXT_PUBLIC_PUBLISHER_NAME`, `NEXT_PUBLIC_CONTACT_EMAIL`. Deploy (env changes need a redeploy:
   `NEXT_PUBLIC_*` values are inlined at build time).
5. **Check**: `curl https://<api-domain>/ready` → chunk count; open the site and ask a question;
   ask 11 questions within a minute → the 11th is rate-limited.

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
