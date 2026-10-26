# Measured RAG

[![ci](https://github.com/phuong-le-tech/rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/phuong-le-tech/rag-pipeline/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![next.js](https://img.shields.io/badge/next.js-16-black)
![postgres](https://img.shields.io/badge/postgres-17%20%2B%20pgvector-336791)

Ask questions about the Kubernetes documentation and **watch every step of the answer**: dense and
keyword search, rank fusion, cross-encoder reranking, the streamed answer and the exact passage
behind each citation.

Built without a RAG framework so every stage is visible, tested and measured. Every default is
backed by an evaluation on 94 reviewed questions.

- **Hybrid retrieval**: pgvector (HNSW) + Postgres full-text, fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** and one-chunk-per-page diversity (+0.15 Hit@5 over plain hybrid)
- **Cited answers with refusal**: every `[n]` is validated against the passages sent to the
  model; 0 invented citations, 14/15 unanswerable questions refused
- **Glass-box UI**: each stage streams over server-sent events with its hits, scores and timing;
  drop a Markdown file or paste text to watch it ingested (load, clean, chunk, embed, store) and
  ask questions over it (kept 24 h, then deleted)
- **Opt-in agent mode**: a hand-written tool-calling loop (no framework) where the model decides
  what to search and how often; on the same 94 + 15 questions it cuts false refusals from 7 to 1
  for ~2.9× the tokens, so the fixed pipeline stays the default
- **Production-ready API**: rate limiting, daily LLM budget, concurrency cap, CORS allowlist,
  CSP, no personal data stored; Docker image with models baked in
- **Runs on free tiers**: local embeddings and reranker (no key), free LLM via OpenRouter

**New to RAG?** The [docs](docs/README.md) teach every stage (concept, code, trade-offs,
experiments) and record every measured decision.

## Contents

[How it works](#how-it-works) · [Results](#results) · [Getting started](#getting-started) ·
[CLI](#cli) · [API](#api) · [Configuration](#configuration) · [Deployment](#deployment) ·
[Security and privacy](#security-and-privacy) · [Testing](#testing) · [Project layout](#project-layout) ·
[Limitations](#limitations) · [License](#license) · [Credits](#credits)

## How it works

```
OFFLINE   Kubernetes docs ─► loader (frontmatter, Hugo shortcodes, HTML → text) ─► heading-aware chunker
          ─► embedder (bge-small, local ONNX) ─► Postgres: pgvector HNSW + weighted tsvector
          (incremental: a page is skipped when its content hash and the pipeline fingerprint match)

ONLINE    question ─► dense search ─┐
                     keyword search ─┴► RRF fusion ─► cross-encoder rerank ─► max 1 chunk per page
          ─► top 5 passages ─► grounded prompt ─► LLM (streamed) ─► answer + validated [n] citations

AGENT     question ─► LLM ─► search_docs(query, k) ─► same retrieval as above ─► numbered passages
(opt-in)       ▲                                                                      │
               └──────────── next search, until it answers (max 4 steps) ◄────────────┘
          ─► answer validated against every passage seen during the loop

EVAL      golden set (94 reviewed questions) ─► hit@k / recall@k / MRR / nDCG per configuration
          unanswerable set (15) + LLM judge ─► faithfulness, correctness, citations, refusals
```

In production:

```
browser ──► Vercel: Next.js UI ──► Railway: FastAPI + ONNX models ──► Railway: Postgres + pgvector
                                         └──► OpenRouter (free LLM)
```

## Results

Retrieval on 94 reviewed questions (25 hand-written, 69 LLM-drafted; corpus `b2e60d8`, local
`bge-small-en-v1.5`, reranker `ms-marco-MiniLM-L-6-v2`). Method, per-group results and caveats in
[docs/06-evaluation.md](docs/06-evaluation.md).

| Config | Hit@1 | Hit@5 | Recall@5 | MRR | nDCG@10 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.543 | 0.766 | 0.725 | 0.647 | 0.662 | 9 |
| keyword | 0.234 | 0.500 | 0.456 | 0.358 | 0.396 | 44 |
| hybrid | 0.511 | 0.766 | 0.697 | 0.609 | 0.620 | 58 |
| hybrid+rerank | 0.628 | 0.894 | 0.821 | 0.742 | 0.748 | 1448 |
| **hybrid+rerank, max 1 chunk/page** (default) | 0.628 | **0.915** | 0.851 | 0.749 | 0.763 | 1314 |
| hybrid+rerank, LLM query rewrite, max 1/page | 0.638 | 0.926 | 0.862 | 0.762 | 0.786 | 1422 |

Reranking is the robust win (+0.13 Hit@5) and capping one chunk per page is a free gain.
LLM-drafted questions score ~0.15 higher than hand-written ones, so results are always reported
by question origin. Query rewriting helps hand-written questions only, so it stays opt-in.

**Answer quality** (default retrieval, answers by Nemotron 3 Super via OpenRouter, judged by
Claude Sonnet, 94 answerable + 15 unanswerable questions):

| Metric | Value |
|---|---:|
| Correct or partly correct vs reference | 0.95 (0.79 + 0.16) |
| Faithfulness (claims supported by the cited passages) | 0.84 |
| Answers with an invented citation | 0.00 |
| Refused on unanswerable / false-premise questions | 14 / 15 |
| Refused although answerable (retrieval misses) | 0.07 |

### Pipeline vs agent

Same 94 answerable and 15 unanswerable questions, same retrieval (hybrid+rerank, max 1/page), same
answer model (Nemotron 3 Super, free) and same judge (`dots-3-note-preview`, free, via OpenRouter).
The agent may search up to 4 times per question; the pipeline searches once. The judge differs
from the Claude Sonnet table above, so these judged numbers are not comparable with it.

| Metric | Pipeline | Agent |
|---|---:|---:|
| Answer rate | 0.926 | 0.989 |
| False refusals (answerable, refused) | 0.074 (7/94) | 0.011 (1/94) |
| Faithfulness | 0.886 | 0.882 |
| Correct (yes) | 0.798 | 0.844 |
| Correct (partial) | 0.131 | 0.122 |
| Citation coverage (sentences with a citation) | 0.458 | 0.222 |
| Invented citations | 0.000 | 0.000 |
| Refused on unanswerable | 15/15 | 15/15 |
| LLM calls per question | 1.0 | 2.5 |
| Searches per question | 1 fixed (not a tool call) | 1.5 |
| Tokens per question (in + out) | 1,886 (1,500 + 385) | 5,374 (4,587 + 787) |
| Latency p50 (s) | 5.0 | 11.2 |
| Latency p95 (s) | 10.9 | 30.3 |

By question origin (correct = yes, over the graded answers; false refusals over the 25 or 69
questions):

| Origin | Correct (yes): pipeline | Correct (yes): agent | False refusals: pipeline | False refusals: agent |
|---|---:|---:|---:|---:|
| Hand-written (25) | 0.727 | 0.792 | 0.080 | 0.000 |
| LLM-drafted (69) | 0.823 | 0.864 | 0.072 | 0.014 |

The agent recovers refusals: false refusals fall from 7 to 1 of 94, and correct answers rise
from 0.798 to 0.844 with faithfulness unchanged. It costs about 2.9x the tokens, 2.3x the median
latency, and half the citation coverage (0.222 against 0.458). The correctness gain is small (about 5
points on 84 to 90 graded answers, not tested for significance), so the pipeline stays the
default and agent mode stays opt-in.

Multi-part questions are not evaluated yet, so the agent's expected edge there is untested. Details and the step-limit runs in
[docs/08-agent.md](docs/08-agent.md).

### Hand-written vs LangChain

The pipeline also runs through LangChain (`--impl langchain`, [chapter 9](docs/09-langchain.md)). On the same half sample (54 questions), model, judge and settings, quality, tokens and calls are indistinguishable; only the orchestration code is smaller (326 against 600 lines, not like-for-like).

| Metric | Hand-written | LangChain |
|---|---:|---:|
| Answer rate (answerable) | 0.872 | 0.872 |
| False refusals | 0.128 | 0.128 |
| Faithfulness (supported claims) | 0.940 | 0.935 |
| Correct: yes | 0.850 | 0.846 |
| Correct: partial | 0.125 | 0.128 |
| Citation coverage (sentences cited) | 0.572 | 0.609 |
| Answers with invented citations | 0.000 | 0.000 |
| Refusal on unanswerable | 1.000 | 1.000 |
| LLM calls per question | 1.00 | 1.00 |
| Tokens per question | 2007 | 2009 |
| Latency p50 (ms) | 9918 | 10693 |
| Latency p95 (ms) | 22979 | 28878 |
| Lines of code | 600 | 326 |

Pipeline mode only, one run each; agent mode is not compared yet.

## Getting started

**Prerequisites**: [uv](https://docs.astral.sh/uv/), Docker, Node.js 22+, and an
[OpenRouter](https://openrouter.ai) key for answers (free models work; embeddings and reranking
run locally without any key).

```bash
git clone https://github.com/phuong-le-tech/rag-pipeline && cd rag-pipeline
uv sync                                   # Python 3.12 venv + dependencies
docker compose up -d                      # Postgres 17 + pgvector on localhost:5433
cp .env.example .env                      # gitignored; set OPENROUTER_API_KEY and RAG_OPENROUTER_MODEL
uv run rag init-db                        # tables and indexes
scripts/fetch_corpus.sh                   # Kubernetes docs -> data/k8s-docs (commit SHA recorded)
RAG_EMBEDDING_THREADS=6 uv run rag ingest data/k8s-docs   # first run ~35-60 min on CPU, re-runs take seconds
```

Then run the API and the UI:

```bash
uv run uvicorn rag.api.app:main --factory --port 8000    # API (models load at startup)
cd web && npm install && npm run dev                      # UI on http://localhost:3000
```

Or stay in the terminal:

```bash
uv run rag ask "How do I roll back a deployment?"
```

## CLI

| Command | What it does |
|---|---|
| `rag init-db` | Create the pgvector extension, tables and indexes (idempotent) |
| `rag chunk PATH [--preview]` | Show how documents are cleaned and chunked (no DB) |
| `rag ingest CORPUS_ROOT [--no-prune]` | Incremental load → chunk → embed → store |
| `rag search QUERY [--mode dense\|keyword\|hybrid] [--rerank] [-k N] [--no-generated]` | Ranked passages, no LLM |
| `rag ask QUESTION [-k N] [--rewrite] [--no-rerank]` | Cited answer, streamed |
| `rag ask QUESTION --impl langchain` | Same pipeline through LangChain (LCEL); also works with `--agent` (LangGraph). Needs `uv sync --extra langchain` |
| `rag ask QUESTION --agent [--no-rerank]` | Agent mode: the LLM decides what to search and how many passages (k 1-10) per search; each step printed; answer not streamed |
| `rag eval [--config NAME] [--limit N]` | Retrieval metrics per configuration, split by question origin |
| `rag eval-answers [--config NAME] [--impl handwritten\|langchain] [--judge-backend B] [--mode agent] [--judge-model ID] [--judge-max-tokens N] [--resume FILE]` | Faithfulness, correctness, citations, refusals (LLM judge) |
| `rag compare A.json B.json` | Table of a hand-written and a LangChain `eval-answers` run (same mode) |
| `rag draft-questions --n N --seed S` | LLM-drafted golden questions (resumable) |
| `rag review-drafts` | Human review of drafts: y / e / n / q (resumable) |

## API

FastAPI service in [`src/rag/api/`](src/rag/api/). No authentication: it is a public demo
protected by the [guards](#security-and-privacy) below.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness: the process is up |
| `GET /ready` | Readiness: database reachable and index non-empty (503 otherwise) |
| `GET /api/collection` | Document and chunk counts, LLM and embedding model names |
| `GET /api/collection?id=` | Counts for one upload collection (404 if unknown or expired) |
| `POST /api/collections` | Upload a `.md`/`.txt` file (multipart `file`) or JSON `{"text"}`; streams `stage` events (`load`, `clean`, `chunk`, `embed`, `store`) then `done {collection, docs, chunks}`. Max 1 MB, UTF-8, 300 chunks. Stored up to 24 h, never logged |
| `POST /api/ask` | Answer a question as a server-sent event stream; optional `collection` (32 hex) asks over an upload |

`POST /api/ask` body: `{"question": "...", "k": 5, "rerank": true, "max_one_per_page": true, "impl": "handwritten"}`.
`impl` is `handwritten` (default) or `langchain` (422 `impl_unavailable` when the extra is not installed);
`GET /api/collection` lists the available `impls`, and the `done` event says which one answered.
(`question` 1-500 characters, `k` 3-10). The response is `text/event-stream`:

| Event | Payload |
|---|---|
| `stage` | one per stage (`dense`, `keyword`, `fusion`, `rerank`): `name`, `ms`, top 10 `hits` |
| `context` | the passages sent to the LLM (after rerank and the per-page cap) |
| `token` | a piece of the answer as the model writes it |
| `done` | full `text`, `refused`, `sources` (`[n]` → chunk id), `invalid` citations, total `ms` |
| `error` | replaces `done`: `code` (`llm_failed`, `daily_limit`, `failed`) and a readable `message` |

Errors before the stream starts are JSON: `422` invalid input, `429 {"error": "rate_limited",
"retry_after": s}`, `503 {"error": "busy"}`.

```bash
curl -N -X POST localhost:8000/api/ask -H 'content-type: application/json' \
  -d '{"question": "What is a Pod?"}'
```

## Configuration

Environment variables (or `.env` locally). Settings use the `RAG_` prefix; API keys don't.

| Variable | Default | Notes |
|---|---|---|
| `RAG_DATABASE_URL` | `postgresql://rag:rag@localhost:5433/rag` | |
| `OPENROUTER_API_KEY` | - | needed for answers with the default backend |
| `RAG_IMPL` | `handwritten` | `langchain` answers through LangChain/LangGraph ([chapter 9](docs/09-langchain.md)); needs the `langchain` extra and the `openrouter` or `fake` backend |
| `RAG_LLM_BACKEND` | `openrouter` | `anthropic` (needs `ANTHROPIC_API_KEY`), `fake` (tests) |
| `RAG_OPENROUTER_MODEL` | - | **required** with the default backend; the API refuses to start without it. Pick one from [openrouter.ai/models?q=free](https://openrouter.ai/models?q=free): free models are withdrawn or rate-limited often, so there is no default |
| `RAG_LLM_MODEL` | `claude-sonnet-5-5` | Anthropic backend only |
| `RAG_LLM_TIMEOUT_S` | `60` | per LLM request |
| `RAG_AGENT_BACKEND` | unset (= `RAG_LLM_BACKEND`) | `--agent` only: `anthropic`, `openrouter`, `fake` |
| `RAG_AGENT_MODEL` | unset (= the backend's model) | `--agent` only |
| `RAG_AGENT_MAX_STEPS` | `4` | `--agent` only: steps per question (searches, invalid calls, empty replies) (1-10) |
| `RAG_AGENT_TIMEOUT_S` | `90` | `--agent` only: checked between model turns |
| `RAG_AGENT_MAX_TOTAL_TOKENS` | `30000` | `--agent` only: input + output tokens per question |
| `RAG_EMBEDDING_BACKEND` | `local` | `openai` (needs `OPENAI_API_KEY`, dim 1536), `fake` |
| `RAG_EMBEDDING_DIM` | `384` | must match the embedding model |
| `RAG_EMBEDDING_THREADS` | unset | ONNX threads for embedder and reranker; 4-6 recommended |
| `RAG_RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | |
| `RAG_CHUNK_MAX_CHARS` | `1500` | changing it re-embeds everything on next ingest |
| `RAG_CORS_ORIGINS` | `http://localhost:3000` | comma-separated origins allowed to call the API |
| `RAG_RATE_LIMIT_PER_MINUTE` | `10` | questions per client IP |
| `RAG_DAILY_LLM_LIMIT` | `300` | LLM calls per UTC day; retrieval keeps working past it |
| `RAG_MAX_CONCURRENT_ASKS` | `2` | extra requests get `503 busy` |
| `RAG_UPLOAD_TTL_HOURS` | `24` | uploads older than this are deleted at the next upload |

Web (`web/`, inlined at build time, so changes need a rebuild):

| Variable | Notes |
|---|---|
| `NEXT_PUBLIC_API_URL` | API base URL (default `http://localhost:8000`) |
| `NEXT_PUBLIC_PUBLISHER_NAME`, `NEXT_PUBLIC_CONTACT_EMAIL` | shown on `/legal` and `/privacy`; a Vercel production build fails without them |

## Deployment

The API and Postgres run on [Railway](https://railway.com), the UI on [Vercel](https://vercel.com).
Expect about $5/month for Railway; Vercel's free tier is enough. Keys are set in each provider's
dashboard, never in the repo.

1. **Postgres**: create a Railway project and add Postgres from the `pgvector/pgvector:pg17` image
   (Railway's pgvector template). Copy its public connection URL.
2. **Publish the index** from your machine, with the local database running:
   ```bash
   scripts/publish_index.sh "<public DATABASE_URL>"   # prints 1,652 docs / 17,730 chunks
   ```
   The index is copied with `pg_dump` (~15 s) instead of re-embedded (35-60 min).
3. **API**: add a Railway service from this repo (it uses [`railway.json`](railway.json) and the
   [`Dockerfile`](Dockerfile)). Set `RAG_DATABASE_URL=${{Postgres.DATABASE_URL}}`,
   `OPENROUTER_API_KEY`, `RAG_OPENROUTER_MODEL`, `RAG_CORS_ORIGINS=https://<your-vercel-domain>` and
   `RAG_EMBEDDING_THREADS=4`. Give it ≥ 2 vCPU and 2 GB RAM (rerank takes ~1.5 s at 4 threads,
   ~4 s at 2) and keep **one replica** (the guards live in memory). Generate a public domain.
   Railway waits for `/ready` before routing traffic.
4. **UI**: import the repo in Vercel with root directory `web`, and set `NEXT_PUBLIC_API_URL`,
   `NEXT_PUBLIC_PUBLISHER_NAME` and `NEXT_PUBLIC_CONTACT_EMAIL`.
5. **Check**: `curl https://<api-domain>/ready` returns the chunk count; ask a question on the
   site; ask 11 questions within a minute and the 11th is rate-limited; from another network
   (a phone on mobile data) you can still ask, which confirms the limiter sees real client IPs.

## Security and privacy

| Measure | What it does |
|---|---|
| Input validation | questions 1-500 characters, `k` 3-10, rejected with 422 before any model runs |
| Rate limit | 10 questions per minute per client IP, keyed on the IP the edge proxy adds (not the client-supplied part of `X-Forwarded-For`) |
| Daily LLM budget | 300 model calls per UTC day, so credits can't be drained |
| Concurrency cap | 2 answers at once; slots are always released, even when a client disconnects early |
| CORS | only the configured UI origin may call the API from a browser |
| Security headers | Content Security Policy (page may only talk to itself and the API), HSTS, `nosniff`, strict referrer, no framing |
| Secrets | API keys only in server environment variables (`SecretStr`), never in the browser bundle |
| Container | non-root user, frozen dependencies, uvicorn access log disabled |

**Data**: the app stores no questions or answers. Its logs hold a random request id, status and
stage timings, never question text or IP addresses. IPs live in memory for about a minute for
rate limiting. Questions are sent to the LLM provider, which may log prompts; the UI tells users
not to enter personal data. Details on the in-app [`/privacy`](web/app/privacy/page.tsx) and
[`/legal`](web/app/legal/page.tsx) pages.

## Testing

```bash
uv run ruff format --check . && uv run ruff check --no-cache .
uv run pytest -m "not integration" --cov=rag    # unit tests, no network or keys (coverage ~83%)
uv run pytest -m integration                    # needs docker compose Postgres
cd web && npx tsc --noEmit && npm test && npm run build
```

External services (LLMs, embedders, rerankers) sit behind protocols with deterministic fakes, so
CI needs no keys. [CI](.github/workflows/ci.yml) runs lint, unit and integration tests against a
pgvector service container, plus the web type-check, tests and production build.

## Project layout

```
src/rag/
  api/                 app.py (FastAPI, SSE stream, guards wiring), guards.py (rate limit, daily budget)
  ingest/              loaders.py (Hugo → Document), chunking.py, pipeline.py (incremental)
  embed/               Embedder protocol: local (fastembed), OpenAI, fake
  retrieval/           search.py (dense, keyword), fusion.py (RRF, per-page cap),
                       rerank.py (cross-encoder), rewrite.py (LLM query rewrite), retriever.py
  generate/            prompt.py, citations.py, generator.py (Claude, OpenAI-compatible, tool calls),
                       answer.py
  agent/               loop.py (tool-calling loop, passage registry, limits), tools.py (search_docs),
                       prompt.py
  evals/               metrics.py, golden.py, runner.py, drafting.py, review.py, judge.py, answers.py
  config.py, db.py, schema.sql, cli.py
web/                   Next.js UI: components/, lib/api.ts (SSE client), app/privacy, app/legal
evals/                 golden.jsonl, unanswerable.jsonl, review audit trail (results gitignored)
docs/                  learning docs: one chapter per stage, decision log, glossary
scripts/               fetch_corpus.sh, publish_index.sh
Dockerfile, railway.json, docker-compose.yml
```

## Limitations

- **Uploads are Markdown or plain text only**: no PDF or URL ingestion yet, and uploads expire after 24 h.
- **One API replica**: rate limit, budget and concurrency counters live in process memory.
- **Free LLM**: upstream rate limits can fail an answer (the UI reports it with a retry) and the
  provider may log prompts. Set `RAG_LLM_BACKEND=anthropic` for Claude.
- **No authentication**: the API is public by design and protected only by the guards above.
- **Stage events arrive together** once retrieval finishes (~1.5 s), then the answer streams.
- **Agent mode is opt-in and slower**: median latency about 11 s against 5 s, and its citations
  are sparser (22% of sentences cited against 46% for the pipeline).

## License

Code: [MIT](LICENSE).

## Credits

Answers quote the [Kubernetes documentation](https://github.com/kubernetes/website),
© The Kubernetes Authors, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
UI designed with Open Design.
