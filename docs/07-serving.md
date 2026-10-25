# 7. Serving: the web UI, API, streaming

> Status: 🚧 the UI in [`web/`](../web/) runs on the **real pipeline** through a FastAPI service
> ([`src/rag/api/`](../src/rag/api/)) that streams each stage. Kubernetes docs only; uploads are
> the next Phase 5 step. Deploy runbook: README "Deploy".

## Concept

A RAG system is usually a black box: a question goes in, an answer comes out. For a demo (and
for debugging) the interesting part is everything in between. The "glass-box" UI shows each
stage of a real run:

```
Source        Pipeline                                                     Ask
k8s docs  →   Parse → Chunk → Embed → Index      (built ahead of time)     question
              Dense ─┐
                     ├→ Fusion (RRF) → Rerank → Generate → Cite        →   answer with [n]
              Keyword┘
```

Two ideas carry it:

- **Show intermediate results, not just spinners.** Each search stage shows its top hits with
  scores and its time. Rerank shows rank changes (↑3, ↓2), so a viewer sees *why* the final
  passages were chosen.
- **Trace every claim.** Hovering a citation `[n]` highlights the same chunk in every list and
  opens the exact passage. That is what "grounded" means, made visible.

### Why streaming, and why server-sent events

Answering takes a few seconds (rerank ~1.5 s, the free LLM 3-6 s). The UI needs results *while*
they happen, so the server pushes updates:

| | Polling | Server-sent events (SSE) | WebSocket |
|---|---|---|---|
| Direction | client asks repeatedly | server → client | both ways |
| Complexity | trivial, wasteful | one long HTTP response, plain text lines | separate protocol |
| Fits | rare updates | progress + token streams | chat with interruptions, games |

We only need server → client, so SSE is the simplest fit. The browser reads it with `fetch` and a
stream reader (not `EventSource`, which can't send a POST body).

## In this repo

### The API ([`src/rag/api/app.py`](../src/rag/api/app.py))

`POST /api/ask` with `{"question", "k", "rerank", "max_one_per_page"}` answers with a stream:

| Event | When | Payload |
|---|---|---|
| `stage` | after retrieval, one per stage: `dense`, `keyword`, `fusion`, `rerank` | `name`, `ms`, top 10 `hits` (id, heading, url, content, score) |
| `context` | once | the passages the LLM will read (after rerank and the one-per-page cap) |
| `token` | as the LLM writes | `text` |
| `done` | end | full `text`, `refused`, `sources` (`[n]` → chunk id), `invalid` citations, total `ms` |
| `error` | instead of `done` | `code` (`llm_failed`, `daily_limit`, `failed`), human `message` |

Two small changes made this possible without duplicating pipeline logic:

- `Retriever.retrieve(query, config, on_stage=...)` calls a hook after each stage with its hits
  and timing. The CLI and evals don't pass it, so nothing changes for them.
- `stream_answer(question, hits, generator)` yields tokens, then the validated `Answer` last.
  `answer_question` is built on it, so citation checking lives in one place.

Other endpoints: `GET /health` (process alive), `GET /ready` (database reachable and the index is
non-empty: Railway waits for it before routing traffic), `GET /api/collection` (counts, model
names for the UI footer).

### Guards ([`src/rag/api/guards.py`](../src/rag/api/guards.py))

A public demo will meet bots and enthusiastic testers. Each guard answers one failure:

| Guard | Default | Protects against |
|---|---|---|
| Input validation | 1-500 chars, `k` 3-10 | huge prompts, nonsense configs (HTTP 422) |
| Rate limit per IP | 10 / minute | one person or bot hammering the API (HTTP 429 + `Retry-After`). Keyed on the *last* `X-Forwarded-For` entry, the one Railway's edge adds: earlier entries are written by the client and could be forged |
| Daily LLM budget | 300 calls / UTC day | draining the OpenRouter credits; retrieval keeps working |
| Concurrency cap | 2 asks at once | CPU exhaustion: the reranker uses all cores (HTTP 503 "busy") |
| CORS | only `RAG_CORS_ORIGINS` | other websites using the API from visitors' browsers |

A slot is released by the stream's `finally` *and* when the stream object is garbage-collected:
a client that disconnects before the first byte makes Starlette drop the stream unstarted, and
without the second path that slot would be lost until a restart (found in review, now tested).

They live in process memory: simple and correct for **one replica**. Two replicas would each
count separately; that's when Redis would be needed.

### Privacy and logging

The API stores nothing. One JSON log line per ask holds a random id, the status and stage
timings, never the question or the IP (uvicorn's access log, which prints IPs, is disabled in the
Docker image). The UI says questions go to a third-party model and links `/privacy`; `/legal`
credits the Kubernetes docs (CC BY 4.0 requires attribution). `next.config.ts` sets a Content
Security Policy: the page may only talk to itself and the API.

### Deployment

```
browser ──► Vercel (Next.js UI) ──► Railway API (FastAPI + ONNX models) ──► Railway Postgres
                                          └──► OpenRouter (free LLM)
```

- The index is **copied, not rebuilt**: `scripts/publish_index.sh` dumps `documents` + `chunks`
  (vectors included) from the local database into Railway's. Re-embedding 17,730 chunks would
  take 35-60 CPU minutes; copying takes ~15 seconds.
- The Docker image bakes in the embedding and reranker models, so a cold start doesn't download
  150 MB before answering.

### The UI ([`web/`](../web/))

| File | What |
|---|---|
| `lib/api.ts` | `getCollection`, `askStream` (fetch + SSE parser), `toChunk`, `splitCitations` |
| `components/GlassBox.tsx` | maps each SSE event onto the view-model; errors become messages, never a frozen screen |
| `components/Panels.tsx`, `Overlays.tsx` | Source, Pipeline, Ask, popovers, settings, ⌘K palette |
| `app/privacy`, `app/legal` | what is processed and stored, publisher, hosting, content credits |

## Trade-offs

- **Stages arrive together.** The retriever runs inside one request; the stage events are sent
  as soon as retrieval finishes (~1.5 s), then tokens stream. Timings shown are real.
- **In-memory guards** mean one replica. Fine for a side project; Redis when traffic grows.
- **One DB connection per request** instead of a pool: simplest, adds a few ms. Pool if needed.
- **Free model**: $0 per answer, but rate limits upstream and prompts may be logged by the provider.
- **Next.js + FastAPI** is two apps and two languages, chosen for a stronger demo UI.

## Try it

```bash
docker compose up -d
uv run uvicorn rag.api.app:main --factory --port 8000
cd web && npm run dev                # http://localhost:3000
```

- Ask "How do I roll back a failed Deployment?", hover `[1]`, compare the Rerank list with Fusion.
- Ask "Who won the 2018 World Cup?": the model declines and Cite is skipped.
- Turn rerank off (`R`) and ask again: which passages change, and how much faster is it?
- Stop the API and reload: the UI says the service is unavailable.
- `curl -N -X POST localhost:8000/api/ask -H 'content-type: application/json' -d '{"question":"What is a Pod?"}'`
  shows the raw event stream.
