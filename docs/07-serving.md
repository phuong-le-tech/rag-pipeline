# 7. Serving: the web UI, API, streaming

> Status: 🚧 the UI in [`web/`](../web/) runs on the **real pipeline** through a FastAPI service
> ([`src/rag/api/`](../src/rag/api/)) that streams each stage. Ask over the Kubernetes docs (the
> demo default), or bring your own documents (Markdown, text or PDF, up to 20 files) and watch
> them being indexed. Deploy runbook: README "Deploy".

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

`POST /api/ask` with `{"question", "k", "rerank", "max_one_per_page", "impl"}` answers with a stream
(`impl` is `handwritten` or `langchain`, see [chapter 9](09-langchain.md); the events are identical). With `"agent": true` the model decides what to search ([chapter 8](08-agent.md)); the stream is the same plus `step` events, and there are no `token` events. `max_one_per_page` only applies to the Kubernetes docs: an upload is often one file, and one chunk per file would leave a single passage. The events:

| Event | When | Payload |
|---|---|---|
| `stage` | after retrieval, one per stage: `dense`, `keyword`, `fusion`, `rerank` | `name`, `ms`, top 10 `hits` (id, heading, url, content, score) |
| `step` | agent mode only, after each search | `step`, `query`, `k`, `numbers` (passage numbers), `duplicate`, `error`, `ms` |
| `context` | once | the passages the LLM will read (after rerank and the one-per-page cap); in agent mode every passage the agent saw, in citation order |
| `token` | as the LLM writes | `text` |
| `done` | end | full `text`, `refused`, `sources` (`[n]` → chunk id), `invalid` citations, `impl`, total `ms`; agent mode adds `agent`, `searches`, `llm_calls`, `stopped` |
| `error` | instead of `done` | `code` (`llm_failed`, `daily_limit`, `failed`), human `message` |

Two small changes made this possible without duplicating pipeline logic:

- `Retriever.retrieve(query, config, on_stage=...)` calls a hook after each stage with its hits
  and timing. The CLI and evals don't pass it, so nothing changes for them.
- `stream_answer(question, hits, generator)` yields tokens, then the validated `Answer` last.
  `answer_question` is built on it, so citation checking lives in one place.

Other endpoints: `GET /health` (process alive), `GET /ready` (database reachable and the index is
non-empty: Railway waits for it before routing traffic), `GET /api/collection` (counts, model
names for the UI footer, and which `impls` and whether `agent` mode are offered).

### Uploads ([`src/rag/ingest/upload.py`](../src/rag/ingest/upload.py))

`POST /api/collections` takes a multipart `file` (`.md`, `.txt` or `.pdf`) or JSON `{"text"}`,
and streams the ingest as the same kind of SSE: `stage` events `load → clean → chunk → embed →
store` (each with `ms` and counts; `chunk` carries the first 3 chunk previews), then
`done {collection, docs, chunks}` with the **collection's** totals. Without a `collection` id it
starts a new collection; with one (a multipart `collection` field or a JSON `"collection"` key) it
adds the file to that collection instead, so a visitor builds a small knowledge base one file at a
time. The id travels in the body, never the URL: it is the collection's only key, and proxies
such as Railway's edge log URLs. Ask about it with
`POST /api/ask {"collection": "<id>"}`; `GET /api/collection?id=` returns its counts.

- **Adding to a collection.** The id must already exist (else 404), so a client cannot create a
  collection under an id it chose (and could guess). A file with the same name replaces the
  earlier one (its source `upload:<id>/<name>` is the upsert key). At most 20 documents per
  collection, checked after chunking and before the costly embedding.
- **PDF.** A body that starts with `%PDF-` goes through `pdf_to_text`
  ([`loaders.py`](../src/rag/ingest/loaders.py), `pypdf`) in a worker thread, so a heavy PDF does
  not block the event loop, and only after taking a concurrency slot, so parses are capped like
  asks. Pages are joined with blank lines. Encrypted, broken or over-300-page PDFs, and PDFs whose
  text passes 1,000,000 characters (more than 300 chunks could hold), are refused with 422; a scan with no text layer comes out empty and is refused too (no
  OCR).
- **Topic-neutral prompts.** An ask over an uploaded collection uses `GENERIC_SYSTEM_PROMPT`, and
  agent mode uses `GENERIC_AGENT_SYSTEM` and `GENERIC_SEARCH_DOCS`: same rules, but "the user's
  own documents" instead of "Kubernetes". The `k8s` collection keeps the measured prompts byte for
  byte, so the eval numbers still describe it.

- **The id is the access control.** `uuid4().hex` is 128 random bits: nobody can guess another
  visitor's collection, so there are no accounts. The search filter makes an upload invisible to
  Kubernetes questions and the reverse.
- **Limits** protect a small CPU box: 1 MB body per file (checked from `Content-Length` and while
  reading, before parsing), UTF-8 or PDF only (else 422), at most 300 chunks per file and 20
  files per collection. Uploads have their own per-IP rate limit (20 per minute, so a full
  collection fits in one minute; they never call the LLM, so they don't need the question limit)
  and share the concurrency slots with ask. The web app retries a file once, 2 s later, when the
  slots are busy (503 is refused before anything is stored), so a batch survives a concurrent
  question. On a laptop, embedding 300 chunks took about 12 s with the
  local model; the work runs in the request thread, like ask.
- **No python-multipart.** The multipart body is parsed with the stdlib `email` parser, so the
  API gains no dependency.
- **24 h TTL.** A daemon thread deletes uploaded documents older than `RAG_UPLOAD_TTL_HOURS`
  every 10 minutes (chunks cascade), and each upload also purges first. Expiry used to wait for
  the next upload, which on a quiet day broke the privacy page's "up to 24 hours". The clock is
  per file, so in a collection built over several hours the earliest files expire first. `k8s` is
  never touched. Railway's database backups may still hold deleted files until they expire.
- **Confidentiality.** Embedding is local, but answering sends the top passages to the model
  provider, which may log them. The line under the question box and the privacy page tell visitors not to upload
  confidential files.
- **Nothing is logged** about an upload but a random id, status and total time: not the text,
  not the filename.

The generic loader (`load_upload`) skips the Hugo cleaning: it only normalises newlines, takes the
title from the first H1 (else the filename) and namespaces the source as `upload:<id>/<name>`.

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

The API stores nothing about questions; uploads are kept for up to 24 hours, then deleted. One JSON log line per ask holds a random id, the status and stage
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
| `lib/api.ts` | `getCollection`, `askStream` / `uploadStream` (fetch + SSE parser), `ingestUpdate` (server stage to card), `toChunk`, `splitCitations` |
| `components/GlassBox.tsx` | maps each SSE event onto the view-model; errors become messages, never a frozen screen; holds the visitor's collection id in memory |
| `components/Panels.tsx`, `Overlays.tsx` | Source, Pipeline, Ask, popovers, settings, ⌘K palette |
| `app/privacy`, `app/legal` | what is processed and stored, publisher, hosting, content credits |

The Source panel takes dropped or chosen `.md`, `.txt` or `.pdf` files (a button, so the keyboard
works; several at once are uploaded one after another, stopping at the first failure) or pasted
text. Once a collection exists the button reads "Add files" and every new file joins it. The server's five ingest stages fill the four Pipeline cards (`load` + `clean` are
"Parse"), the first chunk previews open under the Chunk card, and the collection menu then lists
"Kubernetes docs" plus "Your documents"; picking either clears the conversation, because an answer
belongs to the collection it was asked of. Asks pass the collection id as `collection`. The id lives
only in the page's memory: reloading forgets it (the server deletes the data after 24 h anyway).
404 (collection expired, the next file starts a new one), 413, 422, 429 and 503 each get their
own message.

## Trade-offs

- **One collection per session, in memory.** Simple and private, but a reload loses the handle,
  and there is no way to remove one file or start over short of reloading.
- **Neutral prompts are not evaluated.** The golden set is Kubernetes; answers over uploaded
  documents follow the same rules but have no measured Hit@5 or refusal rate.
- **PDF text is flat.** `pypdf` returns no headings, so PDF chunks split by size rather than by
  section, and their heading path is just the title.
- **Previews are the first 3 chunks**, what the server sends, not the whole chunk list.
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
- Drop a short `.md`, watch Parse to Index fill, open the chunks, ask about it, then switch back
  to "Kubernetes docs" in the collection menu.
- Stop the API and reload: the UI says the service is unavailable.
- `curl -N -X POST localhost:8000/api/ask -H 'content-type: application/json' -d '{"question":"What is a Pod?"}'`
  shows the raw event stream.
