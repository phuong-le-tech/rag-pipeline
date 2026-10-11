# 7. Serving: the glass-box UI, API, streaming

> Status: 🚧 the UI runs in [`web/`](../web/) on an in-browser **mock engine**. The FastAPI
> backend that streams the real pipeline's progress (server-sent events) is the next step.

## Concept

A RAG system is usually a black box: a question goes in, an answer comes out. For a demo (and
for debugging) the interesting part is everything in between. The "glass-box" UI shows each
stage as it runs:

```
Source        Pipeline                                                     Ask
.md file  →   Parse → Chunk → Embed → Index                                question
              Dense ─┐
                     ├→ Fusion (RRF) → Rerank → Generate → Cite        →   answer with [n]
              Keyword┘
```

Two ideas carry it:

- **Show intermediate results, not just spinners.** Each stage card shows its key number
  (chunks, vector size), its time, and for search stages the top hits with scores. Rerank shows
  rank changes (↑3, ↓2), so a viewer sees *why* the final passages were chosen.
- **Trace every claim.** Hovering a citation `[n]` highlights the same chunk in every list and
  opens the exact passage. That is what "grounded" means, made visible.

### Why streaming, and why server-sent events

Ingesting or answering takes seconds. The UI needs progress *while* it happens, so the server
must push updates. Options:

| | Polling | Server-sent events (SSE) | WebSocket |
|---|---|---|---|
| Direction | client asks repeatedly | server → client | both ways |
| Complexity | trivial, wasteful | one long HTTP response, plain text lines | separate protocol, more moving parts |
| Fits | rare updates | progress + token streams | chat with interruptions, games |

The pipeline only needs server → client updates (stage started, stage done, tokens), so SSE is
the simplest tool that fits. FastAPI returns it as a streaming response.

## In this repo: [`web/`](../web/)

| File | What |
|---|---|
| `app/globals.css` | The Open Design prototype's CSS, verbatim: colour tokens on `:root`, dark default, light theme |
| `components/GlassBox.tsx` | View-model (one mutable store + `render()`), actions (ingest, ask, retry), keyboard shortcuts, hover tracing |
| `components/Panels.tsx` | Source, Pipeline (ingest cards, query flow), Ask, run stats |
| `components/Overlays.tsx` | Citation popover, collection menu, settings, ⌘K command palette |
| `lib/mockEngine.ts` | Stand-in pipeline: Markdown parsing, BM25, hashed 384-dim vectors, RRF, heuristic rerank, extractive answers |

The mock engine exists so the UI could be built and reviewed before the backend. Its numbers
are **illustrative only**: real scores, timings and answers come from `src/rag/` once the API
is wired. The swap point is narrow on purpose: `retrieve()` and `compose()` become calls to
the API, and stage updates arrive as SSE events instead of timers.

## Trade-offs

- **Next.js + FastAPI** means two apps and two languages, in exchange for a stronger UI (decision
  log, 2026-10-11). A FastAPI-served page would be one process but weaker for an interview demo.
- **Mutable view-model** mirrors the prototype's logic one-to-one, which made the port faithful
  and short. Idiomatic React would split state per component; worth doing when real events
  replace the simulated flows.
- **Markdown only** in the first version: one loader already exists, so the first slice is about
  streaming and visibility, not parsing PDFs or fetching URLs safely.

## Try it

```bash
cd web && npm install && npm run dev      # http://localhost:3000
```

- Load the Kubernetes docs (press `K`), ask "How do I roll back a failed Deployment?", hover `[1]`.
- Ask "Who won the 2018 World Cup?": the answer is refused and Generate is skipped.
- Turn rerank off (`R`) and ask again: which passages change?
- Drop your own `.md` file: open "Show chunks" and compare the chunks with the outline.
