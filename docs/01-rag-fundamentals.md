# 1. RAG fundamentals

## Concept

An LLM only knows what was in its training data, frozen at a cutoff date, and it cannot
point to where a fact came from. **Retrieval-Augmented Generation (RAG)** fixes both:

1. **Retrieve**: find the passages in *your* documents that are relevant to the question.
2. **Augment**: paste those passages into the prompt as context.
3. **Generate**: ask the LLM to answer *using only that context*, citing it.

The model stops being the source of truth and becomes a reader and summariser of sources
you control. You can update knowledge by re-indexing documents; no retraining.

### The two pipelines

```
OFFLINE (indexing)                      ONLINE (query time)

docs                                    question
 │ load + clean                          │ embed question
 ▼                                       ▼
documents                               search index ──► top-k chunks
 │ chunk                                 │ rerank
 ▼                                       ▼
chunks                                  best chunks
 │ embed                                 │ build prompt
 ▼                                       ▼
index (vectors + keywords)              LLM ──► answer with [n] citations
```

Offline work happens once per document change. Online work happens per question and must
be fast and cheap.

### RAG vs. alternatives

| Approach | Good for | Weak at |
|---|---|---|
| Plain LLM | general knowledge, reasoning | private/fresh data, citations, hallucinations |
| Long context (paste everything) | small corpora (< a few hundred pages) | cost and latency per query, "lost in the middle" |
| Fine-tuning | style, format, domain vocabulary | adding facts reliably, updates, citations |
| **RAG** | large, changing corpora; answers with sources | depends entirely on retrieval quality |

They combine: a fine-tuned model can still use RAG.

### Where RAG fails

Almost every bad answer traces back to one stage. Learn to ask "which stage broke?":

| Symptom | Likely stage | Example |
|---|---|---|
| Right doc exists, never retrieved | **Chunking / retrieval** | Answer split across two chunks; query uses different words than doc |
| Retrieved but ranked too low | **Ranking** | Correct chunk at position 14, only top 5 sent to LLM |
| Retrieved, LLM ignored it | **Generation** | Model answered from its own memory |
| Answer cites the wrong source | **Generation** | Citation number doesn't match the passage used |
| Confident answer, no relevant docs | **Missing refusal** | Should have said "I don't know" |
| Docs say it, but wrongly | **Ingestion** | Stale page, noise from markup, broken table |

This is why the project measures each stage separately (Phase 4): retrieval metrics tell you
if the right chunks were found; answer metrics tell you if the LLM used them faithfully.

## In this repo

| Stage | Module | Phase |
|---|---|---|
| Load + clean | [`ingest/loaders.py`](../src/rag/ingest/loaders.py) | 1 ✅ |
| Chunk | [`ingest/chunking.py`](../src/rag/ingest/chunking.py) | 1 ✅ |
| Store / index | [`schema.sql`](../src/rag/schema.sql), [`ingest/pipeline.py`](../src/rag/ingest/pipeline.py) | 1 ✅ |
| Embed | [`embed/`](../src/rag/embed/) | 1 ✅ |
| Retrieve, fuse, rerank | [`retrieval/`](../src/rag/retrieval/) | 2 ✅ |
| Generate with citations | [`generate/`](../src/rag/generate/) | 3 ✅ |
| Evaluate | [`evals/`](../src/rag/evals/) | 4 ✅ |
| Serve (web UI) | - | 5 ⏳ |

The corpus is the English Kubernetes documentation, about 1,500 Markdown pages. It is
realistic: messy markup, many near-duplicate topics, and users ask real operational questions.

## Trade-offs we accepted

- **Postgres for everything** (vectors + keywords + metadata) instead of a dedicated vector
  DB: one system to run, transactional updates, SQL joins. Cost: tuning is on us at very
  large scale (millions of chunks), which we won't reach.
- **No framework** (LangChain, LlamaIndex) by default: more code to write, but every step is
  visible, testable, and explainable in an interview. A LangChain version of the pipeline and
  agent was added later as an optional second implementation, so the trade-off is measured
  rather than assumed (see [chapter 9](09-langchain.md)).

## Try it

- Pick a question you'd ask about Kubernetes ("How do I roll back a Deployment?"). Before the
  retrieval phase exists, guess which page and section should answer it. Later, check whether
  the system agrees. That is exactly what the golden set (Phase 4) formalises.
