# Docs: learning RAG by building one

This folder explains **what** each part of the pipeline does, **why** it exists, and **how**
this repo implements it. It grows with the code: every phase adds or updates a chapter.

Each chapter follows the same shape:

1. **Concept**: the idea, independent of this repo
2. **In this repo**: where it lives, with links to the code
3. **Trade-offs**: what we chose, what we gave up
4. **Try it**: small experiments to build intuition

## Learning path

| # | Chapter | Status |
|---|---|---|
| 1 | [RAG fundamentals](01-rag-fundamentals.md) | ✅ |
| 2 | [Ingestion: loading, cleaning, chunking](02-ingestion.md) | ✅ |
| 3 | [Storage and indexing: embeddings, pgvector, full-text](03-storage-and-indexing.md) | ✅ |
| 4 | [Retrieval: dense, keyword, hybrid, RRF, reranking](04-retrieval.md) | ✅ |
| 5 | [Generation: grounded prompts, citations, refusal](05-generation.md) | ✅ |
| 6 | [Evaluation: golden sets, metrics, ablations](06-evaluation.md) | ✅ |
| 7 | [Serving: the glass-box UI, API, streaming, deployment](07-serving.md) | 🚧 Phase 5 (live demo with uploads; PDF/URL next) |
| 8 | [Agent mode: the model decides what to search](08-agent.md) | 🚧 Loop, CLI and evaluation done; multi-part questions not evaluated yet |

Reference:

- [Decision log](decisions.md): every significant choice, with the reasoning
- [Glossary](glossary.md): terms you will meet in RAG papers and docs
