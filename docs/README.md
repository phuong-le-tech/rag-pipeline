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
| 3 | [Storage and indexing: embeddings, pgvector, full-text](03-storage-and-indexing.md) | ✅ schema + embedders |
| 4 | Retrieval: dense, BM25, hybrid, RRF, reranking | ⏳ Phase 2 |
| 5 | Generation: prompting, citations, refusal | ⏳ Phase 3 |
| 6 | Evaluation: golden sets, metrics, ablations | ⏳ Phase 4 |
| 7 | Serving: API, streaming, cost, deployment | ⏳ Phase 5 |

Reference:

- [Decision log](decisions.md): every significant choice, with the reasoning
- [Glossary](glossary.md): terms you will meet in RAG papers and docs
