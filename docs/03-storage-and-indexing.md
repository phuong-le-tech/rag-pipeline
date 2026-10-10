# 3. Storage and indexing: embeddings, pgvector, full-text

> Status: schema and embedders exist; ingestion into the DB lands next in Phase 1.

## Concept

To search chunks quickly we build two complementary indexes over the same rows.

### Dense: embeddings + vector search

An **embedding model** maps text to a vector (here 384 numbers) so that texts with similar
*meaning* land close together. "How do I undo a rollout?" and "Rolling back a Deployment"
share few words but get nearby vectors.

- Similarity is usually **cosine similarity**: the angle between vectors, ignoring length.
- Exact search compares the query to every vector: fine for thousands, slow for millions.
- **Approximate nearest neighbour (ANN)** indexes trade a little recall for big speed.
  **HNSW** (Hierarchical Navigable Small World) builds a layered graph you can walk greedily
  toward the nearest vectors. It is fast and accurate, but uses more memory and builds slowly.

Dense search is strong on paraphrases and weak on exact tokens: error codes, flag names
(`--dry-run`), version numbers, rare product names.

### Sparse: full-text / BM25-style search

Classic keyword search ranks documents by how often query terms appear, discounted for
common words. **BM25** is the standard formula. Postgres full-text search is BM25-*like*:
it stems words (`rolling` → `roll`), drops stop-words, and ranks with `ts_rank_cd`.

Sparse search is strong on exact identifiers and weak on synonyms. The two failure profiles
are opposite, which is why **hybrid search** (chapter 4) combines them.

## In this repo: embedders, [`src/rag/embed/`](../src/rag/embed/)

Every embedding backend implements one small interface (a `Protocol`):

```python
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...   # passages
    def embed_query(self, text: str) -> list[float]: ...              # search queries
```

Pick one with `RAG_EMBEDDING_BACKEND` (`make_embedder` in `embed/factory.py`):

| Backend | Implementation | Notes |
|---|---|---|
| `local` (default) | `LocalEmbedder`: `BAAI/bge-small-en-v1.5` via fastembed | runs on your machine (ONNX, no torch, no key), 384 dims, 67 MB weights downloaded once, ~5 min for the whole corpus on a laptop CPU |
| `openai` | `OpenAIEmbedder`: `text-embedding-3-small` | needs `OPENAI_API_KEY`; batches of 128 per call, results re-sorted by `index`, ~$0.07 for the corpus. Set `RAG_EMBEDDING_DIM=1536` |
| `fake` | `FakeEmbedder` | tests only: unit vectors seeded by a hash of the text, no network, no cost |

Anthropic has no embeddings endpoint, so an Anthropic key can't be used here. It is used later
for generation (Phase 3) and as an LLM judge (Phase 4).

### Queries and passages are embedded differently

Some models are **asymmetric**. bge was trained so that a *question* and the *passage that
answers it* land close together, but a question looks nothing like a passage: short,
interrogative, different words. So bge encodes queries with an instruction prefix and passages
without. Mixing them up (embedding a query as a passage) silently lowers retrieval quality.
That is why the interface has two methods. OpenAI's models are symmetric, so their
`embed_query` simply calls `embed`.

### See it work

First run of the local model (2026-10-10): cosine similarity between three queries and three
passages:

| Query ↓ / Passage → | "Rolling back a Deployment: use kubectl rollout undo…" | "A Service exposes a set of Pods… stable IP" | "--advertise-address string…" |
|---|---|---|---|
| "how do I undo a bad release?" | **0.716** | 0.431 | 0.418 |
| "stable network address for my pods" | 0.530 | **0.826** | 0.595 |
| "advertise-address flag" | 0.446 | 0.538 | **0.788** |

The first query shares *no* keyword with the rollback passage ("undo a bad release" vs
"rollout undo… revert") and still matches it best. Keyword search would struggle here. Notice
also that scores never reach 0 for unrelated text (~0.4 is "unrelated" for this model):
similarity is relative, so only the *ranking* matters, not the absolute number.

Why bother with a fake? Unit tests must be fast, free and deterministic. The fake tests the
*plumbing* (ordering, storage, SQL) while saying nothing about *quality*. Quality is what the
eval harness measures, with the real model.

Why batch? Models are much faster on many texts at once than one at a time: the local model
processes 64 chunks per pass, and the OpenAI backend sends 128 per HTTP call (~140 calls instead
of ~18,000).

## In this repo: [`src/rag/schema.sql`](../src/rag/schema.sql)

```
documents (id, source UNIQUE, content_hash, metadata JSONB, updated_at)
    │ 1..n, ON DELETE CASCADE
chunks    (id, document_id, ord, heading, content, embedding VECTOR(384), tsv TSVECTOR)
```

- `source UNIQUE` + `content_hash`: re-ingesting compares hashes and only re-embeds changed
  documents. Embedding calls cost money and time.
- `ON DELETE CASCADE`: deleting a document removes its chunks. No orphans in search results.
- `embedding VECTOR(384)` + **HNSW index** with `vector_cosine_ops`. The dimension comes from
  `RAG_EMBEDDING_DIM` and must match the embedding model.
- `tsv` is a **generated column**: Postgres computes it from `heading` (weight **A**) and
  `content` (weight **B**), so a heading match outranks a body match. A **GIN index** makes
  keyword lookup fast.

Why one database? Both indexes live next to the metadata, updates are transactional (a
document's chunks are swapped atomically), and filtering ("only `tasks` pages") is a plain
`WHERE` clause.

## Trade-offs

- Local `bge-small` (384d): free, private, no key, fast enough on CPU. Smaller than API models,
  so possibly less accurate. The OpenAI backend stays available and will be an ablation row
  in Phase 4 to measure the gap.
- Changing the embedding model means re-embedding everything, and changing the dimension means
  migrating the column.
- Postgres FTS uses English stemming; code identifiers can stem oddly. Watch for it in evals.

## Try it (after `docker compose up -d && uv run rag init-db`)

```bash
docker compose exec db psql -U rag -c "\d chunks"
docker compose exec db psql -U rag -c \
  "SELECT to_tsvector('english', 'Rolling back a Deployment with kubectl rollout undo');"
```

Notice the stemming and which words disappear.
