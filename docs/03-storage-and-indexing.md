# 3. Storage and indexing: embeddings, pgvector, full-text

> Status: schema and embedders exist; ingestion into the DB lands next in Phase 1.

## Concept

To search chunks quickly we build two complementary indexes over the same rows.

### Dense: embeddings + vector search

An **embedding model** maps text to a vector (here 1,536 numbers) so that texts with similar
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
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
```

| Implementation | Use | Notes |
|---|---|---|
| `OpenAIEmbedder` | real indexing | batches of 128 texts per API call, results re-sorted by `index` (the API may return them out of order), dimension checked, blank text rejected, retries with backoff from the SDK |
| `FakeEmbedder` | tests | unit vectors seeded by a hash of the text: same text gives the same vector, no network, no cost |

Why bother with a fake? Unit tests must be fast, free and deterministic. The fake tests the
*plumbing* (ordering, storage, SQL) while saying nothing about *quality*. Quality is what the
eval harness measures, with the real model.

Why batch? One HTTP call per chunk would mean ~18,000 round trips for this corpus. Batches of
128 make it ~140 calls. At ~3.7M tokens, embedding the full corpus with
`text-embedding-3-small` costs roughly $0.07.

## In this repo: [`src/rag/schema.sql`](../src/rag/schema.sql)

```
documents (id, source UNIQUE, content_hash, metadata JSONB, updated_at)
    │ 1..n, ON DELETE CASCADE
chunks    (id, document_id, ord, heading, content, embedding VECTOR(1536), tsv TSVECTOR)
```

- `source UNIQUE` + `content_hash`: re-ingesting compares hashes and only re-embeds changed
  documents. Embedding calls cost money and time.
- `ON DELETE CASCADE`: deleting a document removes its chunks. No orphans in search results.
- `embedding VECTOR(1536)` + **HNSW index** with `vector_cosine_ops`. The dimension comes from
  `RAG_EMBEDDING_DIM` and must match the embedding model.
- `tsv` is a **generated column**: Postgres computes it from `heading` (weight **A**) and
  `content` (weight **B**), so a heading match outranks a body match. A **GIN index** makes
  keyword lookup fast.

Why one database? Both indexes live next to the metadata, updates are transactional (a
document's chunks are swapped atomically), and filtering ("only `tasks` pages") is a plain
`WHERE` clause.

## Trade-offs

- `text-embedding-3-small` (1536d): cheap (~$0.10 to embed the whole corpus) and good. Local
  models (`bge-m3`) are free and private but need a GPU or patience. Both will be compared.
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
