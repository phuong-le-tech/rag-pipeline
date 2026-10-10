# Glossary

**ANN (approximate nearest neighbour)**: vector search that skips exhaustive comparison for
speed, at a small recall cost. See HNSW.

**Asymmetric embedding model**: a model that encodes queries and passages differently (often
with an instruction prefix for queries) because they look different. bge and e5 are
asymmetric; OpenAI's embedding models are symmetric.

**BM25**: classic keyword ranking formula: term frequency, inverse document frequency, length
normalisation. Postgres full-text ranking is BM25-like.

**Bi-encoder**: model that embeds query and passage separately and compares the vectors
(what embedding search does). Fast, precomputable, shallower than a cross-encoder.

**Chunk**: a passage of a document; the unit that is embedded, retrieved and cited.

**Contextual chunk header**: prepending the document title / heading path to a chunk before
embedding it, so the vector carries where the passage comes from.

**Cosine similarity**: angle-based similarity between two vectors, from -1 to 1. Ignores
vector length.

**Cross-encoder / reranker**: model that reads query and passage *together* and scores
relevance. More accurate than embeddings, too slow to run on the whole corpus, so it re-orders
the top-k only.

**Dense retrieval**: search by embedding similarity (meaning).

**Embedding**: fixed-length vector representing text meaning.

**Faithfulness**: whether every claim in an answer is supported by the retrieved context.

**ONNX**: portable format for neural network models; ONNX Runtime runs them fast on CPU
without installing PyTorch. fastembed uses it.

**Iterative index scan**: pgvector 0.8 feature that keeps walking the HNSW graph when a
`WHERE` filter discards candidates, so filtered queries still return `LIMIT` rows.

**Incremental ingestion**: re-indexing only documents whose content (or processing
configuration) changed, detected via hashes, instead of rebuilding the whole index.

**ef_search (HNSW)**: how many candidates the HNSW index explores per query. Higher means
better recall but slower. It also caps the number of rows returned, so it must be ≥ `LIMIT`.

**Grounding**: constraining an LLM to answer only from supplied context rather than its
training data.

**Faithfulness vs. helpfulness**: the tension between answering strictly from sources (safe,
sometimes unhelpful) and filling gaps from model knowledge (helpful, unverifiable, sometimes
wrong).

**Golden set**: hand-checked questions with known relevant sources and reference answers, used
to measure the system.

**Hallucinated citation**: a citation pointing to a source that wasn't provided (e.g. `[7]`
when only 5 passages were given). Detectable mechanically, unlike unsupported claims.

**HNSW**: graph-based ANN index; fast, accurate, memory-hungry.

**Hybrid search**: combining dense and sparse results.

**MRR (mean reciprocal rank)**: average of 1/rank of the first relevant result. Rewards putting
the right chunk first.

**nDCG**: ranking metric that rewards relevant results near the top, with graded relevance.

**Recall@k**: share of questions whose relevant source appears in the top k results.

**Pooled judgments**: building relevance labels by reviewing what retrieval systems actually
return and adding the truly relevant results (TREC practice). Fixes false negatives from an
incomplete golden set.

**Hit@k**: 1 if any relevant result appears in the top k, else 0; averaged over questions.

**RRF (reciprocal rank fusion)**: merges ranked lists by summing `1 / (k + rank)`; needs no
score calibration between systems.

**Sparse retrieval**: keyword-based search (BM25, full-text).

**Stemming**: reducing words to a root (`rolling` → `roll`) so variants match.

**tsvector / GIN**: Postgres full-text document representation and the index type that makes it
fast.
