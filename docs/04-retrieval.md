# 4. Retrieval: dense, keyword, hybrid

> Status: dense, keyword, hybrid (RRF) and cross-encoder reranking all work via `rag search`.
> **Measured on 94 questions (chapter 6):** dense ≫ keyword, reranking +0.13 Hit@5, and capping
> one chunk per page adds a little more. `rag ask` uses hybrid + rerank + max 1 chunk per page.

## Concept

Retrieval answers one question: **which 5-10 chunks out of ~18,000 should the LLM read?** If
the right chunk is not in that set, nothing downstream can fix it. Generation can only be as
good as retrieval.

Most production systems retrieve in **two stages**:

```
query ──► stage 1: cheap, wide          ──► ~50 candidates ──► stage 2: expensive, precise ──► top 5
          (dense + keyword, milliseconds)                      (reranker, reads every pair)
```

### Dense vs. keyword: opposite strengths

| | Dense (embeddings) | Keyword (full-text) |
|---|---|---|
| Matches | meaning ("undo a bad release" ≈ "rollout undo") | exact words, after stemming |
| Strong on | paraphrases, vague questions | flags (`--dry-run`), error codes, names, versions |
| Weak on | rare identifiers, exact strings | synonyms, wording that differs from the docs |
| Score | cosine similarity (0..1) | `ts_rank_cd` (unbounded, depends on the document) |

Because their failures are different, combining them usually beats either one alone. That
claim is a hypothesis, and Phase 4 tests it.

### Fusing with Reciprocal Rank Fusion (RRF)

You can't add a cosine score of 0.76 to a `ts_rank` of 11.8: different scales, different
meanings. RRF ignores scores and uses only **ranks**:

```
RRF(chunk) = Σ over lists of  1 / (k + rank_in_that_list)        k = 60
```

Example, with `k = 60`:

| Chunk | Dense rank | Keyword rank | RRF score |
|---|---|---|---|
| A | 1 | 3 | 1/61 + 1/63 = **0.0323** |
| B | 2 | 1 | 1/62 + 1/61 = **0.0325** ← winner |
| C | 3 | - | 1/63 = 0.0159 |
| D | - | 2 | 1/62 = 0.0161 |

Being found by *both* searches matters more than being #1 in one. `k` damps the top ranks: with
`k = 0`, rank 1 would be worth twice rank 2; with `k = 60`, they're nearly equal, so agreement
dominates. 60 comes from the original paper (Cormack et al., 2009) and is a robust default.

## In this repo: [`src/rag/retrieval/`](../src/rag/retrieval/)

| File | What |
|---|---|
| `fusion.py` | `rrf()`: pure function, ranks only, deterministic tie-breaking |
| `search.py` | `dense_search()` and `keyword_search()`: one SQL query each, return `Hit`s |
| `retriever.py` | `Retriever.retrieve(query, RetrieverConfig)`: picks dense / keyword / hybrid |

`RetrieverConfig(mode, k, candidates, include_generated, rrf_k)`: every field is a switch the
eval harness will flip, so each choice becomes a row in the ablation table.

Every search is scoped to one **collection** (`RetrieverConfig.collection`, default `"k8s"`):
`d.collection = %(collection)s` sits next to the generated-page filter in both queries, so
uploaded documents can never surface in Kubernetes answers or evals.

### Dense search details

```sql
ORDER BY c.embedding <=> :query_vector   -- <=> is pgvector's cosine distance
LIMIT :candidates
```

Score = `1 - distance`. Two pgvector settings matter, and both bite silently:

- **`hnsw.ef_search`** (default 40) is how many candidates the HNSW index explores. If it is
  lower than your `LIMIT`, you get *fewer rows than asked for*, with no error. We set it to
  `max(40, candidates)` before each search.
- **Filtered search**: with `WHERE generated = false`, HNSW first finds 40 nearest vectors,
  *then* filters, so you might get 10 results back. pgvector 0.8's
  `hnsw.iterative_scan = relaxed_order` keeps scanning until enough rows pass the filter.

### Keyword search details

Postgres's `websearch_to_tsquery` ANDs every word: "how do I quickly roll back" requires
`quick & roll & back` all present, so most chunks are excluded. We build an **OR** query instead
and let ranking sort it out:

```sql
to_tsquery('english', replace(plainto_tsquery('english', :q)::text, ' & ', ' | '))
```

Heading words carry weight **A** and body words weight **B**, so a match in a section title
outranks one in the body.

## First real queries (2026-10-10, partial index of ~6,000 chunks)

"how do I see which pods are using the most CPU":

| Mode | #1 | Right answer (`kubectl top pod`)? |
|---|---|---|
| keyword | Pod Lifecycle > In-place Pod resize | ✗ |
| dense | Kubelet Summary API > PodStats | ✓ at #2 |
| hybrid | Pod-level resource managers > … | ✓ at #2 |

What this shows:

- **Keyword with OR is noisy on natural questions.** "pod" appears in thousands of chunks;
  long chunks that repeat it win. Ranking isn't length-normalised yet, which is a candidate fix
  to measure.
- **Dense understood the intent** and found the right command, on a *generated* reference
  page. That's evidence for keeping generated pages (see the decision log).
- **Hybrid inherited some keyword noise at #1.** This is exactly the gap a reranker closes: it
  reads query and chunk together and judges relevance directly.
- The same query also exposed a **cleaning bug**: headings carried Hugo anchors like
  `{#pod-resize}`. Fixed in the loader (chapter 2).

Single anecdotes are not evidence. Phase 4 replaces them with recall@k over a golden set.

## Stage 2: reranking with a cross-encoder ([`rerank.py`](../src/rag/retrieval/rerank.py))

An embedding model is a **bi-encoder**: it encodes the query and each chunk *separately*, then
compares vectors. That's fast (chunk vectors are precomputed) but shallow, because the two texts
never see each other. A **cross-encoder** feeds `query + chunk` through the model *together*
and outputs one relevance score. Every word of the question can attend to every word of the
passage, so it judges actual relevance, not just topic similarity. It's much more accurate but
must run once per pair at query time, so it is only affordable on a short list.

```
first stage (hybrid) ──► 50 candidates ──► cross-encoder scores 50 pairs ──► keep top k
```

`RetrieverConfig(rerank=True)` switches it on. The first stage then returns `candidates`
results instead of `k`, and the reranker re-scores all of them. Model:
`Xenova/ms-marco-MiniLM-L-6-v2` (80 MB, local via fastembed). Reranker scores are logits on
their own scale; only their order matters.

Measured on 2026-10-10 (Apple M4, CPU shared with a running ingest):

| Query | Hybrid #1 | Hybrid + rerank #1 | Latency (warm) |
|---|---|---|---|
| "how do I see which pods are using the most CPU" | Pod-level resource managers ✗ | **kubectl top pod** ✓ | - |
| "how do I roll back a deployment" | Rolling Back a Deployment ✓ | Rolling Back a Deployment ✓ | 35 ms → 3,700 ms |

The reranker fixed the query hybrid got wrong and kept the one it got right, at ~100× the
latency. Is that worth it? Two anecdotes can't say. Phase 4 measures recall and MRR with and
without it, and `candidates` (50 → 20) is the obvious latency knob.

## Trade-offs

- OR keyword semantics: high recall, noisy precision. Fine as a first stage feeding a reranker;
  weak on its own.
- `candidates = 50` per search: more candidates means better recall for the reranker but more
  reranking cost. Tunable, to be measured.
- RRF throws away score magnitudes: a dense hit with similarity 0.95 and one with 0.51 count the
  same if they share a rank. Weighted fusion exists but needs calibration; RRF needs none.

## Try it

```bash
uv run rag search "how do I roll back a deployment" --mode keyword
uv run rag search "how do I roll back a deployment" --mode dense
uv run rag search "how do I roll back a deployment"            # hybrid
uv run rag search "--dry-run flag" --mode dense                 # exact flag: who wins?
uv run rag search "which pods use most CPU" --no-generated      # without generated pages
uv run rag search "which pods use most CPU" --rerank            # second stage on
```

For each query, ask: which mode found the right page, at which rank, and why?
