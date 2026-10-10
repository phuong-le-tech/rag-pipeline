# 6. Evaluation: golden sets, metrics, ablations

> Status: retrieval evaluation works (`rag eval`) on 25 hand-written questions. Next: more
> questions (LLM-drafted, human-reviewed) and answer-level metrics (faithfulness, citations).

## Concept

Every earlier chapter ended with "to be measured". This is where it happens. Without
measurement, "hybrid search is better" is an opinion; with it, it's a number you can defend,
or a belief you have to drop.

An evaluation needs three things:

1. **A golden set**: questions with known correct answers, here the pages that answer them.
2. **Metrics** that turn "did retrieval work?" into numbers.
3. **Ablations**: run the same golden set through each configuration, changing one thing at a
   time, and compare.

### The metrics, in plain words

For each question, retrieval returns a ranked list of chunks. Each chunk comes from a page;
the golden set says which pages are relevant.

| Metric | Question it answers | Example |
|---|---|---|
| **Hit@k** | Is *any* relevant page in the top k? | right page at rank 3 → Hit@1 = 0, Hit@5 = 1 |
| **Recall@k** | What share of the relevant pages are in the top k? | 2 relevant pages, 1 found in top 5 → 0.5 |
| **MRR** (mean reciprocal rank) | How high is the *first* relevant result? | rank 1 → 1.0, rank 2 → 0.5, rank 4 → 0.25, not found → 0 |
| **nDCG@10** | Are relevant results near the top, overall? | rewards rank 1 more than rank 9, smoothly (log discount) |

Each is averaged over all questions. **Hit@5 matters most for RAG**, because the LLM reads the
top 5: if the answer isn't there, generation can't recover. **MRR** shows whether the best
passage comes first.

Two design choices in [`metrics.py`](../src/rag/evals/metrics.py):

- **Relevance per page, ranks per chunk.** The golden set lists pages, which stay stable when
  chunk size changes, so chunking experiments don't invalidate the golden set. But k counts
  chunks, because chunks are what the LLM sees.
- **First find only.** Three chunks from the same relevant page count once in nDCG, so a page
  can't inflate the score by being split into many chunks.

## In this repo

| Piece | File |
|---|---|
| Metrics | `evals/metrics.py`: `hit_at_k`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k` |
| Golden set | `evals/golden.jsonl` (committed) + loader `evals/golden.py` |
| Runner | `evals/runner.py`: `evaluate()`, `ABLATIONS`, `to_markdown()`, `write_results()` |
| CLI | `rag eval [--config NAME]... [--limit N]` |

Each run writes `evals/results/<UTC timestamp>.json` (gitignored) with the per-question
rankings, plus **metadata**: corpus commit (`CORPUS_REF`), pipeline fingerprint, reranker model,
number of questions. A number without that context can't be reproduced, so it isn't evidence.
Before running, `rag eval` checks that every golden source exists in the index and aborts on
typos, because a misspelled path would silently score 0 forever.

### The golden set (v1: 25 questions, hand-written)

Each line: `id`, `question`, `relevant_sources` (pages), `reference_answer`, `tags`, `origin`.
Tags let us slice results by question type:

| Tag | Count | Why it's there |
|---|---|---|
| `howto` / `concept` / `troubleshooting` | 17 / 6 / 3 | cover the main kinds of questions |
| `paraphrase` | 7 | worded unlike the docs ("go back to what was running before"), which stresses dense search |
| `identifier` | 2 | exact flags/fields (`--to-revision`, `maxUnavailable`), which stress keyword search |
| `generated` | 4 | answer lives only in auto-generated reference pages |
| `known-failure` | 1 | failure analysis #1 from chapter 5 |

`origin: human` vs `llm` exists because LLM-drafted questions tend to reuse the passage's own
words, which flatters retrieval. They must be reported separately.

## Results (2026-10-10, corpus `b2e60d8`, `local:BAAI/bge-small-en-v1.5:384:1500:loader2`, reranker MiniLM-L-6, 25 questions)

| Config | Hit@1 | Hit@5 | Recall@5 | MRR | nDCG@10 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.520 | 0.600 | 0.467 | 0.576 | 0.502 | 8 |
| keyword | 0.320 | 0.480 | 0.313 | 0.380 | 0.323 | 30 |
| hybrid | 0.480 | 0.640 | 0.460 | 0.547 | 0.483 | 35 |
| **hybrid+rerank** | 0.480 | **0.800** | **0.547** | **0.613** | **0.548** | 1281 |
| hybrid, no generated | 0.520 | 0.680 | 0.547 | 0.562 | 0.519 | 27 |
| hybrid+rerank, no generated | 0.520 | 0.760 | 0.547 | 0.603 | 0.530 | 1309 |

**Read this with the sample size in mind: with 25 questions, one question moves a score by
0.04.** Differences of 0.04–0.08 are within noise. Only large gaps support conclusions:

- **Dense ≫ keyword** (MRR 0.58 vs 0.38). Keyword search with OR semantics is weak on natural
  questions; dense search understands paraphrases.
- **Hybrid ≈ dense.** Fusing keyword results adds noise about as often as it adds hits.
- **Reranking is the biggest single win**: Hit@5 0.64 → 0.80, four more questions answered in
  the top 5, at ~1.3 s per query. For an interactive `rag ask` that already waits seconds for
  the LLM, it's worth it.
- **Dropping generated pages helps plain hybrid** (Recall@5 0.46 → 0.55) but not reranked
  hybrid, where the reranker already pushes generated noise down, and where 4 questions need
  generated pages. Not conclusive at n = 25.

### Lesson 1: an incomplete golden set punishes correct answers

The first run reported hybrid+rerank Hit@5 = **0.68**. Looking at the five questions that failed
in *every* config showed that three retrieved pages that **do** answer the question but weren't
in the golden set:

| Question | Retrieved page | Verdict after reading it |
|---|---|---|
| "My app takes two minutes to start and the kubelet kills it…" | `concepts/workloads/pods/probes.md` | **relevant**: explains startup probes for slow-starting apps (also added to the liveness/readiness question) |
| "Run one copy of a logging agent on every node" | `concepts/cluster-administration/logging.md` | **relevant**: node-level logging agent on each node, as a DaemonSet |
| "CrashLoopBackOff, how do I find out why?" | `concepts/workloads/pods/pod-lifecycle.md` | **not added**: explains what the back-off *is*, not how to debug it |

After adding the two relevant pages, Hit@5 rose to **0.80**: a bigger change than any retrieval
tweak so far. Search-engine evaluation (TREC) handles this with *pooled judgments*: review
what the systems actually retrieve and add relevant pages to the golden set. The rule here: add
a page only after reading it and confirming it answers the question, never just to raise a
score.

### Lesson 2: what still fails (true retrieval failures)

| Question | What came back instead |
|---|---|
| "which pods **use** the most CPU" (failure #1) | pages about *setting* CPU requests |
| "connect from my laptop to a database pod without a Service" | stateful-app and Service tutorials, not port-forwarding |
| "CrashLoopBackOff, how do I find out why?" | the Pod lifecycle page, three times |

All three are worded the way users talk, not the way the docs are titled. Candidate fixes to
measure next: query rewriting (let the LLM restate the question in documentation vocabulary),
and returning fewer chunks per page so one page can't fill the top 5.

## Trade-offs

- 25 questions is enough to catch large effects and debug failures, not to rank close
  configurations. The next step is ~100 questions.
- Hand-written questions encode my own guesses about what users ask. LLM-drafted ones scale but
  echo the docs' wording. Using both, labelled by `origin`, is the compromise.
- Retrieval metrics ignore answer quality. A perfect top 5 can still yield an unfaithful
  answer, which is why answer-level metrics come next.

## Try it

```bash
uv run rag eval                                   # all configs (reranked ones take ~1 min)
uv run rag eval --config dense --config keyword   # quick comparison
uv run rag eval --limit 10                        # smoke test
```

Open the JSON in `evals/results/`, pick a question that failed, and look at `ranked_sources`.
Is it a retrieval failure, or is the golden set wrong?
