# 6. Evaluation: golden sets, metrics, ablations

> Status: complete. Retrieval evaluated on 94 reviewed questions (10 configurations), answers
> evaluated with an LLM judge (faithfulness, correctness, citations, refusals).

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

### Growing the golden set: draft with an LLM, review by hand

```bash
RAG_LLM_BACKEND=openrouter uv run rag draft-questions --n 100 --seed golden-v2
uv run rag review-drafts        # y = accept, e = edit question, n = reject, q = quit
```

1. **Sample**: one chunk per page, deterministic for a given seed, 15% from generated pages,
   chunks under 300 characters skipped.
2. **Draft**: the LLM writes one question a real user would ask that the chunk answers, in the
   user's words, plus a short reference answer, or `{"skip": true}` for boilerplate.
3. **Flag**: drafts whose content words mostly appear in the passage get the tag
   `high-overlap`. They make retrieval look easier than it is; rewrite or reject them.
4. **Review**: every draft is shown with its answer and source page. Accepted drafts are
   appended to `golden.jsonl` with the tags `reviewed` (and `edited` if you rewrote the
   question). Decisions go to `evals/drafts-reviewed.txt`, so a review can stop and resume.

An unreviewed LLM question never enters the golden set: the person reviewing is the quality
gate.

## Results v2: 94 questions (2026-10-10, corpus `b2e60d8`, `local:BAAI/bge-small-en-v1.5:384:1500:loader2`, reranker MiniLM-L-6, rewrites by Nemotron 3 Super)

25 hand-written + 69 LLM-drafted questions reviewed against their source passages (see
"Growing the golden set"). Results file `evals/results/2026-10-10T23-13-58Z.json`.

| Config | Hit@1 | Hit@5 | Recall@5 | MRR | nDCG@10 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.543 | 0.766 | 0.725 | 0.647 | 0.662 | 9 |
| keyword | 0.234 | 0.500 | 0.456 | 0.358 | 0.396 | 44 |
| hybrid | 0.511 | 0.766 | 0.697 | 0.609 | 0.620 | 58 |
| hybrid+rerank | 0.628 | 0.894 | 0.821 | 0.742 | 0.748 | 1448 |
| hybrid, no generated | 0.543 | 0.766 | 0.709 | 0.626 | 0.636 | 39 |
| hybrid+rerank, no generated | 0.628 | 0.862 | 0.800 | 0.720 | 0.721 | 1370 |
| hybrid+rerank, max 2/page | 0.628 | 0.894 | 0.837 | 0.744 | 0.755 | 1323 |
| **hybrid+rerank, max 1/page** (`rag ask` default) | 0.628 | 0.915 | 0.851 | 0.749 | 0.763 | 1314 |
| hybrid+rerank, rewrite | 0.638 | 0.904 | 0.817 | 0.756 | 0.764 | 3611 |
| hybrid+rerank, rewrite, max 1/page | 0.638 | 0.926 | 0.862 | 0.762 | 0.786 | 1422* |

\* Rewrites were cached from the previous row; live, each new question adds one LLM call.

**By who wrote the question** (Hit@5 / MRR):

| Config | Hand-written (25) | LLM-drafted (69) |
|---|---|---|
| dense | 0.600 / 0.576 | 0.826 / 0.673 |
| hybrid+rerank | 0.800 / 0.613 | 0.928 / 0.789 |
| hybrid+rerank, max 1/page | 0.880 / 0.628 | 0.928 / 0.793 |
| hybrid+rerank, rewrite | 0.920 / 0.714 | 0.899 / 0.771 |
| hybrid+rerank, rewrite, max 1/page | 0.960 / 0.719 | 0.913 / 0.777 |

What the bigger set changed:

1. **LLM-drafted questions are easier, even after review.** Dense search scores 0.83 Hit@5 on
   them vs 0.60 on hand-written ones. They still lean on the passage's vocabulary, even when
   reworded. Without the breakdown, every headline number would look ~0.15 better than it is
   for real users. Always report question origin separately.
2. **Reranking is the robust win.** Hit@5 0.77 → 0.89 and MRR 0.61 → 0.74, consistent in
   every subgroup. Worth its ~1.3 s.
3. **Max 1 chunk per page is a free, consistent gain** (+0.08 Hit@5 on hand-written
   questions, neutral elsewhere). It is now the `rag ask` default.
4. **Query rewriting was overstated by the first 25 questions.** It helps hand-written
   questions a lot (0.80 → 0.92) but slightly hurts ones already phrased like the docs, and it
   costs an LLM call per query. Its 0.96 on the first set mostly reflected my own question
   style. It stays an opt-in flag (`rag ask --rewrite`).
5. **Keep the generated pages.** With reranking, removing them lowers Hit@5 (0.894 → 0.862).
   The open question from chapter 2 is settled.
6. **Dense ≫ keyword** at every sample size.

## Running `eval-answers`: flags and the judge

`rag eval-answers` answers every golden question, then grades the answers with a judge model.
The flags that matter for a comparison:

- `--mode pipeline|agent`: `pipeline` retrieves once and answers; `agent` lets the LLM call the
  search tool (see [08-agent.md](08-agent.md)). Both write the same metrics, plus calls, tokens,
  latency and cost.
- `--k N`: passages per search in pipeline mode only (default: the ablation's k). In agent mode
  the model picks k per search (default 5, max 10), so `--k` does not apply there.
- `--judge-model ID`: the judge model, with `--judge-backend` choosing the provider.
- `--price-in` / `--price-out`: USD per million tokens, so cost per question is computed from the
  token counts (default 0, which reports zero cost).
- `--resume FILE`: continue an interrupted run from its checkpoint. The file must exist, so a
  typo cannot start a full, quota-spending run.

Pick a judge from a different model family than the generator. An LLM judge tends to rate its
own family's wording as correct and faithful, so a same-family judge can hide the errors you
are trying to measure.

## Answer-level results (2026-10-11, `rag eval-answers`)

Retrieval: `hybrid+rerank, max 1/page` (the `rag ask` default). Answers:
`nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter. Judge: `claude-sonnet-5-5`, a
different model family, so it is less likely to go easy on the answers (LLM judges tend to
prefer their own model's outputs). Results file
`evals/results/2026-10-10T23-36-46Z-answers-rejudged.json`.

| Metric | All (94) | Hand-written (25) | LLM-drafted (69) |
|---|---:|---:|---:|
| Answered (not refused) | 0.926 | 0.88 | 0.94 |
| Faithfulness (share of claims supported by the passages) | 0.836 | 0.80 | 0.85 |
| Answers with *every* claim supported | 0.52 | 0.45 | 0.54 |
| Correct (yes / partial) vs reference answer | 0.79 / 0.16 | 0.68 / 0.14 | 0.75 / 0.15 |
| Sentences carrying a citation | 0.41 | | |
| Answers with an invented citation number | **0.00** | | |
| Refused on unanswerable questions | **14 / 15** | | |

How to read it:

- **Correctness is high, strict faithfulness is not.** 95% of answers are correct or partly
  correct, but only half have *every* claim backed by the passages. The typical unsupported
  claim is a true detail from the model's own memory ("`--sort-by=cpu` puts the highest
  first"). That's harmless here and dangerous elsewhere, and it's exactly what a strict judge
  should flag. Nemotron fills gaps more than Claude did in chapter 5.
- **Citations are valid but sparse.** No invented citation numbers in 94 answers, but only
  41% of sentences cite anything: the model cites the main claim and not the follow-ups. A
  stronger answering model, or a prompt example of fully cited answers, is the next lever.
- **The 7 refusals on answerable questions are retrieval misses**, not cowardice. For example
  `h03 "what does --to-revision do"`: dense search treats "revision" ≈ "version" and returns
  CRD versioning pages, and keyword search stems the flag to a common word. The model
  correctly refused rather than guess. Exact-identifier boosting is a candidate fix.
- **The one unanswerable question answered** ("How much memory should I give Docker Desktop…")
  got "300 MiB [1]", lifted from an unrelated resize example: a plausible number grabbed from a
  passage that doesn't apply. Refusal prompts can't fully stop this; a judge pass at answer
  time could.

### Should a low reranker score trigger a refusal? Measured: no

| Threshold on top reranker score | Wrongly refused (of 94 answerable) | Correctly refused (of 15 unanswerable) |
|---|---:|---:|
| −4 | 2 | 5 |
| 0 | 4 | 10 |
| +2 | 7 | 12 |

False-premise questions score *high* (made-up `kubectl --auto-heal` flag: 3.36; fake
"PodAutoBalancer" resource: 4.24) because the passages *look* relevant: real flags, real
resources. The model's own refusal already catches 14 of 15 including those, so a threshold
would only add wrong refusals. Decision: no score threshold.

### Lesson: LLM judges need defensive parsing

On the first run Sonnet returned invalid JSON for **13 of 94 verdicts** (a stray `]` after the
claims list). Re-asking fixed 8; the other 5 failed 3 times in a row, a systematic quirk on
certain inputs. The final parser tries strict JSON, then repairs minor syntax slips
(`json-repair`), then still enforces the expected shape, so garbage is rejected and slips are
not. The cleanest fix is structured output (tool calling with a JSON schema), planned for when
the generator interface next changes. Rule of thumb: count judge failures in every report.
Silently dropped verdicts bias the average.

## Results v1 (2026-10-10, corpus `b2e60d8`, `local:BAAI/bge-small-en-v1.5:384:1500:loader2`, reranker MiniLM-L-6, 25 questions)

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

### Fix #1: cap chunks per page (`max_per_source`), confirmed on 94 questions, now the default

The CrashLoopBackOff question got the Pod Lifecycle page *three times* in its top 5. Capping how
many chunks one page may contribute frees those slots for other pages
([`fusion.cap_per_source`](../src/rag/retrieval/fusion.py), applied after reranking):

| Config | Hit@1 | Hit@5 | Recall@5 | MRR | nDCG@10 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| hybrid+rerank | 0.480 | 0.800 | 0.547 | 0.613 | 0.548 | 1298 |
| hybrid+rerank, max 2/page | 0.480 | 0.800 | 0.607 | 0.619 | 0.572 | 1312 |
| **hybrid+rerank, max 1/page** | 0.480 | **0.880** | **0.640** | **0.628** | **0.584** | 1242 |

Two more questions find a relevant page in the top 5, every metric moves the same way, and
it costs nothing. One caveat: with max 1/page the top 5 are five *distinct* pages, which a
page-level metric naturally rewards. If an answer needs two chunks from the same page, answer
quality could suffer, so the answer-level evaluation must confirm the gain before this becomes
the default for `rag ask`.

### Fix #2: query rewriting (`rewrite`), helps hand-written questions only, opt-in

[`rewrite.py`](../src/rag/retrieval/rewrite.py): an LLM restates the question in documentation
vocabulary ("go back to what was running before" → "kubectl rollout undo deployment previous
revision"). The rewrite is used for **searching only**; the reranker still scores passages
against the **original** question, since that is the user's real intent. Rewrites are cached
so each question costs one LLM call per run. On 25 questions it looked like the biggest win
(Hit@5 0.96); on 94 it adds only +0.01 overall (see Results v2), so it stays `rag ask --rewrite`.

### Operational lessons from the first batch LLM run

- **Set request timeouts.** The SDKs default to ~10 minutes per request plus retries; one stuck
  connection to a free endpoint hung the drafting job. `RAG_LLM_TIMEOUT_S` is now 60 s.
- **Checkpoint every result.** The first drafting run kept 57 drafts in memory and lost them
  when killed. `draft-questions` now appends and flushes each draft and resumes by id.
- **Free tiers have daily caps.** OpenRouter allows 50 free-model requests per day without
  credits, and the cap covers *every* free model on the account, so switching models doesn't help.

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
