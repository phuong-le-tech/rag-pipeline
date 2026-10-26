# 9. LangChain: the same pipeline, written with a framework

The pipeline and the agent were first written by hand (chapters 4 to 8). This chapter adds a
second implementation with LangChain and LangGraph, so the two can be measured against each
other on the same questions.

## Concept

A framework earns its place when it removes code without changing behaviour. The only honest
way to find out is to hold everything else still: same database, same retrieval, same prompt,
same model, same judge, and change only the orchestration.

## In this repo

[`src/rag/lc/`](../src/rag/lc/) holds the LangChain version. Everything else is shared.

| Piece | Hand-written | LangChain |
|---|---|---|
| Retrieval | `Retriever` (SQL, RRF, rerank) | `HybridRetriever`, a `BaseRetriever` that **wraps** the hand-written one |
| Answer | `answer_question` | `lc_answer_question`: an LCEL chain, `prompt \| model \| parser` |
| Agent | `run_agent` loop | `run_lc_agent`: a LangGraph `StateGraph` (agent, tools, gate, force nodes) |
| Prompt, refusal, citation check | `build_prompt`, `REFUSAL`, `finish_answer` | the same functions |
| Tool behaviour, stop rules | `_run_call`, `_limit_reason` | the same functions |

Choose the implementation with `RAG_IMPL=langchain`, or `--impl langchain` on `rag ask` and
`rag eval-answers`. The web API takes `impl` in the request body.

Install the optional extra first: `uv sync --extra langchain`.

In the web UI, a **LangChain** switch in Settings (also "Toggle LangChain" in the command
palette) picks the implementation per question. It appears only when the server offers it, and
the answer header, including a refusal, says "LangChain" when it answered. It combines with the
Agent switch: the LangGraph agent runs, and its steps appear after the run.

### Things worth knowing

- **Anthropic is not available with `impl=langchain` yet.** `langchain-anthropic` requires
  `anthropic<2`, and this project pins `anthropic>=2.1`. Use `openrouter` (or `fake` offline).
- **Retrieval is not re-implemented on purpose.** LangChain's `PGVector` store would change the
  schema and the scores. The SQL search and the reranker are the measured part of this project.
- **The agent trace appears after the run** with LangChain, in the CLI and the UI (the graph
  returns at the end), instead of live as with the hand-written loop.

## Comparing the two

```bash
rag eval-answers --mode pipeline --impl handwritten
rag eval-answers --mode pipeline --impl langchain
rag compare evals/results/<handwritten>-answers.json evals/results/<langchain>-answers.json
```

Repeat with `--mode agent`. For a cheaper run, pass `--limit N` and a shortened `--unanswerable` file. `rag compare` refuses two runs of the same implementation or of
different modes. The judge model is shared, so it grades both the same way.

"Lines of code" counts non-blank lines in `lc/*.py` against `generate/answer.py`,
`generate/generator.py` and `agent/loop.py`. It understates what LangChain depends on, because
the LangChain side imports the hand-written prompt, tools and citation checks.

## Results

Pipeline mode, one run each on 2026-10-26, same model (`nvidia/nemotron-3-ultra-550b-a55b:free`),
same judge (`dots-studio/dots-3-note-preview:free`), `hybrid+rerank, max 1/page`, k = 5, and the
same half sample: the first 47 answerable and first 7 unanswerable golden questions (54 in total).

| Metric | Hand-written | LangChain |
|---|---:|---:|
| Answer rate (answerable) | 0.872 | 0.872 |
| False refusals | 0.128 | 0.128 |
| Faithfulness (supported claims) | 0.940 | 0.935 |
| Correct: yes | 0.850 | 0.846 |
| Correct: partial | 0.125 | 0.128 |
| Citation coverage (sentences cited) | 0.572 | 0.609 |
| Answers with invented citations | 0.000 | 0.000 |
| Refusal on unanswerable | 1.000 | 1.000 |
| LLM calls per question | 1.00 | 1.00 |
| Tokens per question | 2007 | 2009 |
| Latency p50 (ms) | 9918 | 10693 |
| Latency p95 (ms) | 22979 | 28878 |
| Lines of code | 600 | 326 |

The two implementations are indistinguishable on quality. Every difference is within what one more
run of the same code would change (n = 54, single run, free model). Token use matches to within
2 tokens per question, which confirms the prompts are identical. The latency gap is also within
free-tier noise. The one clear difference is code size: the LangChain orchestration is about half
the lines, but it reuses the hand-written prompt, tools and citation checks, so it is not a
like-for-like count.

### Agent mode

Same model, judge and settings, one run each on a smaller sample: the first 25 answerable and 4
unanswerable questions (29 in total), so the questions are a subset of the pipeline sample.

| Metric | Hand-written | LangChain |
|---|---:|---:|
| Answer rate (answerable) | 1.000 | 1.000 |
| False refusals | 0.000 | 0.000 |
| Faithfulness (supported claims) | 0.864 | 0.925 |
| Correct: yes | 0.913 | 1.000 |
| Correct: partial | 0.087 | 0.000 |
| Citation coverage (sentences cited) | 0.390 | 0.406 |
| Answers with invented citations | 0.000 | 0.000 |
| Refusal on unanswerable | 1.000 | 1.000 |
| LLM calls per question | 2.34 | 2.31 |
| Searches per question | 1.34 | 1.31 |
| Tokens per question | 4534 | 4401 |
| Latency p50 (ms) | 11585 | 15226 |
| Latency p95 (ms) | 29364 | 55974 |
| Lines of code | 600 | 326 |

The agents behave alike: about 2.3 model calls and 1.3 searches per question, and about 4,400
to 4,500 tokens. One answer in each run hit the step limit. The quality gap (correct 0.913
against 1.000) is one or two answers out of 23 graded, so it is noise at this sample size, not a
LangChain advantage. The latency gap (p95 29 s against 56 s) comes from a handful of slow calls
on a rate-limited free model in single runs. It is not evidence that the graph itself is slower: its
own overhead is milliseconds next to seconds of model time.

Both samples are halves or less of the 109-question golden set, each configuration ran once, and
the free model can change between days, so treat the tables as "no visible difference", not as
proof of equivalence. The raw result files are in `evals/results/` (gitignored); re-run
`rag eval-answers --impl ...` and `rag compare` to reproduce.

## Trade-offs

- A second implementation to keep in step with the first. The shared helpers keep that small,
  and the paired tests (`tests/test_lc_*.py`) pin the behaviours that must match.
- LangChain adds about 20 packages to the lockfile, so it is an optional extra.
