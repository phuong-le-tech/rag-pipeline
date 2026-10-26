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

### Things worth knowing

- **Anthropic is not available with `impl=langchain` yet.** `langchain-anthropic` requires
  `anthropic<2`, and this project pins `anthropic>=2.1`. Use `openrouter` (or `fake` offline).
- **Retrieval is not re-implemented on purpose.** LangChain's `PGVector` store would change the
  schema and the scores. The SQL search and the reranker are the measured part of this project.
- **The agent trace prints after the run** in the CLI with LangChain (the graph returns at the
  end), instead of live as with the hand-written loop.

## Comparing the two

```bash
rag eval-answers --mode pipeline --impl handwritten
rag eval-answers --mode pipeline --impl langchain
rag compare evals/results/<handwritten>-answers.json evals/results/<langchain>-answers.json
```

Repeat with `--mode agent`. `rag compare` refuses two runs of the same implementation or of
different modes. The judge model is shared, so it grades both the same way.

"Lines of code" counts non-blank lines in `lc/*.py` against `generate/answer.py`,
`generate/generator.py` and `agent/loop.py`. It understates what LangChain depends on, because
the LangChain side imports the hand-written prompt, tools and citation checks.

## Results

Not run yet. The numbers go here after the first paired run, with the model, corpus commit and
date recorded in each results file's `meta`.

## Trade-offs

- A second implementation to keep in step with the first. The shared helpers keep that small,
  and the paired tests (`tests/test_lc_*.py`) pin the behaviours that must match.
- LangChain adds about 20 packages to the lockfile, so it is an optional extra.
