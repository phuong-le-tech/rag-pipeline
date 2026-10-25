# 8. Agent mode: the model decides what to search

> Status: 🚧 Loop, `search_docs` tool, `rag ask --agent` and the pipeline-vs-agent evaluation are
> done. Multi-part questions: pending owner review. Results below.

## Concept

A RAG pipeline is a **workflow**: the code decides the steps. Every question gets one search,
the top passages go into a prompt, and the model writes an answer. Predictable, cheap, easy to
measure, but it cannot retry a poor search or split a question into parts.

An **agent** gives the model a choice. The model decides whether to search, what to search for,
and how many times. The code only runs the steps the model asks for and enforces limits.

That choice is **tool use**. The app describes a tool to the model (a name, a description and a
JSON schema for its arguments). Instead of text, the model can answer with a tool call. The app
runs the tool, sends the result back, and the model continues. A short transcript:

```
user:       Does a StatefulSet keep its pod names?
assistant:  tool call  search_docs(query="StatefulSet stable pod names")
tool:       [1] StatefulSets: Pod identity  https://kubernetes.io/...
            Each Pod gets a stable hostname such as web-0 ...
assistant:  Yes. Each pod keeps its ordinal name, such as web-0 [1].
```

The model never touches the index itself. It only sends a query string; the app searches,
numbers the passages and returns them. Everything the model knows about the docs comes through
that tool result, so the citation rules from chapter 5 still apply.

## In this repo

- [`src/rag/agent/loop.py`](../src/rag/agent/loop.py): `run_agent` yields a `ToolCallEvent` and
  `ToolResultEvent` for each step, then one `AgentResult`. The CLI prints the steps as they arrive.
- [`src/rag/agent/tools.py`](../src/rag/agent/tools.py): the `search_docs` schema and
  `parse_search_args`. Bad arguments (not JSON, empty query, `k` outside 1-10) become an error
  sentence the model reads, not a crash. `SearchTool` runs retrieval with the model's `k`. The
  CLI's `-k` does not apply in agent mode: the model picks `k` (1-10) for each search.
- [`src/rag/agent/prompt.py`](../src/rag/agent/prompt.py): the agent's rules (answer only from
  tool results, cite numbers it was shown, refuse otherwise) and the text of the forced final turn.
- [`src/rag/generate/generator.py`](../src/rag/generate/generator.py): `ToolGenerator` has one
  `chat` method per provider. `ClaudeGenerator` and `OpenAICompatibleGenerator` translate the
  neutral transcript into each API's tool format; `ScriptedToolGenerator` replays fixed turns for
  tests. `make_tool_generator` picks the backend from `RAG_AGENT_BACKEND`.

### One citation registry for the whole run

Each search numbers its passages, but a second search may find the same chunk. The loop keeps one
registry for the whole question: a chunk gets its number the first time it appears and keeps it.
So `[2]` means the same passage in every turn, and the final answer's citations can be checked
against one list. A search that repeats an earlier query (ignoring case and spacing, and
regardless of `k`) is not run again; the model is told which numbers it already has.

### Limits

Each limit answers one way an agent can go wrong:

| Limit | Default | Setting | What happens when it trips |
|---|---|---|---|
| Steps (searches, including invalid calls, duplicates and empty replies) | 4 | `RAG_AGENT_MAX_STEPS` | extra tool calls get "Step limit reached"; one forced final turn |
| Time | 90 s | `RAG_AGENT_TIMEOUT_S` | checked between turns only, so a slow turn can overrun it |
| Tokens | 30,000 | `RAG_AGENT_MAX_TOTAL_TOKENS` | checked between turns; then the forced final turn |
| Duplicate queries | ignored | none | the model gets the earlier numbers, no second search |
| Forced final turn | tools off | none | the model must answer or refuse; if it returns a tool call or nothing, the answer is the exact refusal sentence |

Every tool call in a turn gets a tool result, even the ones past the limit. The providers reject
the next request otherwise.

The model is called once per step, plus one forced final turn. Each limit has a unit test that
checks the loop stops, so a bad model cannot run up the bill.

## Trade-offs

- **Latency.** Each search reranks (about 1.5 s on CPU), and each model turn takes seconds on a
  free model. Every extra search adds its own rerank and turn. Measured: median 11.2 s against
  5.0 s for the pipeline (see Results).
- **Cost.** Several model calls per question instead of one. The token budget caps the worst case.
- **Control versus flexibility.** The pipeline does the same thing every time, so it is easy to
  debug. The agent can handle comparisons and multi-part questions, but a bad query is now the
  model's mistake, not a fixed step we can tune.
- **Non-determinism.** The same question can take a different path on another run. Evaluation has
  to score answers over repeated runs and report the path, not just the answer.
- **Free models and tool calling.** Tool-call support and argument quality vary by model. A model
  that sends malformed arguments wastes a step. A model that never calls the tool answers without
  passages: the loop accepts that text unless it starts with the refusal sentence, and any `[n]`
  it writes is flagged as an invalid citation. The system prompt tells it to search first and to
  refuse when the passages do not answer, but nothing forces that.
- **Answers are not streamed in v1.** The loop needs the whole turn to see whether it is a tool
  call or the answer. Streaming the final turn is a later change.
- **Opt-in.** `rag ask` without `--agent` is unchanged, so the measured pipeline defaults stay
  valid.

## Results

Same 94 answerable and 15 unanswerable questions for both modes, run with
`rag eval-answers --mode pipeline` and `--mode agent`. Setup, from each results file's `meta`:

- Retrieval `hybrid+rerank, max 1/page`. The pipeline uses k=5; the agent's model picks k per
  search (default 5, max 10).
- Answer model (both modes): `openrouter:nvidia/nemotron-3-super-120b-a12b:free`.
- Judge: `openrouter:dots-studio/dots-3-note-preview:free`, output budget 8192 tokens.
- Agent limits: 4 steps (`max_steps`), 90 s timeout, 30,000 tokens per question.
- Corpus `b2e60d8de29742f310420999423fd50f87a6b6b6`; fingerprint
  `local:BAAI/bge-small-en-v1.5:384:1500:loader2`.
- Results files (gitignored, under `evals/results/`): `2026-10-12T22-55-05Z-answers.json`
  (pipeline) and `2026-10-13T00-03-30Z-answers.json` (agent).

| Metric | Pipeline | Agent |
|---|---:|---:|
| Answer rate | 0.926 | 0.989 |
| False refusals (answerable, refused) | 0.074 (7/94) | 0.011 (1/94) |
| Faithfulness | 0.886 | 0.882 |
| Correct (yes) | 0.798 | 0.844 |
| Correct (partial) | 0.131 | 0.122 |
| Citation coverage (sentences with a citation) | 0.458 | 0.222 |
| Invented citations | 0.000 | 0.000 |
| Refused on unanswerable | 15/15 | 15/15 |
| LLM calls per question | 1.0 | 2.5 |
| Searches per question | 1 fixed (not a tool call) | 1.5 |
| Tokens per question (in + out) | 1,886 (1,500 + 385) | 5,374 (4,587 + 787) |
| Latency p50 (s) | 5.0 | 11.2 |
| Latency p95 (s) | 10.9 | 30.3 |

By origin (correct = yes over graded answers: pipeline 22 hand-written and 62 LLM-drafted, agent
24 and 66; false refusals over 25 and 69):

| Origin | Correct (yes): pipeline | Correct (yes): agent | False refusals: pipeline | False refusals: agent |
|---|---:|---:|---:|---:|
| Hand-written (25) | 0.727 | 0.792 | 0.080 | 0.000 |
| LLM-drafted (69) | 0.823 | 0.864 | 0.072 | 0.014 |

The LLM-drafted questions score higher on both modes, as in chapter 6.

**Reproduction.** The pipeline run reproduces the judge-independent published numbers: false
refusals 0.074 (earlier 0.07) and invented citations 0.000. Unanswerable refusals are 15/15 here
against 14/15 in the earlier run. The refusal is the answer model's output, decided before any
judge runs; the one question that was answered before (`u10-docker-desktop`) is refused this time.
The judged numbers (faithfulness, correctness) come from dots-3, not Claude Sonnet, so they are not
comparable with the Sonnet table in chapter 6.

**Step limit.** 10 of 109 runs stopped at the 4-step limit (`stopped == "max_steps"`): six
answerable (`llm-0011`, `llm-0031`, `llm-0032`, `llm-0035`, `llm-0052`, `llm-0069`) and four
unanswerable (`u05-helm-values`, `u09-eks-ipv6`, `u10-docker-desktop`, `u11-fake-flag`). Five of the
six answerable runs were answered; `llm-0069` was refused, the only answerable false refusal in
agent mode. All four unanswerable runs were refused.

**Judge gaps.** Each mode has two missing faithfulness verdicts (`judge_failures`): `h21-cron-schedule`
and `llm-0022` for the pipeline, `h21-cron-schedule` and `llm-0017` for the agent. The correctness
verdict is also missing once more per mode (`llm-0052` pipeline, `llm-0020` agent). That gap is not
in `judge_failures`; those answers drop out of the correctness denominators.

**Conclusion.** The agent buys back refusals and gives a small correctness gain. It costs about
2.9x the tokens, 2.3x the median latency, and half the citation coverage. It stays opt-in, and the
pipeline stays the default.

Latency includes rate-limit backoff from free-tier 429 retries, which hits the agent harder (about
2.5 model calls per question against one for the pipeline), so the 2.3x median is partly backoff.

**Next.**
- Citation coverage halved: tighten the citation rule in `AGENT_SYSTEM`
  (`src/rag/agent/prompt.py`), then re-run the same 94 questions before changing anything else.
- Four unanswerable questions used all four steps: check whether a lower step cap keeps the
  refusals and saves tokens.
- Multi-part question set (pending owner review): this is where the agent should earn its cost.
- Re-judge both runs with a second judge family, since judged numbers depend on the judge.

## Try it

```bash
uv run rag ask --agent "What is the difference between a Deployment and a StatefulSet?"
```

Read the step lines: each `→ search_docs("...", k=5)` is one model decision. Compare the number of
searches with a simple question such as "What is a Pod?". Then try a backend and model of your
choice (the default is the free OpenRouter model):

```bash
RAG_AGENT_BACKEND=anthropic RAG_AGENT_MODEL=claude-sonnet-5-5 \
  uv run rag ask --agent "How do I roll back a failed Deployment?"
```

Try `RAG_AGENT_MAX_STEPS=1` on the comparison question. What does the forced final turn answer
with, and does it refuse?
