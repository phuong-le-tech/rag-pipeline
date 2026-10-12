# 8. Agent mode: the model decides what to search

> Status: 🚧 Phase 1 done (loop, `search_docs` tool, `rag ask --agent` with a step trace).
> Results: pending the Phase 2 evaluation.

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
  `parse_search_args`. Bad arguments (not JSON, empty query, `k` outside 1–10) become an error
  sentence the model reads, not a crash. `SearchTool` runs retrieval with the model's `k`.
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
| Searches | 4 | `RAG_AGENT_MAX_STEPS` | extra tool calls get "Step limit reached"; one forced final turn |
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
  free model. Every extra search adds its own rerank and turn; the total is not measured yet.
- **Cost.** Several model calls per question instead of one. The token budget caps the worst case.
- **Control versus flexibility.** The pipeline does the same thing every time, so it is easy to
  debug. The agent can handle comparisons and multi-part questions, but a bad query is now the
  model's mistake, not a fixed step we can tune.
- **Non-determinism.** The same question can take a different path on another run. Evaluation has
  to score answers over repeated runs and report the path, not just the answer.
- **Free models and tool calling.** Tool-call support and argument quality vary by model. A model
  that sends malformed arguments wastes a step; a model that never calls the tool just answers
  from nothing and is refused by the citation rules.
- **Answers are not streamed in v1.** The loop needs the whole turn to see whether it is a tool
  call or the answer. Streaming the final turn is a later change.
- **Opt-in.** `rag ask` without `--agent` is unchanged, so the measured pipeline defaults stay
  valid.

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
