# 5. Generation: grounded prompts, citations, refusal

> Status: `rag ask` works end to end with Claude (`claude-sonnet-5-5`). First real run below.

## Concept

Retrieval found the passages; generation turns them into an answer. Three properties matter:

1. **Grounded**: every claim comes from the retrieved passages, not the model's memory. The
   model's memory may be outdated, wrong, or about a different Kubernetes version.
2. **Cited**: every claim points to the passage that supports it, so a reader can verify it in
   one click.
3. **Honest**: when the passages don't contain the answer, say so instead of improvising.

An LLM will happily do the opposite of all three if you let it. The prompt asks for these
properties, and the code **checks** what it can.

### Anatomy of the prompt ([`prompt.py`](../src/rag/generate/prompt.py))

```
SYSTEM: You answer questions about Kubernetes using ONLY the numbered context passages...
        - After every sentence that states a fact, cite its passage number(s) like [n].
        - Cite only passage numbers that exist in the context.
        - If the context does not contain the answer, reply with exactly:
          "I don't know based on the provided documentation."

USER:   Context passages:

        [1] Deployments > Rolling Back a Deployment
        Source: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
        Use kubectl rollout undo deployment/nginx-deployment ...

        ---

        [2] ...

        Question: How do I roll back a deployment?
```

Design choices, and why:

- **Numbered blocks** make citations mechanical: `[2]` must mean block 2, which code can check.
- **Heading path + URL in each block** let the model say *where* something comes from, and
  give it the section context that a bare chunk lacks.
- **An exact refusal sentence** turns "did the model refuse?" into a string comparison
  instead of a judgement call, which makes refusals countable in evals.
- **Question last.** Models attend well to the end of the prompt, and the instructions about
  the passages come before the passages.

### Checking citations ([`citations.py`](../src/rag/generate/citations.py))

The answer is parsed for `[1]`, `[1, 3]`, `[2][4]` (but not `a[0]` in code). Each number is
checked against the number of passages:

- **Valid citations** become the *Sources* list, showing only passages actually used, in the
  order cited.
- **Invalid citations** (`[7]` with 5 passages) are **hallucinated sources**. They are never
  turned into links, and the CLI prints a warning. Phase 4 counts them as a metric.

### Refusal, two layers ([`answer.py`](../src/rag/generate/answer.py))

1. **No passages retrieved → refuse without calling the LLM.** It's free, instant and can't
   hallucinate.
2. **Passages retrieved but irrelevant → the model refuses** with the exact sentence, detected
   by `startswith(REFUSAL)`.

A third layer (refuse when the reranker's best score is below a threshold) was **measured and
rejected** in chapter 6: questions with a false premise (a made-up flag or resource) get *high*
reranker scores because the passages look relevant, while the model's own refusal already
catches 14 of 15 unanswerable questions.

## In this repo

| Piece | File |
|---|---|
| Prompt | `generate/prompt.py`: `build_prompt(question, hits) -> (system, user)` |
| Citations | `generate/citations.py`: `parse_citations`, `check_citations` |
| LLM | `generate/generator.py`: `Generator` protocol, `ClaudeGenerator`, `FakeGenerator` |
| Orchestration | `generate/answer.py`: `answer_question(...) -> Answer` |
| CLI | `rag ask "question" [-k 5] [--no-rerank] [--no-generated]` |

`Answer` carries `text`, `sources` (cited only), `invalid_citations`, `refused`, and
`retrieved` (everything the model saw). That last field is for evals: to judge faithfulness
you need exactly what the model was shown.

Each LLM wrapper is a few dozen lines: one request shape, `generate()` for a full answer,
`stream()` for token-by-token output in the CLI. Pick one with `RAG_LLM_BACKEND`:

| Backend | Class | Model setting | Key | Cost |
|---|---|---|---|---|
| `anthropic` | `ClaudeGenerator` | `RAG_LLM_MODEL` (`claude-sonnet-5-5`) | `ANTHROPIC_API_KEY` | ~1.2¢ per answer on Sonnet |
| `openrouter` (default since 2026-10-11) | `OpenAICompatibleGenerator` | `RAG_OPENROUTER_MODEL` (`nvidia/nemotron-3-super-120b-a12b:free`) | `OPENROUTER_API_KEY` | free tier: rate-limited, prompts may be logged |
| `fake` | `FakeGenerator` | – | – | tests only |

`OpenAICompatibleGenerator` speaks the OpenAI Chat Completions format, which most providers
copy (OpenRouter, Groq, Gemini, Ollama, vLLM), so switching provider means changing a base URL
and a model name, not writing code. Two provider quirks it handles: streaming may send chunks
with no choices (keep-alives), and `content` may be `null`.

Rough cost of Phase 4 on Claude (assuming Sonnet at ~$3 / $15 per million input / output
tokens): ≈ $8 per full answer-and-judge run over 100 questions, ≈ $2.50 on Haiku, and $0 for
all retrieval-only evaluations, which use only local models.

**API key handling:** `ANTHROPIC_API_KEY` is read by `Settings` (from the environment or
`.env`) into a `SecretStr`, which prints as `**********` if logged, and is passed to the
client explicitly. pydantic-settings reads `.env` but does *not* export it to the process
environment, so relying on the SDK's own env lookup would silently ignore `.env`.

## First real run (2026-10-10, full index, hybrid + rerank, k=5)

| Question | Result | Verdict |
|---|---|---|
| How do I roll back a deployment? | `kubectl rollout undo …`, `--to-revision=2`, how to verify, notes on revision history; cites [3] and [2] only | ✓ correct, every claim cited |
| How do I see which pods use the most CPU? | Shows `kubectl top pod <name>` for one Pod, says the docs given don't cover sorting, suggests running it per Pod | ✗ missed `kubectl top pod --sort-by=cpu` |
| How do I bake a sourdough bread? | "I don't know based on the provided documentation." | ✓ refused, no sources |

End-to-end time: 4–7 s per question, including Python start-up and loading the embedding and
reranker models.

### Failure analysis #1: "which pods use the most CPU"

Ask **which stage broke** (chapter 1) before fixing anything:

1. **Generation?** No. The model said the passages didn't cover sorting, which was true of the
   5 passages it received. Grounding worked. Its last sentence ("run it per Pod") is uncited
   advice, a small breach of the rules worth tracking.
2. **Retrieval?** Yes. All 5 passages were about *setting* CPU requests and limits, none about
   *observing* usage. Worse, the result depends on phrasing: "which pods **are using** the most
   CPU" ranked the `kubectl top pod` page #1 (chapter 4), while "which pods **use** the most
   CPU" doesn't retrieve it at all.
3. **Chunking?** Partly. The literal `--sort-by=cpu` example lives in `quick-reference.md` in a
   chunk whose heading path is "Delete all pods matching the awk pattern…": a cheat sheet of
   unrelated commands under a misleading heading, so its contextual header points the wrong way.

### Same passages, different model: Claude vs. a free model

Same three questions, same retrieved passages, generated by `nvidia/nemotron-3-super-120b-a12b:free`
via OpenRouter. (`google/gemma-4-31b-it:free` and `gemma-4-26b` both failed with HTTP 429: free
models share upstream capacity, and Google's pool was saturated. That's the practical cost of
"free".)

| Question | Claude Sonnet | Nemotron 3 Super (free) |
|---|---|---|
| Roll back a deployment | ✓ detailed, every claim cited | ✓ concise, every claim cited, adds a `rollout status` step |
| Pods using most CPU | ✗ said the passages don't cover sorting | ~ claims `kubectl top pod` lists *each* pod's CPU, which is true in practice but **not stated in passage [5]** (one named Pod) |
| Sourdough bread | ✓ refused | ✓ refused |

The middle row is the **faithfulness vs. helpfulness** trade-off in one example. Nemotron filled
the gap with outside knowledge: correct here, but unverifiable from its citation, and the same
habit produces confident wrong answers when its memory is wrong. Claude stayed strictly
grounded and was less helpful. An LLM judge (Phase 4) scores this as "unsupported claim".
Neither model can fix the root cause, which is retrieval.

Candidate fixes, none applied yet because they must be *measured*: query rewriting (expand the
question before searching), a smaller chunk size for cheat-sheet pages, more candidates before
reranking. This question goes into the golden set (Phase 4) so every fix is scored against it.

## Trade-offs

- Prompt rules are requests, not guarantees. Only citation *validity* is enforced in code.
  Whether a cited passage really *supports* the sentence (faithfulness) needs an LLM judge
  (Phase 4).
- An exact refusal sentence is brittle if the model paraphrases it ("I don't know, based
  on…"). Evals will show how often that happens.
- Streaming shows tokens immediately, but citations can only be validated after the full
  answer arrives, so the Sources list prints at the end.

## Try it (once `ANTHROPIC_API_KEY` is in `.env`)

```bash
uv run rag ask "How do I roll back a deployment?"
uv run rag ask "How do I see which pods use the most CPU?"
uv run rag ask "How do I bake a sourdough bread?"      # should refuse
uv run rag ask "What does --advertise-address do?" --no-rerank
```

Check each answer: is every sentence cited? Open a cited URL: does it actually say that?
