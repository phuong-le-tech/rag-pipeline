# 5. Generation: grounded prompts, citations, refusal

> Status: `rag ask` works end to end with a fake LLM in tests. The first real Claude run waits
> for `ANTHROPIC_API_KEY` in `.env`.

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

A third layer (refuse when the reranker's best score is below a threshold) is easy to add, but
the threshold must come from data, not a guess. Phase 4 will calibrate it on questions that
have no answer in the docs.

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

The LLM wrapper is ~60 lines over the Anthropic SDK: one request shape, `generate()` for a full
answer, `stream()` for token-by-token output in the CLI. Model `claude-sonnet-5-5` by default
(`RAG_LLM_MODEL`), a balance of quality and cost; Haiku is the cheaper ablation row.

**API key handling:** `ANTHROPIC_API_KEY` is read by `Settings` (from the environment or
`.env`) into a `SecretStr`, which prints as `**********` if logged, and is passed to the
client explicitly. pydantic-settings reads `.env` but does *not* export it to the process
environment, so relying on the SDK's own env lookup would silently ignore `.env`.

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
