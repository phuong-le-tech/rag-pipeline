# rag-pipeline — instructions for Claude Code

Portfolio RAG system over the Kubernetes docs: hybrid retrieval (pgvector + Postgres FTS,
RRF fusion, cross-encoder rerank), cited answers with refusal, and an evaluation harness behind
every default. The owner is learning RAG through this project and will demo it in interviews.

## Before any work
- Read `.claude/plans/rag-pipeline.plan.md`: phases, decisions, risks. Tick boxes as tasks land.
- **Status: Phases 1–4 done. Agent mode Phase 1 done (`.claude/plans/agent-mode.plan.md`). Next: Phase 5, the "glass-box" web UI** (upload file/URL/text,
  live pipeline progress, visible retrieval steps, clickable citations). Done: Next.js UI on
  the real pipeline via FastAPI + SSE, Kubernetes docs only, guards, privacy/legal, deploy
  config (`.claude/plans/production-v1.plan.md`). Next in Phase 5: uploads (Markdown first).
- Work the current phase in order; do not start a later phase without being asked.

## Docs are a deliverable
- `docs/` teaches the project: one chapter per pipeline stage, each with
  **Concept → In this repo (code links) → Trade-offs → Try it**. Index: `docs/README.md`.
- Every change that adds or alters behaviour updates docs **in the same commit**: the relevant
  chapter, the status table in `docs/README.md`, new terms in `docs/glossary.md`, and any real
  choice in `docs/decisions.md` (newest first, context / decision / consequence). The decision
  log is history: add entries, don't rewrite old ones.
- Starting a new phase means writing its chapter (Phase 5 → `07-serving.md`).
- Explain the *why* for a learner: plain language, small examples, no copy-pasted API docs.
- Keep README's highlights, CLI, API, Configuration, Deployment and Layout in sync with the code
  (no roadmap in the README; phases live in `.claude/plans/rag-pipeline.plan.md`).

## Commands
```bash
uv sync                                   # Python 3.12 venv + deps
docker compose up -d                      # Postgres 17 + pgvector on localhost:5433
uv run rag init-db                        # schema (idempotent)
scripts/fetch_corpus.sh [ref]             # k8s docs -> data/k8s-docs (+ CORPUS_REF sha)
uv run rag chunk data/k8s-docs --preview  # inspect cleaning + chunking, no DB
RAG_EMBEDDING_THREADS=6 uv run rag ingest data/k8s-docs   # incremental; ALWAYS the corpus root
                                          # (first full run ~35-60 min on CPU, re-runs seconds)
uv run rag search "QUERY" [--mode M] [--rerank]   # ranked passages, no LLM
uv run rag ask "QUESTION" [--rewrite]     # cited answer (defaults: hybrid+rerank+max 1/page)
uv run rag ask "QUESTION" --agent         # agent mode: LLM decides what to search (prints steps)
uv run rag eval [--config NAME] [--limit N]   # retrieval metrics, split by question origin
uv run rag eval-answers [--config NAME] [--judge-backend B] [--mode pipeline|agent] [--judge-model M] [--judge-max-tokens N] [--resume F]   # faithfulness, citations, refusal
uv run rag draft-questions --n 100 --seed S   # resumable
uv run rag review-drafts                  # human review: y/e/n/q, resumable, appends to golden
uv run ruff format . && uv run ruff check --no-cache .   # --no-cache: match CI exactly
uv run pytest -m "not integration" --cov=rag   # unit tests
uv run pytest -m integration                   # needs compose DB
uv run uvicorn rag.api.app:main --factory --port 8000   # API for the UI (DB + index needed)
scripts/publish_index.sh "$TARGET_DATABASE_URL"   # copy local index to a remote Postgres
cd web && npm install && npm run dev           # glass-box UI, http://localhost:3000
cd web && npm test && npm run build            # API-client tests + production build
```
Long LLM or ingest jobs run in the background with output to `data/*.log`; never poll with sleep.

## Layout
- `src/rag/config.py`: `Settings`, env vars prefixed `RAG_` (pydantic-settings, `.env`);
  `pipeline_fingerprint()` = backend:model:dim:max_chars:loaderN
- `src/rag/db.py` + `schema.sql`: raw psycopg; `{embedding_dim}` templated, literal braces `{{}}`
- `src/rag/ingest/`: `loaders.py` (Hugo Markdown → `Document`; frontmatter, shortcodes,
  HTML→text, heading anchors, `generated` flag), `chunking.py` (heading-aware →
  `Chunk(ord, heading, content)`), `pipeline.py` (incremental; skip when
  `(content_hash, fingerprint)` matches; one transaction per doc; prunes missing docs)
- `src/rag/embed/`: `Embedder` protocol (`embed` for passages, `embed_query` for queries —
  never mix them); `make_embedder`: `local` default (bge-small via fastembed, 384d, no key),
  `openai` (needs `OPENAI_API_KEY`, dim 1536), `fake` (tests). No Anthropic embeddings exist.
- `src/rag/retrieval/`: `search.py` (dense via pgvector with ef_search + iterative scan;
  keyword via FTS with OR semantics), `fusion.py` (`rrf`, `cap_per_source`), `rerank.py`
  (cross-encoder, `RAG_RERANKER_MODEL`), `rewrite.py` (`QueryRewriter`, search-only, cached),
  `retriever.py` (`Retriever` + `RetrieverConfig`: mode, k, candidates, include_generated,
  rrf_k, rerank, max_per_source, rewrite — every field is an ablation switch)
- `src/rag/generate/`: `prompt.py` (numbered passages + rules, exact `REFUSAL` sentence),
  `citations.py` (parse/validate `[n]`), `generator.py` (`RAG_LLM_BACKEND`: `anthropic` →
  `ClaudeGenerator`, `openrouter` → `OpenAICompatibleGenerator`, `fake`; request timeout
  `RAG_LLM_TIMEOUT_S`), `answer.py` (`answer_question -> Answer`)
- `src/rag/agent/`: `loop.py` (`run_agent`: turns, global citation registry, limits), `tools.py`
  (`search_docs` schema and argument checks), `prompt.py` (agent rules, forced final turn)
- `src/rag/evals/`: `metrics.py` (hit@k, recall@k, MRR, nDCG: page relevance, chunk ranks),
  `golden.py` (JSONL loader, unknown-source check), `runner.py` (`ABLATIONS`, `evaluate`,
  tables, `breakdown_markdown` by origin), `drafting.py` (sampling, drafts, `with_retry`),
  `review.py` (resumable review ledger), `judge.py` (LLM judge prompt + tolerant parsing,
  citation stats, unanswerable set), `answers.py` (`evaluate_answers`, `RetryingGenerator`)
- `src/rag/cli.py`: typer app `rag`: init-db, chunk, ingest, search, ask, eval, eval-answers,
  draft-questions, review-drafts. `ask_config()` holds the measured `rag ask` defaults.
- `evals/`: `golden.jsonl` (94: 25 human, 69 reviewed LLM drafts tagged `reviewer:…`),
  `unanswerable.jsonl` (15), `drafts*.jsonl/.txt` (audit trail); `evals/results/` gitignored
- `src/rag/api/`: `app.py` (`create_app(settings, services)`; `/health`, `/ready`,
  `/api/collection`, `POST /api/ask` → SSE `stage`/`context`/`token`/`done`/`error`;
  `run_ask` always releases its concurrency slot), `guards.py` (`RateLimiter`, `DailyBudget`).
  Uses `Retriever.retrieve(..., on_stage=)` and `generate.answer.stream_answer`.
- `web/`: Next.js 16 / React 19 UI ported from the Open Design prototype. `app/globals.css` is
  the design's CSS verbatim (tokens on `:root`, dark default); `components/GlassBox.tsx` holds
  a mutable view-model + actions and maps SSE events onto it; `Panels.tsx` / `Overlays.tsx`
  render; `lib/api.ts` is the API client; `app/privacy`, `app/legal`; `next.config.ts` sets CSP.
- Deploy: `Dockerfile` (models baked in, `--no-access-log` so IPs never hit logs),
  `railway.json`, `scripts/publish_index.sh`. Runbook in README "Deploy".
- Integration tests use a separate `rag_test` DB created by `tests/conftest.py` (dim 8);
  they never touch the real index and skip if Postgres is down.

## Skills (invoke the matching one before starting; don't skip)
- New feature or behaviour change: `superpowers:brainstorming` first, then
  `superpowers:writing-plans` for anything multi-step.
- Writing code: `superpowers:test-driven-development` (the TDD rule below); keep diffs minimal
  (`ponytail`).
- Bug, failing test or surprising eval number: `superpowers:systematic-debugging` before any fix.
- User input, uploads, URL fetching (Phase 5 loaders, API): `security-and-hardening`.
- UI work (Phase 5): `ui-ux-pro-max`, then `web-design-guidelines` to check it.
- Latency work (reranker, ingest, API): `performance-optimization`; measure before and after.
- Logging, progress events, tracing: `observability-and-instrumentation`.
- A real design choice: `documentation-and-adrs` → entry in `docs/decisions.md`.
- Commits and branches: `git-workflow-and-versioning`.
- After each task, before committing: `ponytail:ponytail-review` or `/code-review` on the diff.
- Before saying "done": `superpowers:verification-before-completion` (run tests, show output).
- Codebase questions: `graphify` query first when `graphify-out/graph.json` exists.

## Conventions
- TDD: write the failing test first, run it, then implement. Coverage ≥ 80%.
- Tests in `tests/test_<module>.py`, Arrange-Act-Assert; `CliRunner` + `monkeypatch` for
  outside dependencies; DB tests carry `@pytest.mark.integration`.
- Frozen dataclasses for records; pure functions where possible; no ORM, SQL lives in `.sql`
  or near the query.
- External services (LLMs, embedders, rerankers) sit behind a `Protocol` with a deterministic
  fake for tests. Unit tests never hit the network; CI needs no keys.
- Library errors: `ValueError` naming the bad value or path. CLI errors: catch the specific
  exception, red message + actionable hint, `typer.Exit(1)` (`llm_error_hint` for LLM errors).
- Batch LLM jobs: request timeout, retry with backoff, write each result as it's produced,
  resume by id.
- Commands kebab-case (`init-db`), modules snake_case, line length 100 (ruff,
  `known-first-party = ["rag"]`).
- Conventional commits (`feat:`, `fix:`, `docs:`, `ci:`…).

## Non-negotiables
- Never commit `data/`, `.env`, API keys or eval result dumps (`evals/results/`). Never read
  `.env` directly or print keys; check them via `Settings` (SecretStr).
- Never hand-tune eval numbers; README and docs numbers come from `rag eval` /
  `rag eval-answers` output, with corpus ref, fingerprint and results file recorded.
- Always report eval results by question origin (human vs LLM-drafted): LLM questions score
  ~0.15 higher and can hide regressions.
- Golden entries change only after reading the source passage; record reviewer and reason.
- Retrieval changes must be measurable: add or keep a `RetrieverConfig` switch and an
  `ABLATIONS` row. Defaults change only on n=94-scale evidence, not anecdotes.
- Keep LLM wrappers thin; no LangChain/LlamaIndex.

## Gotchas
- Python pinned to 3.12 (`<3.13`) for ML wheels; system Python is 3.14.
- DB on port 5433, not 5432.
- Bumping cleaning output? Bump `LOADER_VERSION` in `ingest/loaders.py`, or incremental ingest
  skips every doc (raw hashes unchanged) and keeps stale vectors.
- ONNX can oversubscribe the CPU (load average 40+): set `RAG_EMBEDDING_THREADS`.
- OpenRouter free models: shared upstream pools return 429; the account-wide daily cap is
  50 requests without credits, 1,000 with credits. Nemotron 3 Super is the working default.
- Reasoning judge models need a large output budget (`--judge-max-tokens`, default 8192) or their JSON is truncated.
- LLM judges sometimes emit invalid JSON: `parse_judgement` re-asks, then repairs, then
  validates shape; always report judge failures.
- API guards (rate limit, daily budget, concurrency) are in process memory: run one replica.
- Never log question text or client IPs (privacy page promises it).
- Ruff caches results: lint with `--no-cache` before pushing (CI has no cache).
- The repo sits inside the parent `Project/` folder but has its own git remote
  (`phuong-le-tech/rag-pipeline`); run git commands from this folder.
- `git commit` hooks reject `-n`-style flags (including `bash -n`/`[ -n` in the same command);
  never bypass hooks.
