# rag-pipeline — instructions for Claude Code

Portfolio RAG system over the Kubernetes docs: hybrid retrieval (pgvector + Postgres FTS,
RRF fusion, cross-encoder rerank), cited answers with refusal, and an eval harness whose
ablation table is the README centrepiece.

## Before any work
- Read `.claude/plans/rag-pipeline.plan.md`: phases, decisions, risks. Tick boxes as tasks land.
- Work the current phase in order; do not start a later phase without being asked.

## Docs are a deliverable (the owner is learning RAG through this project)
- `docs/` teaches the project: one chapter per pipeline stage, each with
  **Concept → In this repo (code links) → Trade-offs → Try it**. Index: `docs/README.md`.
- Every change that adds or alters behaviour updates docs **in the same commit**: the relevant
  chapter, the status table in `docs/README.md`, new terms in `docs/glossary.md`, and any real
  choice in `docs/decisions.md` (newest first, context / decision / consequence).
- Starting a new phase means writing its chapter (e.g. `04-retrieval.md` for Phase 2).
- Explain the *why* for a learner: plain language, small examples, no copy-pasted API docs.
- When eval numbers exist, docs quote them from `rag eval` output, never invented.

## Commands
```bash
uv sync                                   # Python 3.12 venv + deps
docker compose up -d                      # Postgres 17 + pgvector on localhost:5433
uv run rag init-db                        # schema (idempotent)
scripts/fetch_corpus.sh [ref]             # k8s docs -> data/k8s-docs (+ CORPUS_REF sha)
uv run rag chunk data/k8s-docs --preview  # inspect cleaning + chunking, no DB
RAG_EMBEDDING_THREADS=6 uv run rag ingest data/k8s-docs   # incremental; ALWAYS the corpus root
                                          # (first full run ~35-60 min on CPU, re-runs seconds)
uv run ruff format . && uv run ruff check .
uv run pytest -m "not integration" --cov=rag   # unit tests
uv run pytest -m integration                   # needs compose DB
```

## Layout
- `src/rag/config.py`: `Settings`, env vars prefixed `RAG_` (pydantic-settings, `.env` supported)
- `src/rag/db.py` + `schema.sql`: raw psycopg; `{embedding_dim}` templated, literal braces doubled `{{}}`
- `src/rag/ingest/loaders.py`: Hugo Markdown → `Document` (frontmatter, shortcode cleanup, URL)
- `src/rag/ingest/chunking.py`: heading-aware chunker → `Chunk(ord, heading, content)`
- `src/rag/embed/`: `Embedder` protocol (`embed` for passages, `embed_query` for queries —
  never mix them), `make_embedder(settings)` picks `RAG_EMBEDDING_BACKEND`: `local` default
  (`LocalEmbedder`, bge-small via fastembed, 384d, no key), `openai` (needs `OPENAI_API_KEY`,
  set dim 1536), `fake` (tests only). No Anthropic embeddings exist.
- `src/rag/ingest/pipeline.py`: incremental ingest; skip when `(content_hash, fingerprint)`
  matches; one transaction per doc; prunes docs missing from disk
- `src/rag/retrieval/`: `search.py` (dense via pgvector, keyword via FTS with OR semantics),
  `fusion.py` (`rrf`), `rerank.py` (cross-encoder, `RAG_RERANKER_MODEL`, MiniLM default),
  `retriever.py` (`Retriever` + `RetrieverConfig` switches = ablation rows, incl. `rerank`)
- `src/rag/cli.py`: typer app, entry point `rag` (`init-db`, `chunk`, `ingest`, `search`)
- **Changing cleaning output? Bump `LOADER_VERSION` in `ingest/loaders.py`**, or incremental
  ingest will skip every doc (raw-file hashes don't change) and keep stale vectors.
- Integration tests use a separate `rag_test` DB created by `tests/conftest.py` (dim 8);
  they never touch the real index and skip if Postgres is down
- Planned: `retrieval/`, `generate/`, `evals/` (see plan)

## Conventions
- TDD: write the failing test first, run it, then implement. Coverage ≥ 80%.
- Tests in `tests/test_<module>.py`, Arrange-Act-Assert; `CliRunner` + `monkeypatch` for
  outside dependencies; DB tests carry `@pytest.mark.integration`.
- Frozen dataclasses for records; pure functions where possible; no ORM, SQL lives in `.sql`
  or near the query.
- External services (OpenAI, Anthropic, rerankers) sit behind a `Protocol` with a
  deterministic fake for tests. Unit tests never hit the network.
- Library errors: `ValueError` naming the bad value or path. CLI errors: catch the specific
  exception, red message + fix hint, `typer.Exit(1)` (see `init_db_cmd`).
- Commands kebab-case (`init-db`), modules snake_case, line length 100 (ruff).
- Conventional commits (`feat:`, `fix:`, `chore:`…).

## Non-negotiables
- Never commit `data/`, `.env`, API keys or eval result dumps (`evals/results/`).
- Never hand-tune eval numbers; every number in the README comes from `rag eval` output
  with the corpus `CORPUS_REF` recorded.
- Retrieval changes must be measurable: add or keep a `RetrieverConfig` switch so the
  ablation table can compare it.
- Keep the LLM wrapper thin; no LangChain/LlamaIndex.

## Gotchas
- The project pins Python 3.12 (`<3.13`) for ML wheel availability; system Python is 3.14.
- DB listens on port 5433, not 5432, to avoid clashing with a local Postgres.
- This repo sits inside the parent `Project/` git repo; run git commands from this folder.
- `git commit` hooks reject `-n`-style flags; never bypass hooks.
