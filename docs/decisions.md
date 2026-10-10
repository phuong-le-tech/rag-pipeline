# Decision log

Newest first. Each entry: context, decision, consequences. Revisit when evals say otherwise.

## 2026-10-10: Cap chunks per page; rewrite queries for search only
- **Context**: failure analysis showed one page filling the top 5; user wording differs from
  doc wording.
- **Decision**: `RetrieverConfig.max_per_source` (applied after reranking) and
  `RetrieverConfig.rewrite` (LLM rewrite used for first-stage search; reranker keeps the
  original question). Both are ablation rows, not defaults, until measured end to end.
- **Consequence**: max 1/page lifted Hit@5 0.80 → 0.88 at no cost (n = 25); rewrite adds one
  cached LLM call per question, still to be measured. Answer-level evals decide the defaults.

## 2026-10-10: Batch LLM jobs must time out and checkpoint
- **Context**: a drafting run hung on one request (SDK default ~10 min timeout) and lost 57
  in-memory drafts when stopped.
- **Decision**: `RAG_LLM_TIMEOUT_S=60` on both clients; drafts appended and flushed one by one,
  resumable by id; error hints distinguish per-minute and daily rate limits.
- **Consequence**: interruptions cost at most one request; reruns are free for finished work.

## 2026-10-10: Retrieval eval design (page relevance, chunk ranks, pooled corrections)
- **Context**: need defensible numbers for every retrieval choice.
- **Decision**: golden set lists relevant *pages*; metrics count *chunk* ranks with first-find
  only; every run records corpus ref + fingerprint. Golden entries are extended only after
  reading a retrieved page and confirming it answers the question.
- **Consequence**: chunking experiments don't invalidate the golden set. First real result:
  hybrid+rerank Hit@5 0.80, dense ≫ keyword, n = 25 so only large gaps are conclusive.
  Reranking stays on for `rag ask`.

## 2026-10-10: OpenRouter free models as an alternative LLM backend
- **Context**: owner wants to iterate without paying per call.
- **Decision**: add `OpenAICompatibleGenerator` (OpenAI Chat Completions format via the `openai`
  SDK) and `RAG_LLM_BACKEND=openrouter`. Claude stays the default backend. Default free model
  changed the same day from `google/gemma-4-31b-it:free` (HTTP 429, shared upstream pool) to
  `nvidia/nemotron-3-super-120b-a12b:free` (worked). Error hints now distinguish rate limits,
  authentication and network failures.
- **Consequence**: free iteration, subject to rate limits and possible prompt logging (public
  docs, so acceptable). Smaller free models may follow citation rules less reliably; Phase 4
  compares free vs. Claude answer quality instead of assuming.

## 2026-10-10: Grounded prompt with numbered passages, validated citations, exact refusal
- **Context**: answers must be verifiable and must not invent sources or facts.
- **Decision**: numbered context blocks (heading path + URL + text); rules to cite `[n]` per
  factual sentence and to reply with one exact refusal sentence. Code validates citation numbers
  and refuses without an LLM call when retrieval is empty. No score-threshold refusal until
  Phase 4 calibrates one.
- **Consequence**: hallucinated citation numbers are caught mechanically; faithfulness of valid
  citations still needs an LLM judge. Model default `claude-sonnet-5-5`, configurable.

## 2026-10-10: API key via `Settings.anthropic_api_key` (SecretStr)
- **Context**: pydantic-settings reads `.env` but doesn't export it, so the SDK's own
  env-var lookup would miss a key stored in `.env`.
- **Decision**: read `ANTHROPIC_API_KEY` (no `RAG_` prefix) into a `SecretStr`, pass it to the
  client explicitly.
- **Consequence**: `.env` and exported variables both work; the key never prints in reprs.

## 2026-10-10: MiniLM-L-6 cross-encoder as default reranker (not bge-reranker-v2-m3)
- **Context**: the plan named `bge-reranker-v2-m3` (2.3 GB, ~570 MB int8). On this CPU the
  small MiniLM already costs ~3.7 s per query for 50 candidates.
- **Decision**: default `RAG_RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-6-v2` via fastembed (no
  torch); bge-v2-m3 stays one setting away.
- **Consequence**: reranking is usable interactively; Phase 4 compares no-rerank / MiniLM /
  bge-v2-m3 on quality vs. latency before anything is called "better".

## 2026-10-10: Loader version in the pipeline fingerprint
- **Context**: fixing heading anchors changed cleaned text but not raw files, so content hashes
  matched and incremental ingest would have skipped every document.
- **Decision**: `LOADER_VERSION` constant, part of the fingerprint; bump on any cleaning change.
- **Consequence**: cleaning fixes trigger a full re-embed (~1 h on CPU). Forgetting the bump
  silently keeps stale vectors, so it's called out in `CLAUDE.md`.

## 2026-10-10: Keyword search uses OR semantics; RRF k=60 for hybrid
- **Context**: `websearch_to_tsquery` ANDs all words and excludes most chunks for natural
  questions.
- **Decision**: OR the stemmed terms, rank with `ts_rank_cd` (heading weight A > body B). Fuse
  with dense results by RRF (k=60), which needs no score calibration.
- **Consequence**: high recall, noisy precision (first real query: keyword #1 off-topic). Relies
  on the reranker to clean up. Length normalisation is a candidate fix, to be measured.

## 2026-10-10: Raise `hnsw.ef_search`, enable iterative scans
- **Context**: HNSW returns at most `ef_search` (default 40) rows and filters after the search.
- **Decision**: set `ef_search = max(40, candidates)` and `iterative_scan = relaxed_order` per
  search; re-sort by distance in an outer query.
- **Consequence**: asking for 50 candidates returns 50, with or without the generated filter.

## 2026-10-10: Incremental ingestion keyed on content hash + pipeline fingerprint
- **Context**: re-embedding the whole corpus takes 35–60 min on CPU; upstream changes few pages.
- **Decision**: skip a document when `(content_hash, backend:model:dim:max_chars)` matches the
  stored pair; otherwise re-chunk, re-embed and swap its chunks in one transaction. Prune
  documents missing from disk. Embed `heading path + content` (contextual chunk headers).
- **Consequence**: re-runs take seconds; changing model or chunk size safely re-embeds
  everything; no mixed-model vectors. Pruning means `rag ingest` must get the corpus root.

## 2026-10-10: Cap ONNX threads (`RAG_EMBEDDING_THREADS`)
- **Context**: default threading oversubscribed the CPU (load average 46 on 10 cores).
- **Decision**: expose a thread cap; 6 measured fastest on the dev machine (121 vs 204 ms/chunk).
- **Consequence**: one more setting; the best value depends on the machine and its load.

## 2026-10-10: Local embeddings by default (bge-small via fastembed)
- **Context**: no OpenAI key; Anthropic offers no embeddings endpoint.
- **Decision**: default `RAG_EMBEDDING_BACKEND=local` with `BAAI/bge-small-en-v1.5` (384d) on
  fastembed/ONNX (no torch). Keep `OpenAIEmbedder` as an alternative backend.
- **Consequence**: free, private, reproducible, ~5 min to embed the corpus on CPU. Smaller
  model may cost some accuracy, so Phase 4 measures local vs. OpenAI. Switching backend means
  changing `RAG_EMBEDDING_DIM` and re-embedding everything. The interface gained `embed_query`
  because bge encodes queries and passages differently.

## 2026-10-10: Keep generated reference pages, tag them, measure
- **Context**: profiling showed generated reference (CLI flags, kubectl/kubeadm generated, API
  definitions) is 25% of documents but 55% of chunks, mostly HTML tables.
- **Decision**: convert HTML to text, keep every page, set `Document.generated` from
  `auto_generated: true` frontmatter or a `generated/` path segment.
- **Consequence**: flag and API questions stay answerable; Phase 4 adds an ablation row
  "without generated pages" to prove whether they help or crowd results.
- **Open issue**: 3,066 tiny generated chunks (one-line API field sections). Candidate fix:
  merge sections shorter than N chars into their parent. Measure first.

## 2026-10-10: Drop `code_sample` shortcodes during cleaning
- **Context**: k8s pages embed YAML examples via `{{% code_sample file="…" %}}`; the files live
  outside `content/en/docs`.
- **Decision**: remove the shortcode for now.
- **Consequence**: questions answered mainly by a YAML manifest may retrieve poorly. Candidate
  fix: resolve the file and inline it as a fenced code block. Measure first.

## 2026-10-10: Structure-aware chunking, 1,500 characters
- **Context**: corpus is well-structured Markdown.
- **Decision**: split on headings, pack paragraphs up to 1,500 chars, store the heading path.
- **Consequence**: chunks stay on-topic, heading path gives free context and keyword weight.
  Size is a starting point, to be tuned with evals.

## 2026-10-10: Kubernetes docs as corpus
- **Context**: needed a real, messy, sizable corpus with an open licence.
- **Decision**: `kubernetes/website` English docs (CC BY 4.0), commit SHA recorded per fetch.
- **Consequence**: realistic cleaning work (Hugo shortcodes), real user questions, reproducible
  evals.

## 2026-10-10: Postgres + pgvector, no vector DB, no framework
- **Context**: portfolio project; clarity and explainability matter more than scale.
- **Decision**: one Postgres for vectors, full-text and metadata; thin wrappers instead of
  LangChain/LlamaIndex.
- **Consequence**: transactional updates and SQL filters for free; more code written by hand,
  every step visible and testable.

## 2026-10-10: Python 3.12
- **Context**: system Python is 3.14; ML wheels (torch, sentence-transformers) lag new releases.
- **Decision**: pin `>=3.12,<3.13`.
