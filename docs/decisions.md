# Decision log

Newest first. Each entry: context, decision, consequences. Revisit when evals say otherwise.

## 2026-10-13: Uploads are capability-id collections that expire after 24 h
**Context.** Visitors should ask questions over their own Markdown/text without accounts, on a
public demo with a small CPU box and a privacy page that promises not to keep questions.
**Decision.** `POST /api/collections` creates a collection named by `uuid4().hex`; knowing the id is
the only access control. Limits: 1 MB, UTF-8, 300 chunks, shared rate limit and slots. Expiry
is `RAG_UPLOAD_TTL_HOURS` (24), enforced lazily at the start of each upload instead of by a
scheduler. Multipart is parsed with the stdlib to avoid a new dependency.
**Consequences.** Anyone who learns an id can read that upload (acceptable for 24 h demo data; ids
are never logged). Expired rows linger until the next upload happens, so retention is "at most 24 h
plus until the next upload"; the privacy page says "up to 24 hours". Embedding runs in the request
thread, so a burst of uploads is bounded by the concurrency slots, not by a queue.

## 2026-10-13: Uploads get their own `collection`, not a second table or database
**Context.** Phase 5 uploads must not leak into Kubernetes answers, evals or `rag ingest` pruning
(which deletes every document missing from the run).
**Decision.** `documents.collection` (default `'k8s'`). Searches filter on it
(`RetrieverConfig.collection`), `ingest` reads and prunes only its own collection, `/api/collection`
counts only `k8s`, and `publish_index.sh` deletes only `k8s` rows instead of truncating.
**Consequences.** Default behaviour is unchanged (same Hit@5 / MRR). A filtered HNSW scan can return
fewer rows than asked, so iterative scan stays on and an integration test checks it. `source` is
still globally unique: upload sources must be namespaced. `publish_index.sh` copies explicit ids, so
publishing into a target that already holds uploads can collide on ids (not handled yet).
On an existing DB the new column has no planner statistics until autovacuum or
`ANALYZE documents`; before that the planner may skip HNSW for an exact scan (Hit@5 moved
0.894 → 0.904 until we analysed).

## 2026-10-13: Agent mode stays opt-in; answer judge is a free OpenRouter model
- **Context**: first measured comparison of pipeline and agent on the same 94 answerable and 15
  unanswerable questions, same retrieval and answer model. The owner runs no paid Claude calls, so
  the Sonnet judge used for the earlier answer table is no longer an option.
- **Decision**: the pipeline stays the default everywhere; `rag ask --agent` and
  `eval-answers --mode agent` stay opt-in. The judge is `dots-studio/dots-3-note-preview:free` on
  OpenRouter, with its own output budget (`--judge-max-tokens`, default 8192), from a different
  family than the answer model.
- **Consequence**: the agent recovers refusals (false refusals 7 to 1 of 94; 0 of 25 hand-written)
  and raises correct-yes from 0.798 to 0.844 with faithfulness equal (0.886 vs 0.882). The gain is
  about 5 points on 84 to 90 graded answers, not tested for significance. It costs about 2.9x the
  tokens, 2.3x the median latency (11.2 s vs 5.0 s), and half the citation coverage (0.222 vs
  0.458); 10 of 109 runs hit the 4-step limit. Judged numbers are only comparable within one judge,
  and the free judge is not Sonnet: judged tables from different judges must not be mixed.

## 2026-10-12: Agent mode is opt-in; hand-written loop; 4-step cap; separate agent model
- **Context**: the pipeline does one search per question, so it cannot retry a bad query or
  split a comparison into parts. We want to measure an agent against the pipeline before any
  change to defaults.
- **Decision**: `rag ask --agent` only; the pipeline stays the default everywhere. The loop is
  hand-written (no agent framework) so each limit and the citation registry are plain code.
  Hard limits: 4 searches (`RAG_AGENT_MAX_STEPS`), 90 s checked between turns, 30,000 tokens,
  then one forced final turn with tools off. The agent has its own `RAG_AGENT_BACKEND` and
  `RAG_AGENT_MODEL`, falling back to the LLM settings, so the model can differ from the answer model.
  The final answer is not streamed in v1.
- **Consequence**: more model calls and latency per question; behaviour varies run to run and must
  be measured in Phase 2 before it becomes a default. Tool-calling quality depends on the free
  model. Smoke test 2026-10-12: the free Nemotron 3 Super model on OpenRouter makes valid tool
  calls (1 search, 2 LLM calls, ~2,550 tokens on a Deployment vs StatefulSet question).

## 2026-10-12: Production v1 on Railway + Vercel, Kubernetes docs only
- **Context**: the owner wants people to test the glass-box demo; it must be safe, cheap and
  honest about data. Uploads would add storage, cleanup and abuse surface.
- **Decision**: FastAPI service streaming each stage over SSE; guards for input size, rate (10/min
  per IP), daily LLM calls (300), concurrency (2) and CORS; nothing stored, no question text or IP
  in logs; privacy and legal pages with CC BY 4.0 attribution; index copied to Railway Postgres
  with `pg_dump`; models baked into the image. Uploads stay off for v1.
- **Consequence**: one replica only (guards in memory); ~$5/month for Railway; the free LLM's
  upstream limits can still fail an answer, which the UI reports with a retry.

## 2026-10-11: Free OpenRouter model is the default LLM backend
- **Context**: Claude Sonnet costs too much for everyday use and demos.
- **Decision**: `RAG_LLM_BACKEND` defaults to `openrouter` with
  `nvidia/nemotron-3-super-120b-a12b:free`; `anthropic` stays selectable per run.
- **Consequence**: $0 per answer, but rate limits (1,000 requests/day with credits) and possible
  prompt logging. Phase 4 numbers already came from Nemotron answers, so defaults stay valid;
  the Sonnet judge for `eval-answers` is now opt-in via `--judge-backend anthropic`.

## 2026-10-11: Phase 5 UI in Next.js; Markdown uploads first
- **Context**: the glass-box demo is shown in interviews, so it must look polished; every new
  input format needs its own loader.
- **Decision**: Next.js frontend (design generated with Open Design) talking to a FastAPI
  backend that streams pipeline progress over server-sent events. First version accepts
  Markdown only, which the current loader already handles; PDF, URL and pasted text come later.
- **Consequence**: two apps and two languages to build and deploy, in exchange for a stronger
  UI. Markdown-only keeps the first slice about streaming and visibility, not parsing.

## 2026-10-11: No reranker-score refusal threshold; Sonnet judge with repair fallback
- **Context**: answer evals (Nemotron answers, Sonnet judge). The model refused 14/15
  unanswerable questions on its own; false-premise questions get *high* reranker scores.
- **Decision**: no score threshold (it would wrongly refuse 2-7 answerable questions to catch
  few extra). Judge verdicts: strict JSON, re-ask up to 3 times, then `json-repair` fallback,
  then shape validation; judge failures always reported.
- **Consequence**: 0 judge failures in the final run (13 on the first). Structured output via
  tool calling is the cleaner long-term fix.

## 2026-10-11: `rag ask` defaults from the 94-question evaluation
- **Context**: v2 golden set (25 human + 69 reviewed LLM drafts), all 10 configs, results split
  by question origin.
- **Decision**: default = hybrid + rerank + `max_per_source=1`, generated pages included.
  Query rewriting is opt-in (`--rewrite`).
- **Consequence**: Hit@5 0.915 / MRR 0.749 overall. Rewriting would add +0.01 overall
  (+0.08 on hand-written, −0.02 on LLM-drafted questions) for one LLM call per query: not worth
  it by default. The 25-question estimate for rewriting (0.96) was inflated by question style,
  which is the lesson behind always reporting results by origin.

## 2026-10-11: Golden set v2 reviewed by Claude, spot-checked by the owner
- **Context**: 97 LLM drafts needed review; the owner asked Claude to do it.
- **Decision**: Claude reviewed each draft against its source passage text (not memory), tagged
  accepted rows `reviewer:claude`, logged every reason in `evals/drafts-review-notes.jsonl`;
  the owner spot-checks a sample. Query-parameter questions on generated API pages were
  rejected because ~160 pages share identical text (ambiguous relevance).
- **Consequence**: 94 questions, reviewer recorded per row, so results can be re-split by
  reviewer if the spot-check finds problems.

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
  silently keeps stale vectors.

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
- **Context**: re-embedding the whole corpus takes 35-60 min on CPU; upstream changes few pages.
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
