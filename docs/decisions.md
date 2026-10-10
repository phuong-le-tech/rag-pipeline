# Decision log

Newest first. Each entry: context, decision, consequences. Revisit when evals say otherwise.

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
