# Decision log

Newest first. Each entry: context, decision, consequences. Revisit when evals say otherwise.

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
