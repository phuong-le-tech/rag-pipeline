# 2. Ingestion: loading, cleaning, chunking

## Concept

Retrieval can only find what ingestion put in the index, in the shape ingestion gave it.
Garbage in, garbage retrieved. Ingestion has three jobs:

1. **Load**: read source files and extract metadata (title, URL, section, version).
2. **Clean**: remove text that carries no meaning for a reader (markup, template tags,
   navigation), because it pollutes both embeddings and keyword search.
3. **Chunk**: cut documents into passages small enough to be precise, large enough to be
   self-contained.

### Why chunk at all?

- Embedding models have input limits, and one vector for a 5,000-word page averages away the
  details: it is "about Deployments" but not about "rollback" specifically.
- The LLM context is limited and costs money per token; you want to send the 5 best
  passages, not 5 whole pages.
- Citations become precise: `[2]` points to a paragraph, not a page.

### Chunking strategies

| Strategy | How | Pros | Cons |
|---|---|---|---|
| Fixed size | every N characters/tokens, maybe with overlap | trivial | cuts mid-sentence, mixes topics |
| Recursive | split on `\n\n`, then `\n`, then `. `… until small enough | respects paragraphs | ignores document structure |
| **Structure-aware** | split on headings, then pack paragraphs | chunks stay inside one topic; heading = free context | needs structured input (Markdown, HTML) |
| Semantic | split where embedding similarity between sentences drops | topic-aware on unstructured text | slow, needs embeddings at ingest, unstable |
| Parent/child | index small chunks, return their larger parent | precise match + rich context | more storage, more logic |

**Chunk size** is a real trade-off: small chunks match precisely but lose context ("it" no
longer says what "it" is); big chunks keep context but dilute the embedding and waste prompt
tokens. Typical range: 200–1,000 tokens. We start at 1,500 characters (~350 tokens) and will
*measure* alternatives in Phase 4 instead of guessing.

**Overlap** (repeating the end of one chunk at the start of the next) helps fixed-size
chunking. Structure-aware chunking needs it less because cuts fall on paragraph boundaries.

## In this repo

### Loader: [`src/rag/ingest/loaders.py`](../src/rag/ingest/loaders.py)

Kubernetes docs are written for Hugo, a static site generator. A raw page looks like:

```markdown
---
title: Deployments
weight: 10
---
<!-- overview -->
A _Deployment_ provides declarative updates for Pods.
{{< feature-state for_k8s_version="v1.19" state="stable" >}}
{{< note >}}
Do not manage ReplicaSets owned by a Deployment.
{{< /note >}}
```

The loader turns that into a `Document`:

| Field | Value | Why we keep it |
|---|---|---|
| `source` | `concepts/workloads/deployment.md` | stable ID for re-ingestion |
| `url` | `https://kubernetes.io/docs/concepts/workloads/deployment/` | clickable citations |
| `title` | `Deployments` | prepended as `# Deployments` so every chunk's heading path starts with it |
| `section` | `concepts` | future metadata filter ("only tasks pages") |
| `content_hash` | sha256 of the raw file | skip unchanged files on re-ingest |

Cleaning rules (`clean_hugo`):

- HTML comments → removed.
- `note` / `caution` / `warning` blocks → content kept, labelled `**Note:** …`. Warnings
  are often exactly what users ask about.
- Every other shortcode (`feature-state`, `code_sample`, `glossary_definition`…) → removed.
  They are template calls, not prose. Known loss: `code_sample` pulls in YAML examples we
  currently drop. Logged in the [decision log](decisions.md).
- Drafts and pages that are empty after cleaning → skipped.
- HTML (`html_to_text`) → readable text. Table rows become lines with cells joined by ` | `,
  `<code>`/`<tt>` become backticks, block tags become line breaks, inline tags are dropped,
  entities are decoded (`&amp;` → `&`). Only *known* tag names are touched, so placeholders
  such as `<pod-name>` in commands survive.
- Every document gets `generated = True` when its frontmatter says `auto_generated: true`
  **or** it sits in a `generated/` folder (the frontmatter flag alone misses kubeadm's pages).

### What the real corpus looked like (profiled 2026-10-10, ref `b2e60d8`)

Profiling before indexing paid off immediately:

| | Docs | Chunks | Median chunk | Chunks < 100 chars |
|---|---|---|---|---|
| Hand-written | 1,236 | 8,072 | 658 chars | 397 |
| Generated reference | 416 | 9,791 | 432 chars | 3,066 |
| **All** | 1,652 | 17,863 | 581 chars | 3,463 |

- 70% of the *files* sit under `reference/`, and generated pages produce **55% of all chunks**
  from 25% of the documents. Left unchecked they would crowd search results with flag tables.
- Before HTML cleanup the corpus contained ~55,000 `<td>` cells. Embedding that markup would
  have made vectors partly "about HTML".
- Generated pages produce most of the tiny chunks: API reference pages put a heading on every
  field, so sections are one line long.

We **keep** generated pages but tag them, so Phase 4 can measure retrieval with and without
them. See the [decision log](decisions.md). The lesson generalises: always profile a corpus
(counts, length distribution, leftover markup) before you embed it.

### Chunker: [`src/rag/ingest/chunking.py`](../src/rag/ingest/chunking.py)

Structure-aware, in two steps:

1. **Split into sections** at Markdown headings, tracking the heading *path*:
   `# Deployments` → `## Updating` → `### Rolling back` gives
   `Deployments > Updating > Rolling back`. A sibling heading replaces its branch
   (`## Scaling` after `### Rolling back` gives `Deployments > Scaling`). `#` lines inside
   code fences are ignored. In bash they are comments, not headings.
2. **Pack paragraphs** into chunks up to `max_chars`, never splitting a paragraph unless it is
   longer than `max_chars` on its own (then it is hard-split).

Each chunk is `Chunk(ord, heading, content)`. The heading path is stored in its own column and
weighted higher in keyword search (chapter 3), so a query like "rolling back" matches the
section title even if the body says "undo".

## Trade-offs

- Heading split assumes well-structured Markdown. True for k8s docs, false for scanned PDFs.
- Hard-splitting oversized paragraphs (often big code blocks or tables) can cut mid-line. Rare
  in this corpus; revisit if eval failures point here.
- Characters, not tokens: simpler and model-independent, slightly imprecise budget.

## Try it

```bash
uv run rag chunk README.md --preview
RAG_CHUNK_MAX_CHARS=400 uv run rag chunk README.md --preview   # smaller chunks
```

Once the corpus is fetched:

```bash
uv run rag chunk data/k8s-docs/concepts/workloads/controllers/deployment.md --preview
```

Questions to ask yourself while reading the output:

- Could you answer "how do I roll back a Deployment?" from **one** chunk alone?
- Does any chunk start with "it" or "this" and lose its meaning out of context?
- Is there leftover markup noise?
