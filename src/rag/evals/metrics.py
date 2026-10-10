"""Retrieval metrics over a ranked list of chunks, each represented by its source page.

Relevance is judged per page (the golden set lists pages, which stay stable when chunking
changes); ranks count chunks, because chunks are what the LLM is shown. When several chunks
come from the same relevant page, only the first counts as a new find.
"""

import math
from collections.abc import Iterable, Sequence


def _check(relevant: set[str], k: int | None = None) -> None:
    if not relevant:
        raise ValueError("relevant must contain at least one source")
    if k is not None and k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant page appears in the top k chunks, else 0.0."""
    _check(relevant, k)
    return 1.0 if any(source in relevant for source in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of distinct relevant pages found in the top k chunks."""
    _check(relevant, k)
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first chunk from a relevant page (0.0 if none). Averaged: MRR."""
    _check(relevant)
    for rank, source in enumerate(ranked, start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance, first find per page only."""
    _check(relevant, k)
    found: set[str] = set()
    dcg = 0.0
    for rank, source in enumerate(ranked[:k], start=1):
        if source in relevant and source not in found:
            found.add(source)
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
