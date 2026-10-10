"""Reciprocal Rank Fusion (Cormack et al., 2009).

Each ranked list votes ``1 / (k + rank)`` for every item it contains; votes are summed.
Only ranks are used, never raw scores, so lists whose scores live on different scales
(cosine similarity vs. ts_rank) combine without any calibration. ``k`` damps the advantage of
the very top ranks; 60 is the value from the original paper and a robust default.
"""

from collections.abc import Callable, Hashable, Sequence

DEFAULT_K = 60


def rrf[T: Hashable](
    rankings: Sequence[Sequence[T]], k: int = DEFAULT_K, limit: int | None = None
) -> list[tuple[T, float]]:
    """Fuse ranked lists into one, best first. Ties keep first-appearance order."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    scores: dict[T, float] = {}
    for ranking in rankings:
        seen: set[T] = set()
        for position, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    fused = sorted(scores.items(), key=lambda pair: -pair[1])
    return fused if limit is None else fused[:limit]


def cap_per_source[T](
    ranked: Sequence[T], key: Callable[[T], Hashable], max_per_source: int | None, limit: int
) -> list[T]:
    """Keep ranking order but allow at most ``max_per_source`` items per source, up to ``limit``.

    Without a cap, one long page can fill the whole top k with near-duplicate chunks and push
    every other relevant page out of the LLM's context.
    """
    if max_per_source is None:
        return list(ranked[:limit])
    counts: dict[Hashable, int] = {}
    kept: list[T] = []
    for item in ranked:
        source = key(item)
        if counts.get(source, 0) < max_per_source:
            counts[source] = counts.get(source, 0) + 1
            kept.append(item)
            if len(kept) == limit:
                break
    return kept
