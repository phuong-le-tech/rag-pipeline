from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Turns text into fixed-length vectors.

    ``embed`` is for passages being indexed; ``embed_query`` is for search queries. Some models
    (bge, e5) are asymmetric and encode the two differently, so callers must pick the right one.
    """

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
