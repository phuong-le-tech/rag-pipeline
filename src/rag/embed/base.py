from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Turns texts into fixed-length vectors, one per input, in input order."""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
