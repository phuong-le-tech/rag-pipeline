import hashlib
import math
import random
from collections.abc import Sequence


class FakeEmbedder:
    """Deterministic unit vectors seeded by the text's hash. For tests: no network, no cost.

    Identical texts get identical vectors; different texts get unrelated ones, so it checks
    plumbing (storage, ordering, SQL) but says nothing about retrieval quality.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        raw = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        length = math.sqrt(sum(x * x for x in raw))
        return [x / length for x in raw]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]
