from collections.abc import Sequence
from typing import Any

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 128
MAX_RETRIES = 5


class OpenAIEmbedder:
    """OpenAI embeddings in batches. Retries with backoff come from the SDK (``max_retries``).

    ``client`` is injectable so tests can pass a stub; by default the SDK reads
    ``OPENAI_API_KEY`` from the environment.
    """

    def __init__(
        self,
        dim: int,
        model: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        client: Any = None,
    ) -> None:
        if dim <= 0 or batch_size <= 0:
            raise ValueError(f"dim and batch_size must be positive, got {dim}, {batch_size}")
        self.dim = dim
        self.model = model
        self.batch_size = batch_size
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(max_retries=MAX_RETRIES)
        return self._client

    def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model, input=list(batch), dimensions=self.dim
        )
        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        for vector in vectors:
            if len(vector) != self.dim:
                raise ValueError(f"{self.model} returned {len(vector)} dims, expected {self.dim}")
        return vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        for i, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"cannot embed blank text at index {i}")
        return [
            vector
            for start in range(0, len(texts), self.batch_size)
            for vector in self._embed_batch(texts[start : start + self.batch_size])
        ]
