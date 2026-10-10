from collections.abc import Sequence
from typing import Any

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH_SIZE = 64


class LocalEmbedder:
    """On-device embeddings via fastembed (ONNX runtime, no torch, no API key).

    bge models are asymmetric: ``passage_embed`` encodes documents, ``query_embed`` adds the
    query instruction. Weights download once (~67 MB for bge-small) into fastembed's cache.
    ``model`` is injectable so tests can pass a stub.
    """

    def __init__(
        self,
        dim: int,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: Any = None,
    ) -> None:
        if dim <= 0 or batch_size <= 0:
            raise ValueError(f"dim and batch_size must be positive, got {dim}, {batch_size}")
        self.dim = dim
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _to_list(self, vector: Any) -> list[float]:
        if len(vector) != self.dim:
            raise ValueError(f"{self.model_name} returned {len(vector)} dims, expected {self.dim}")
        return [float(x) for x in vector]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        for i, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"cannot embed blank text at index {i}")
        if not texts:
            return []
        vectors = self.model.passage_embed(list(texts), batch_size=self.batch_size)
        return [self._to_list(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("cannot embed a blank query")
        return self._to_list(next(iter(self.model.query_embed(text))))
