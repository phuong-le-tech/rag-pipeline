"""Second-stage reranking with a cross-encoder.

An embedding model encodes query and chunk *separately* and compares vectors: fast, but the
two texts never "see" each other. A cross-encoder reads ``(query, chunk)`` together in one pass
and outputs a relevance score: far more accurate, far too slow for 18,000 chunks, fine for 50.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol

from rag.config import Settings
from rag.retrieval.search import Hit

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 32


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...


def rerank_text(hit: Hit) -> str:
    return f"{hit.heading}\n\n{hit.content}" if hit.heading else hit.content


class CrossEncoderReranker:
    """Local cross-encoder via fastembed (ONNX). ``model`` is injectable for tests."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        threads: int | None = None,
        model: Any = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.threads = threads
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name, threads=self.threads)
        return self._model

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.rerank(
            query, [rerank_text(h) for h in hits], batch_size=self.batch_size
        )
        rescored = [replace(h, score=float(s)) for h, s in zip(hits, scores, strict=True)]
        return sorted(rescored, key=lambda h: -h.score)


def make_reranker(settings: Settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        model_name=settings.reranker_model, threads=settings.embedding_threads
    )
