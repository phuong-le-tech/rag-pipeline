"""One entry point for every retrieval strategy, selected by ``RetrieverConfig``.

Each config field is a switch the eval harness can flip, so every retrieval choice shows up as
a row in the ablation table.
"""

from dataclasses import dataclass, replace
from typing import Literal, get_args

import psycopg

from rag.embed import Embedder
from rag.retrieval.fusion import DEFAULT_K, rrf
from rag.retrieval.search import Hit, dense_search, keyword_search

Mode = Literal["dense", "keyword", "hybrid"]


@dataclass(frozen=True)
class RetrieverConfig:
    mode: Mode = "hybrid"
    k: int = 10  # results returned
    candidates: int = 50  # results fetched from each first-stage search
    include_generated: bool = True
    rrf_k: int = DEFAULT_K

    def __post_init__(self) -> None:
        if self.mode not in get_args(Mode):
            raise ValueError(f"mode must be one of {get_args(Mode)}, got {self.mode!r}")
        if self.k <= 0 or self.rrf_k <= 0:
            raise ValueError(f"k and rrf_k must be positive, got {self.k}, {self.rrf_k}")
        if self.candidates < self.k:
            raise ValueError(f"candidates ({self.candidates}) must be >= k ({self.k})")


class Retriever:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder) -> None:
        self.conn = conn
        self.embedder = embedder

    def _ranked_lists(self, query: str, config: RetrieverConfig) -> list[list[Hit]]:
        lists = []
        if config.mode in ("dense", "hybrid"):
            vector = self.embedder.embed_query(query)
            lists.append(
                dense_search(self.conn, vector, config.candidates, config.include_generated)
            )
        if config.mode in ("keyword", "hybrid"):
            lists.append(
                keyword_search(self.conn, query, config.candidates, config.include_generated)
            )
        return lists

    def retrieve(self, query: str, config: RetrieverConfig) -> list[Hit]:
        if not query.strip():
            raise ValueError("query must not be blank")
        lists = self._ranked_lists(query, config)
        if config.mode != "hybrid":
            return lists[0][: config.k]
        by_id = {hit.chunk_id: hit for hits in lists for hit in hits}
        fused = rrf([[h.chunk_id for h in hits] for hits in lists], config.rrf_k, config.k)
        return [replace(by_id[chunk_id], score=score) for chunk_id, score in fused]
