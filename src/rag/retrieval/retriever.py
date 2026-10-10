"""One entry point for every retrieval strategy, selected by ``RetrieverConfig``.

Each config field is a switch the eval harness can flip, so every retrieval choice shows up as
a row in the ablation table.
"""

from dataclasses import dataclass, replace
from typing import Literal, get_args

import psycopg

from rag.embed import Embedder
from rag.retrieval.fusion import DEFAULT_K, cap_per_source, rrf
from rag.retrieval.rerank import Reranker
from rag.retrieval.rewrite import QueryRewriter
from rag.retrieval.search import Hit, dense_search, keyword_search

Mode = Literal["dense", "keyword", "hybrid"]


@dataclass(frozen=True)
class RetrieverConfig:
    mode: Mode = "hybrid"
    k: int = 10  # results returned
    candidates: int = 50  # results fetched from each first-stage search
    include_generated: bool = True
    rrf_k: int = DEFAULT_K
    rerank: bool = False  # cross-encoder re-scores all candidates, then keeps top k
    max_per_source: int | None = None  # cap chunks per page in the final top k
    rewrite: bool = False  # LLM restates the question in doc vocabulary for search only

    def __post_init__(self) -> None:
        if self.mode not in get_args(Mode):
            raise ValueError(f"mode must be one of {get_args(Mode)}, got {self.mode!r}")
        if self.max_per_source is not None and self.max_per_source <= 0:
            raise ValueError(f"max_per_source must be positive, got {self.max_per_source}")
        if self.k <= 0 or self.rrf_k <= 0:
            raise ValueError(f"k and rrf_k must be positive, got {self.k}, {self.rrf_k}")
        if self.candidates < self.k:
            raise ValueError(f"candidates ({self.candidates}) must be >= k ({self.k})")


class Retriever:
    def __init__(
        self,
        conn: psycopg.Connection,
        embedder: Embedder,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.conn = conn
        self.embedder = embedder
        self.reranker = reranker
        self.rewriter = rewriter

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

    def _first_stage(self, query: str, config: RetrieverConfig, limit: int) -> list[Hit]:
        lists = self._ranked_lists(query, config)
        if config.mode != "hybrid":
            return lists[0][:limit]
        by_id = {hit.chunk_id: hit for hits in lists for hit in hits}
        fused = rrf([[h.chunk_id for h in hits] for hits in lists], config.rrf_k, limit)
        return [replace(by_id[chunk_id], score=score) for chunk_id, score in fused]

    def retrieve(self, query: str, config: RetrieverConfig) -> list[Hit]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if config.rerank and self.reranker is None:
            raise ValueError("config.rerank is set but the Retriever has no reranker")
        if config.rewrite and self.rewriter is None:
            raise ValueError("config.rewrite is set but the Retriever has no rewriter")
        search_query = self.rewriter.rewrite(query) if config.rewrite else query
        # Reranking and the per-page cap both need a deeper list than k to choose from.
        deep = config.rerank or config.max_per_source is not None
        hits = self._first_stage(search_query, config, config.candidates if deep else config.k)
        if config.rerank:
            hits = self.reranker.rerank(query, hits)
        return cap_per_source(hits, lambda h: h.source, config.max_per_source, config.k)
