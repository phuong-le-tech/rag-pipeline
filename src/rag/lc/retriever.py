"""LangChain retriever over the hand-written hybrid search.

Search, fusion and reranking stay in ``rag.retrieval``; this only adapts the result so LCEL
chains and LangGraph tools can call it. Retrieval metrics are therefore identical by construction.
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rag.retrieval.retriever import RetrieverConfig, StageHook
from rag.retrieval.search import Hit


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Any  # rag.retrieval.retriever.Retriever (or any object with the same retrieve())
    config: RetrieverConfig
    on_stage: StageHook | None = None  # per-request: the API streams each stage to the UI

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = self.retriever.retrieve(query, self.config, on_stage=self.on_stage)
        # The whole Hit rides in metadata so later steps lose nothing (ids, urls, headings).
        return [Document(page_content=h.content, metadata={"hit": h}) for h in hits]


def hits_of(docs: list[Document]) -> tuple[Hit, ...]:
    return tuple(d.metadata["hit"] for d in docs)
