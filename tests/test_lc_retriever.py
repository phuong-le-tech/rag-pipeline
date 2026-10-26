import pytest

from rag.lc.retriever import HybridRetriever, hits_of
from rag.retrieval.retriever import Retriever, RetrieverConfig
from rag.retrieval.search import Hit


def hit(chunk_id, content):
    url = f"https://kubernetes.io/docs/x{chunk_id}/"
    return Hit(chunk_id, 1.0, f"x{chunk_id}.md", url, "Title", "Title > Part", content, False)


HITS = [hit(1, "one"), hit(2, "two"), hit(3, "three")]


class StubRetriever:
    def __init__(self, hits=HITS):
        self.hits = hits
        self.calls = []

    def retrieve(self, query, config, on_stage=None):
        self.calls.append((query, config))
        if on_stage:
            on_stage("dense", self.hits, 1.5)
        return list(self.hits)


def test_invoke_returns_documents_in_hit_order():
    docs = HybridRetriever(retriever=StubRetriever(), config=RetrieverConfig()).invoke("q")
    assert [d.page_content for d in docs] == ["one", "two", "three"]


def test_hits_round_trip():
    docs = HybridRetriever(retriever=StubRetriever(), config=RetrieverConfig()).invoke("q")
    assert hits_of(docs) == tuple(HITS)


def test_config_reaches_the_wrapped_retriever():
    stub, config = StubRetriever(), RetrieverConfig(k=3, rerank=False)
    HybridRetriever(retriever=stub, config=config).invoke("how to roll back")
    assert stub.calls == [("how to roll back", config)]


def test_on_stage_is_forwarded():
    seen = []
    retriever = HybridRetriever(
        retriever=StubRetriever(),
        config=RetrieverConfig(),
        on_stage=lambda name, hits, ms: seen.append((name, len(hits), ms)),
    )
    retriever.invoke("q")
    assert seen == [("dense", 3, 1.5)]


def test_blank_query_raises_value_error_like_the_real_retriever():
    real = Retriever(conn=None, embedder=None)
    with pytest.raises(ValueError, match="blank"):
        HybridRetriever(retriever=real, config=RetrieverConfig()).invoke("   ")
