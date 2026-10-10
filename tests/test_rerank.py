import pytest

from rag.config import Settings
from rag.embed import FakeEmbedder
from rag.retrieval.rerank import CrossEncoderReranker, make_reranker
from rag.retrieval.retriever import Retriever, RetrieverConfig
from rag.retrieval.search import Hit


def hit(chunk_id, content, heading="H", score=0.0):
    return Hit(chunk_id, score, f"s{chunk_id}.md", "u", "T", heading, content, False)


class StubCrossEncoder:
    """Mimics fastembed TextCrossEncoder: scores = count of query words in the document."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, documents, batch_size):
        docs = list(documents)
        self.calls.append((query, docs, batch_size))
        words = query.lower().split()
        for d in docs:
            yield float(sum(d.lower().count(w) for w in words))


def test_rerank_orders_by_model_score_and_replaces_scores():
    model = StubCrossEncoder()
    reranker = CrossEncoderReranker(model=model, batch_size=8)
    hits = [hit(1, "nothing here"), hit(2, "rollback rollback"), hit(3, "one rollback")]

    ranked = reranker.rerank("rollback", hits)

    assert [h.chunk_id for h in ranked] == [2, 3, 1]
    assert [h.score for h in ranked] == [2.0, 1.0, 0.0]


def test_rerank_sends_heading_with_content():
    model = StubCrossEncoder()

    CrossEncoderReranker(model=model).rerank("q", [hit(1, "body", heading="Page > Section")])

    assert model.calls[0][1] == ["Page > Section\n\nbody"]


def test_rerank_empty_hits_skips_model():
    model = StubCrossEncoder()

    assert CrossEncoderReranker(model=model).rerank("q", []) == []
    assert model.calls == []


def test_make_reranker_uses_configured_model():
    reranker = make_reranker(Settings(reranker_model="Xenova/ms-marco-MiniLM-L-6-v2"))

    assert reranker.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"


def test_rerank_config_requires_a_reranker():
    with pytest.raises(ValueError, match="reranker"):
        Retriever(conn=None, embedder=FakeEmbedder(8)).retrieve("q", RetrieverConfig(rerank=True))


class KeywordOnlyRetriever(Retriever):
    """Bypasses the DB: first stage returns fixed candidates, best-first."""

    def __init__(self, candidates, reranker):
        super().__init__(conn=None, embedder=FakeEmbedder(8), reranker=reranker)
        self._candidates = candidates

    def _ranked_lists(self, query, config):
        return [self._candidates[: config.candidates]]


def test_retriever_caps_chunks_per_source_after_reranking():
    same_page = [Hit(i, 0.0, "one.md", "u", "T", "H", "rollback", False) for i in range(1, 4)]
    other = [hit(10, "rollback too")]
    retriever = KeywordOnlyRetriever(
        same_page + other, CrossEncoderReranker(model=StubCrossEncoder())
    )

    hits = retriever.retrieve(
        "rollback",
        RetrieverConfig(mode="keyword", k=2, candidates=10, rerank=True, max_per_source=1),
    )

    assert [h.source for h in hits] == ["one.md", "s10.md"]


def test_config_rejects_bad_max_per_source():
    with pytest.raises(ValueError):
        RetrieverConfig(max_per_source=0)


def test_retriever_reranks_all_candidates_then_truncates_to_k():
    candidates = [hit(i, "filler") for i in range(1, 10)] + [hit(10, "rollback answer")]
    retriever = KeywordOnlyRetriever(candidates, CrossEncoderReranker(model=StubCrossEncoder()))

    hits = retriever.retrieve(
        "rollback", RetrieverConfig(mode="keyword", k=2, candidates=10, rerank=True)
    )

    assert [h.chunk_id for h in hits][0] == 10
    assert len(hits) == 2
