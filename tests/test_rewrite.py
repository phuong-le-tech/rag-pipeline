import pytest

from rag.embed import FakeEmbedder
from rag.generate.generator import FakeGenerator
from rag.retrieval.retriever import Retriever, RetrieverConfig
from rag.retrieval.rewrite import QueryRewriter
from rag.retrieval.search import Hit


class CountingGenerator(FakeGenerator):
    def generate(self, system, user):
        self.calls.append((system, user))
        return self.answer


def test_rewriter_returns_cleaned_query_and_caches_per_question():
    gen = CountingGenerator('  "kubectl rollout undo deployment previous revision"\n')
    rewriter = QueryRewriter(gen)

    first = rewriter.rewrite("how do I go back to what was running before?")
    second = rewriter.rewrite("how do I go back to what was running before?")

    assert first == "kubectl rollout undo deployment previous revision"
    assert second == first
    assert len(gen.calls) == 1
    assert "Kubernetes documentation" in gen.calls[0][0]


def test_rewriter_falls_back_to_original_on_empty_output():
    rewriter = QueryRewriter(CountingGenerator("   "))

    assert rewriter.rewrite("original question?") == "original question?"


def test_rewriter_keeps_only_first_line():
    rewriter = QueryRewriter(CountingGenerator("kubectl top pod --sort-by=cpu\nExplanation: ..."))

    assert rewriter.rewrite("q") == "kubectl top pod --sort-by=cpu"


class RecordingRetriever(Retriever):
    def __init__(self, rewriter, reranker=None):
        super().__init__(conn=None, embedder=FakeEmbedder(8), reranker=reranker, rewriter=rewriter)
        self.searched = []

    def _ranked_lists(self, query, config, on_stage=None):
        self.searched.append(query)
        return [[Hit(1, 1.0, "a.md", "u", "T", "H", "c", False)]]


class RecordingReranker:
    def __init__(self):
        self.queries = []

    def rerank(self, query, hits):
        self.queries.append(query)
        return list(hits)


def test_retriever_searches_with_rewrite_but_reranks_with_original_question():
    rewriter = QueryRewriter(CountingGenerator("rollout undo"))
    reranker = RecordingReranker()
    retriever = RecordingRetriever(rewriter, reranker)

    retriever.retrieve(
        "go back to before?", RetrieverConfig(mode="keyword", rerank=True, rewrite=True)
    )

    assert retriever.searched == ["rollout undo"]
    assert reranker.queries == ["go back to before?"]


def test_retriever_without_rewrite_flag_searches_original():
    retriever = RecordingRetriever(QueryRewriter(CountingGenerator("rewritten")))

    retriever.retrieve("original?", RetrieverConfig(mode="keyword"))

    assert retriever.searched == ["original?"]


def test_rewrite_flag_requires_a_rewriter():
    retriever = RecordingRetriever(rewriter=None)

    with pytest.raises(ValueError, match="rewriter"):
        retriever.retrieve("q", RetrieverConfig(rewrite=True))
