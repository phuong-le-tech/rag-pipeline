from langchain_core.language_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage

from rag.agent.loop import Limits
from rag.agent.tools import SearchResult
from rag.config import Settings
from rag.lc.agent import run_lc_agent
from rag.lc.chain import LangChainGenerator
from rag.lc.factory import lc_agent_answerer, lc_pipeline_answerer, make_agent_model
from rag.lc.llm import fake_tool_chat_model
from rag.retrieval.retriever import RetrieverConfig
from rag.retrieval.search import Hit


def hit(chunk_id, content):
    url = f"https://kubernetes.io/docs/x{chunk_id}/"
    return Hit(chunk_id, 1.0, f"x{chunk_id}.md", url, "Title", "Title > Part", content, False)


HITS = (hit(1, "Use kubectl rollout undo."),)


class StubRetriever:
    def __init__(self, hits=HITS):
        self.hits = hits

    def retrieve(self, query, config, on_stage=None):
        return list(self.hits)


def generator(*texts):
    return LangChainGenerator(FakeListChatModel(responses=list(texts)))


def test_pipeline_answerer_returns_answered():
    answerer = lc_pipeline_answerer(StubRetriever(), generator("Use undo [1]."), RetrieverConfig())
    answered = answerer("how?")
    assert answered.stopped == "pipeline"
    assert (answered.llm_calls, answered.tool_calls) == (1, 0)
    assert [s.number for s in answered.answer.sources] == [1]
    assert answered.input_tokens is None  # the fake reports no usage


def test_pipeline_answerer_reports_token_deltas_per_question():
    usage = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    replies = iter([AIMessage(content="Use undo [1].", usage_metadata=usage)] * 2)
    gen = LangChainGenerator(GenericFakeChatModel(messages=replies))
    answerer = lc_pipeline_answerer(StubRetriever(), gen, RetrieverConfig())
    answerer("first")
    second = answerer("second")
    assert (second.input_tokens, second.output_tokens) == (10, 4)


def test_pipeline_answerer_without_passages_makes_no_llm_call():
    answerer = lc_pipeline_answerer(StubRetriever(()), generator("unused"), RetrieverConfig())
    answered = answerer("how?")
    assert answered.llm_calls == 0 and answered.answer.refused


def test_agent_answerer_counts_calls():
    search_call = AIMessage(
        content="",
        tool_calls=[{"name": "search_docs", "args": {"query": "rollback"}, "id": "c1"}],
    )
    answerer = lc_agent_answerer(
        lambda: fake_tool_chat_model([search_call, AIMessage(content="Use undo [1].")]),
        lambda query, k: SearchResult(HITS, (), 1.0),
        Limits(),
    )
    answered = answerer("how?")
    assert (answered.llm_calls, answered.tool_calls, answered.stopped) == (2, 1, "answered")


def test_fake_agent_model_runs_the_scripted_search_then_answers():
    model = make_agent_model(Settings(_env_file=None, llm_backend="fake"))
    result = run_lc_agent("how?", model, lambda query, k: SearchResult(HITS, (), 1.0), Limits())
    assert result.stopped == "answered" and "fake agent answer" in result.answer.text


def test_agent_model_uses_the_agent_model_name_on_openrouter():
    settings = Settings(
        _env_file=None,
        llm_backend="openrouter",
        openrouter_model="vendor/main",
        agent_model="vendor/agent",
        OPENROUTER_API_KEY="k",
    )
    assert make_agent_model(settings).model_name == "vendor/agent"
