"""Glue between the LangChain implementation and the hand-written CLI, evals and API."""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from rag.agent.loop import Limits
from rag.agent.tools import SearchResult
from rag.config import Settings
from rag.evals.answers import Answered, answered_from_agent, pipeline_answerer
from rag.generate.answer import Answer
from rag.generate.generator import Generator
from rag.lc.agent import run_lc_agent
from rag.lc.chain import lc_answer_question
from rag.lc.llm import fake_tool_chat_model, make_chat_model
from rag.lc.retriever import HybridRetriever
from rag.retrieval.retriever import RetrieverConfig


def make_agent_model(settings: Settings, max_retries: int | None = None) -> BaseChatModel:
    """The agent's chat model: ``agent_backend``/``agent_model`` win over the main LLM settings."""
    backend = settings.agent_backend or settings.llm_backend
    if backend == "fake":  # same script as ScriptedToolGenerator in make_tool_generator
        search = {"name": "search_docs", "args": {"query": "rollback"}, "id": "fake-1"}
        return fake_tool_chat_model(
            [
                AIMessage(content="", tool_calls=[search]),
                AIMessage(content="This is a fake agent answer [1]."),
            ]
        )
    updates: dict = {"llm_backend": backend}
    if settings.agent_model and backend == "openrouter":
        updates["openrouter_model"] = settings.agent_model
    return make_chat_model(settings.model_copy(update=updates), max_retries)


def lc_pipeline_answerer(
    retriever: object, generator: Generator, config: RetrieverConfig
) -> Callable[[str], Answered]:
    """Same call shape and token accounting as ``pipeline_answerer``; the chain is LCEL."""

    def answer_fn(
        question: str, retriever: object, generator: Generator, config: RetrieverConfig
    ) -> Answer:
        return lc_answer_question(
            question, HybridRetriever(retriever=retriever, config=config), generator
        )

    return pipeline_answerer(retriever, generator, config, answer_fn=answer_fn)


def lc_agent_answerer(
    make_model: Callable[[], BaseChatModel],
    search: Callable[[str, int], SearchResult],
    limits: Limits,
) -> Callable[[str], Answered]:
    def answer(question: str) -> Answered:
        return answered_from_agent(run_lc_agent(question, make_model(), search, limits))

    return answer
