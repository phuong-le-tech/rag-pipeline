"""LCEL version of retrieve -> grounded prompt -> LLM -> validated, cited answer.

The prompt text, refusal rule and citation validation are the hand-written ones
(``build_prompt``, ``finish_answer``), so a comparison measures orchestration only.
"""

from collections.abc import Callable, Iterator
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from rag.generate.answer import Answer, finish_answer
from rag.generate.generator import Usage
from rag.generate.prompt import REFUSAL, build_prompt
from rag.lc.retriever import HybridRetriever, hits_of

PROMPT = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{user}")])


class UsageCallback(BaseCallbackHandler):
    """Adds the provider's reported token counts to a ``Usage`` (same counter evals read)."""

    def __init__(self, usage: Usage) -> None:
        self.usage = usage

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                meta = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if meta:
                    self.usage.add(meta.get("input_tokens"), meta.get("output_tokens"))


class LangChainGenerator:
    """The ``Generator`` protocol over an LCEL chain: prompt | chat model | string parser."""

    def __init__(self, model: BaseChatModel) -> None:
        self.usage = Usage()
        self._config = {"callbacks": [UsageCallback(self.usage)]}
        self._chain = PROMPT | model | StrOutputParser()

    def generate(self, system: str, user: str) -> str:
        return self._chain.invoke({"system": system, "user": user}, self._config)

    def stream(self, system: str, user: str) -> Iterator[str]:
        yield from self._chain.stream({"system": system, "user": user}, self._config)


def lc_answer_question(
    question: str,
    retriever: HybridRetriever,
    generator: LangChainGenerator,
    on_text: Callable[[str], None] | None = None,
) -> Answer:
    """Answer ``question``. With ``on_text``, tokens are streamed to it as they arrive."""

    def answer(inputs: dict) -> Answer:
        hits = inputs["hits"]
        if not hits:
            return finish_answer(REFUSAL, ())  # nothing relevant: no LLM call
        system, user = build_prompt(inputs["question"], hits)
        if on_text is None:
            return finish_answer(generator.generate(system, user), hits)
        pieces = []
        for piece in generator.stream(system, user):
            pieces.append(piece)
            on_text(piece)
        return finish_answer("".join(pieces), hits)

    chain = RunnableParallel(
        question=RunnablePassthrough(), hits=retriever | RunnableLambda(hits_of)
    ) | RunnableLambda(answer)
    return chain.invoke(question)
