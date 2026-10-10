"""Question -> retrieve -> grounded prompt -> LLM -> validated, cited answer."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rag.generate.citations import check_citations
from rag.generate.generator import Generator
from rag.generate.prompt import REFUSAL, build_prompt
from rag.retrieval.retriever import RetrieverConfig
from rag.retrieval.search import Hit

__all__ = ["REFUSAL", "Answer", "Source", "answer_question"]


class SupportsRetrieve(Protocol):
    def retrieve(self, query: str, config: RetrieverConfig) -> list[Hit]: ...


@dataclass(frozen=True)
class Source:
    number: int  # the [n] used in the answer
    hit: Hit


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[Source, ...]  # only passages actually cited, in citation order
    invalid_citations: tuple[int, ...]  # cited numbers that match no passage
    refused: bool
    retrieved: tuple[Hit, ...]  # everything sent to the LLM (for evals)


def _refusal(retrieved: tuple[Hit, ...] = ()) -> Answer:
    return Answer(REFUSAL, (), (), True, retrieved)


def answer_question(
    question: str,
    retriever: SupportsRetrieve,
    generator: Generator,
    config: RetrieverConfig,
    on_text: Callable[[str], None] | None = None,
) -> Answer:
    """Answer ``question``. With ``on_text``, tokens are streamed to it as they arrive."""
    hits = tuple(retriever.retrieve(question, config))
    if not hits:
        return _refusal()  # nothing relevant: refuse without paying for an LLM call

    system, user = build_prompt(question, hits)
    if on_text is None:
        text = generator.generate(system, user)
    else:
        pieces = []
        for piece in generator.stream(system, user):
            on_text(piece)
            pieces.append(piece)
        text = "".join(pieces)

    if text.strip().startswith(REFUSAL):
        return Answer(text.strip(), (), (), True, hits)
    report = check_citations(text, num_sources=len(hits))
    sources = tuple(Source(n, hits[n - 1]) for n in report.cited)
    return Answer(text, sources, report.invalid, False, hits)
