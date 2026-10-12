"""Question -> retrieve -> grounded prompt -> LLM -> validated, cited answer."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

from rag.generate.citations import check_citations
from rag.generate.generator import Generator
from rag.generate.prompt import REFUSAL, build_prompt
from rag.retrieval.retriever import RetrieverConfig
from rag.retrieval.search import Hit

__all__ = ["REFUSAL", "Answer", "Source", "answer_question", "stream_answer"]


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


def _finish(text: str, hits: tuple[Hit, ...]) -> Answer:
    if text.strip().startswith(REFUSAL):
        return Answer(text.strip(), (), (), True, hits)
    report = check_citations(text, num_sources=len(hits))
    sources = tuple(Source(n, hits[n - 1]) for n in report.cited)
    return Answer(text, sources, report.invalid, False, hits)


def stream_answer(
    question: str, hits: tuple[Hit, ...], generator: Generator
) -> Iterator[str | Answer]:
    """Yield answer tokens as they arrive, then the validated ``Answer`` last."""
    if not hits:
        yield _refusal()  # nothing relevant: refuse without paying for an LLM call
        return
    system, user = build_prompt(question, hits)
    pieces = []
    for piece in generator.stream(system, user):
        pieces.append(piece)
        yield piece
    yield _finish("".join(pieces), hits)


def answer_question(
    question: str,
    retriever: SupportsRetrieve,
    generator: Generator,
    config: RetrieverConfig,
    on_text: Callable[[str], None] | None = None,
) -> Answer:
    """Answer ``question``. With ``on_text``, tokens are streamed to it as they arrive."""
    hits = tuple(retriever.retrieve(question, config))
    if on_text is None and hits:
        system, user = build_prompt(question, hits)
        return _finish(generator.generate(system, user), hits)  # evals use generate()
    for item in stream_answer(question, hits, generator):
        if isinstance(item, Answer):
            return item
        on_text(item)
    raise AssertionError("stream_answer always ends with an Answer")
