from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage

from rag.generate.answer import Answer, stream_answer
from rag.generate.prompt import REFUSAL, build_prompt
from rag.lc.chain import LangChainGenerator, lc_answer_question
from rag.lc.retriever import HybridRetriever
from rag.retrieval.retriever import RetrieverConfig
from rag.retrieval.search import Hit


def hit(chunk_id, content):
    url = f"https://kubernetes.io/docs/x{chunk_id}/"
    return Hit(chunk_id, 1.0, f"x{chunk_id}.md", url, "Title", "Title > Part", content, False)


HITS = [hit(1, "Use kubectl rollout undo."), hit(2, "Revisions are kept.")]


class StubRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, config, on_stage=None):
        return list(self.hits)


def retriever(hits=HITS):
    return HybridRetriever(retriever=StubRetriever(hits), config=RetrieverConfig())


def test_answer_has_validated_citations():
    model = FakeListChatModel(responses=["Use kubectl [1]."])
    answer = lc_answer_question("how?", retriever(), LangChainGenerator(model))
    assert isinstance(answer, Answer) and not answer.refused
    assert answer.text == "Use kubectl [1]."
    assert [(s.number, s.hit.chunk_id) for s in answer.sources] == [(1, 1)]


def test_empty_retrieval_refuses_without_llm_call():
    model = FakeListChatModel(responses=["never used"])
    answer = lc_answer_question("how?", retriever([]), LangChainGenerator(model))
    assert answer.refused and answer.text.startswith(REFUSAL)
    assert model.i == 0


def test_out_of_range_citation_is_reported():
    model = FakeListChatModel(responses=["Use kubectl [9]."])
    answer = lc_answer_question("how?", retriever(), LangChainGenerator(model))
    assert answer.invalid_citations == (9,)


def test_model_refusal_is_a_refusal():
    model = FakeListChatModel(responses=[REFUSAL])
    answer = lc_answer_question("how?", retriever(), LangChainGenerator(model))
    assert answer.refused and answer.sources == ()


def test_tokens_stream_then_match_final_text():
    model = FakeListChatModel(responses=["Use kubectl [1]."])
    pieces: list[str] = []
    answer = lc_answer_question(
        "how?", retriever(), LangChainGenerator(model), on_text=pieces.append
    )
    assert len(pieces) > 1 and "".join(pieces) == answer.text == "Use kubectl [1]."


def test_generator_works_with_the_handwritten_stream_answer():
    model = FakeListChatModel(responses=["Use kubectl [1]."])
    items = list(stream_answer("how?", tuple(HITS), LangChainGenerator(model)))
    assert isinstance(items[-1], Answer) and items[-1].text == "Use kubectl [1]."
    assert "".join(items[:-1]) == items[-1].text


def test_prompt_is_identical_to_the_handwritten_prompt():
    class Recorder(BaseCallbackHandler):
        messages = None

        def on_chat_model_start(self, serialized, messages, **kwargs):
            Recorder.messages = messages[0]

    system, user = build_prompt("how?", HITS)
    model = FakeListChatModel(responses=["Use kubectl [1]."], callbacks=[Recorder()])
    lc_answer_question("how?", retriever(), LangChainGenerator(model))
    assert [(m.type, m.content) for m in Recorder.messages] == [
        ("system", system),
        ("human", user),
    ]


def test_generate_counts_reported_tokens():
    reply = AIMessage(
        content="Use kubectl [1].",
        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    )
    generator = LangChainGenerator(GenericFakeChatModel(messages=iter([reply])))
    generator.generate("sys", "user")
    assert (generator.usage.input_tokens, generator.usage.output_tokens) == (10, 4)
    assert generator.usage.reported


def test_usage_stays_unreported_when_the_provider_sends_none():
    generator = LangChainGenerator(FakeListChatModel(responses=["x"]))
    generator.generate("sys", "user")
    assert not generator.usage.reported
