from types import SimpleNamespace

import pytest

from rag.config import Settings
from rag.generate.answer import REFUSAL, answer_question
from rag.generate.citations import check_citations, parse_citations
from rag.generate.generator import (
    ClaudeGenerator,
    FakeGenerator,
    OpenAICompatibleGenerator,
    make_generator,
)
from rag.generate.prompt import build_prompt
from rag.retrieval.retriever import RetrieverConfig
from rag.retrieval.search import Hit


def hit(chunk_id, content, title="Deployments", heading="Deployments > Rolling back"):
    url = f"https://kubernetes.io/docs/x{chunk_id}/"
    return Hit(chunk_id, 1.0, f"x{chunk_id}.md", url, title, heading, content, False)


class StubRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def retrieve(self, query, config):
        self.calls.append((query, config))
        return self.hits


# --- prompt -----------------------------------------------------------------------------


def test_prompt_numbers_context_blocks_with_heading_and_url():
    system, user = build_prompt("How to roll back?", [hit(1, "Use rollout undo."), hit(2, "B")])

    assert "[1] Deployments > Rolling back" in user
    assert "https://kubernetes.io/docs/x1/" in user
    assert "Use rollout undo." in user
    assert "[2]" in user
    assert user.rstrip().endswith("How to roll back?")
    assert "only" in system.lower() and "[n]" in system


def test_prompt_rejects_empty_context():
    with pytest.raises(ValueError):
        build_prompt("q", [])


# --- citations ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Use undo [1].", [1]),
        ("Both [1, 3] and [2][4].", [1, 3, 2, 4]),
        ("Repeat [2] then [2] again.", [2]),
        ("No citations here.", []),
        ("Array index a[0] and [x] are not citations.", []),
    ],
)
def test_parse_citations_in_first_appearance_order(text, expected):
    assert parse_citations(text) == expected


def test_check_citations_flags_numbers_outside_context():
    report = check_citations("Fine [1], invented [7], also [0].", num_sources=3)

    assert report.cited == (1,)
    assert report.invalid == (7, 0)


# --- answer -------------------------------------------------------------------------------


def test_answer_returns_text_and_only_cited_sources():
    hits = [hit(1, "Use rollout undo."), hit(2, "Unrelated."), hit(3, "History limit.")]
    gen = FakeGenerator("Run `kubectl rollout undo` [1]. History is capped [3].")

    result = answer_question("How to roll back?", StubRetriever(hits), gen, RetrieverConfig())

    assert result.text.startswith("Run `kubectl rollout undo`")
    assert [s.number for s in result.sources] == [1, 3]
    assert result.sources[0].hit.chunk_id == 1
    assert result.invalid_citations == ()
    assert result.refused is False
    assert "How to roll back?" in gen.calls[0][1]


def test_answer_refuses_without_calling_llm_when_nothing_retrieved():
    gen = FakeGenerator("should not be used")

    result = answer_question("Unknown topic?", StubRetriever([]), gen, RetrieverConfig())

    assert result.refused is True
    assert result.text == REFUSAL
    assert gen.calls == []


def test_answer_reports_invalid_citations():
    gen = FakeGenerator("Claim [1]. Made-up [9].")

    result = answer_question("q", StubRetriever([hit(1, "a")]), gen, RetrieverConfig())

    assert result.invalid_citations == (9,)
    assert [s.number for s in result.sources] == [1]


def test_answer_marks_model_refusal():
    gen = FakeGenerator(REFUSAL)

    result = answer_question("q", StubRetriever([hit(1, "a")]), gen, RetrieverConfig())

    assert result.refused is True
    assert result.sources == ()


def test_answer_passes_retriever_config_through():
    retriever = StubRetriever([hit(1, "a")])
    config = RetrieverConfig(k=3, rerank=True)

    answer_question("q", retriever, FakeGenerator("x [1]"), config)

    assert retriever.calls == [("q", config)]


def test_answer_streams_tokens_to_callback():
    pieces = []

    result = answer_question(
        "q",
        StubRetriever([hit(1, "a")]),
        FakeGenerator("Yes [1]."),
        RetrieverConfig(),
        pieces.append,
    )

    assert pieces == ["Yes [1]."]
    assert result.text == "Yes [1]."


# --- Claude wrapper (stub client, no network) ---------------------------------------------


class StubMessages:
    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        blocks = [
            SimpleNamespace(type="thinking", text="hidden"),
            SimpleNamespace(type="text", text="Hi [1]."),
        ]
        return SimpleNamespace(content=blocks)


def test_claude_generator_sends_system_and_user_and_joins_text_blocks():
    messages = StubMessages()
    gen = ClaudeGenerator(
        model="claude-sonnet-5-5", max_tokens=256, client=SimpleNamespace(messages=messages)
    )

    text = gen.generate("SYS", "USER")

    assert text == "Hi [1]."
    assert messages.requests == [
        {
            "model": "claude-sonnet-5-5",
            "max_tokens": 256,
            "system": "SYS",
            "messages": [{"role": "user", "content": "USER"}],
        }
    ]


def test_make_generator_picks_backend():
    assert isinstance(make_generator(Settings(llm_backend="fake")), FakeGenerator)
    claude = make_generator(Settings(llm_backend="anthropic", llm_model="claude-haiku-5-5"))
    assert isinstance(claude, ClaudeGenerator)
    assert claude.model == "claude-haiku-5-5"


# --- OpenAI-compatible wrapper (OpenRouter), stub client ------------------------------------


class StubCompletions:
    def __init__(self, text="Ans [1].", pieces=("Ans ", None, "[1].")):
        self.text = text
        self.pieces = pieces
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        if request.get("stream"):
            return iter(
                [SimpleNamespace(choices=[])]  # providers sometimes send empty keep-alive chunks
                + [
                    SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))])
                    for p in self.pieces
                ]
            )
        message = SimpleNamespace(content=self.text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def openai_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_openai_compatible_generator_sends_system_then_user():
    completions = StubCompletions()
    gen = OpenAICompatibleGenerator(
        model="m:free", max_tokens=128, client=openai_client(completions)
    )

    assert gen.generate("SYS", "USER") == "Ans [1]."
    assert completions.requests == [
        {
            "model": "m:free",
            "max_tokens": 128,
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "USER"},
            ],
        }
    ]


def test_openai_compatible_generator_streams_and_skips_empty_chunks():
    completions = StubCompletions()
    gen = OpenAICompatibleGenerator(model="m", max_tokens=64, client=openai_client(completions))

    assert list(gen.stream("S", "U")) == ["Ans ", "[1]."]
    assert completions.requests[0]["stream"] is True


def test_openai_compatible_generator_handles_null_content():
    completions = StubCompletions(text=None)
    gen = OpenAICompatibleGenerator(model="m", max_tokens=64, client=openai_client(completions))

    assert gen.generate("S", "U") == ""


def test_make_generator_sets_request_timeout_on_both_backends():
    settings = Settings(llm_timeout_s=45)

    assert make_generator(settings.model_copy(update={"llm_backend": "openrouter"})).timeout == 45
    assert make_generator(settings.model_copy(update={"llm_backend": "anthropic"})).timeout == 45


def test_make_generator_openrouter_uses_its_base_url_and_model():
    settings = Settings(llm_backend="openrouter", openrouter_model="google/gemma-4-31b-it:free")

    gen = make_generator(settings)

    assert isinstance(gen, OpenAICompatibleGenerator)
    assert gen.model == "google/gemma-4-31b-it:free"
    assert gen.base_url == "https://openrouter.ai/api/v1"
