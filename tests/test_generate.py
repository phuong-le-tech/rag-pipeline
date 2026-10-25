from types import SimpleNamespace

import pytest

from rag.config import MissingModelError, Settings
from rag.generate.answer import REFUSAL, Answer, answer_question, stream_answer
from rag.generate.citations import check_citations, parse_citations
from rag.generate.generator import (
    ClaudeGenerator,
    FakeGenerator,
    OpenAICompatibleGenerator,
    ScriptedToolGenerator,
    ToolCall,
    Turn,
    _anthropic_messages,
    _openai_messages,
    make_generator,
    make_tool_generator,
)
from rag.generate.prompt import build_prompt, format_numbered
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


def test_format_numbered_uses_the_given_number_and_keeps_block_layout():
    block = format_numbered([(3, hit(1, "Use rollout undo."))])

    assert block.startswith("[3] Deployments > Rolling back\n")
    assert "\nSource: https://kubernetes.io/docs/x1/\n" in block
    assert block.endswith("Use rollout undo.")


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
    settings = Settings(llm_timeout_s=45, openrouter_model="x/y:free")

    assert make_generator(settings.model_copy(update={"llm_backend": "openrouter"})).timeout == 45
    assert make_generator(settings.model_copy(update={"llm_backend": "anthropic"})).timeout == 45


def test_make_generator_openrouter_uses_its_base_url_and_model():
    settings = Settings(llm_backend="openrouter", openrouter_model="google/gemma-4-31b-it:free")

    gen = make_generator(settings)

    assert isinstance(gen, OpenAICompatibleGenerator)
    assert gen.model == "google/gemma-4-31b-it:free"
    assert gen.base_url == "https://openrouter.ai/api/v1"


def test_default_backend_is_openrouter_and_needs_a_configured_model(monkeypatch):
    monkeypatch.delenv("RAG_LLM_BACKEND", raising=False)
    monkeypatch.delenv("RAG_OPENROUTER_MODEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_backend == "openrouter"
    assert settings.openrouter_model is None
    with pytest.raises(MissingModelError, match="RAG_OPENROUTER_MODEL"):
        make_generator(settings)
    with pytest.raises(MissingModelError, match="RAG_OPENROUTER_MODEL"):
        make_tool_generator(settings)


def test_agent_model_alone_is_enough_for_the_tool_generator(monkeypatch):
    monkeypatch.delenv("RAG_OPENROUTER_MODEL", raising=False)
    settings = Settings(_env_file=None, llm_backend="openrouter", agent_model="a/b:free")
    assert make_tool_generator(settings).model == "a/b:free"


def test_model_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("RAG_OPENROUTER_MODEL", "from/env:free")
    settings = Settings(_env_file=None, llm_backend="openrouter")
    assert make_generator(settings).model == "from/env:free"


STREAM_HIT = Hit(1, 0.9, "a.md", "https://x/a", "A", "A > B", "Use rollout undo.", False)


def test_stream_answer_yields_tokens_then_answer():
    gen = FakeGenerator("Use rollout undo [1].")
    items = list(stream_answer("How to roll back?", (STREAM_HIT,), gen))
    assert items[0] == "Use rollout undo [1]."
    final = items[-1]
    assert isinstance(final, Answer) and not final.refused
    assert final.sources[0].hit == STREAM_HIT


def test_stream_answer_without_hits_refuses_without_llm():
    gen = FakeGenerator()
    items = list(stream_answer("q", (), gen))
    assert len(items) == 1 and items[0].refused and gen.calls == []


# --- scripted tool generator --------------------------------------------------------------


def test_scripted_tool_generator_replays_turns_then_refuses():
    gen = ScriptedToolGenerator(
        [Turn("", (ToolCall("c1", "search_docs", '{"query": "x"}'),)), Turn("Done [1].")]
    )
    assert gen.chat("s", [{"role": "user", "content": "q"}], []).tool_calls[0].name == "search_docs"
    assert gen.chat("s", [], []).text == "Done [1]."
    assert gen.chat("s", [], []).text == REFUSAL
    assert len(gen.calls) == 3


# --- tool calling: chat() on the real backends ---------------------------------------------

SEARCH_SPEC = {
    "name": "search_docs",
    "description": "Search the Kubernetes docs.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


class StubToolMessages:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return self.response


def test_claude_chat_maps_tool_use_both_ways():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Let me search."),
            SimpleNamespace(type="tool_use", id="t1", name="search_docs", input={"query": "pdb"}),
        ],
        usage=SimpleNamespace(input_tokens=120, output_tokens=15),
    )
    messages = StubToolMessages(response)
    gen = ClaudeGenerator("m", 100, client=SimpleNamespace(messages=messages))
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "text": "",
            "tool_calls": [ToolCall("t0", "search_docs", '{"query": "a"}')],
        },
        {"role": "tool", "id": "t0", "content": "[1] ..."},
    ]

    turn = gen.chat("sys", history, [SEARCH_SPEC], allow_tools=False)

    sent = messages.requests[0]
    assert sent["model"] == "m"
    assert sent["max_tokens"] == 100
    assert sent["system"] == "sys"
    assert sent["tools"] == [
        {
            "name": "search_docs",
            "description": "Search the Kubernetes docs.",
            "input_schema": SEARCH_SPEC["input_schema"],
        }
    ]
    assert sent["tool_choice"] == {"type": "none"}
    assert sent["messages"] == [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t0", "name": "search_docs", "input": {"query": "a"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "[1] ..."}],
        },
    ]
    assert turn == Turn(
        "Let me search.", (ToolCall("t1", "search_docs", '{"query": "pdb"}'),), 120, 15
    )


def test_claude_chat_allow_tools_omits_tool_choice_and_no_tools_omits_both():
    messages = StubToolMessages(SimpleNamespace(content=[], usage=None))
    gen = ClaudeGenerator("m", 100, client=SimpleNamespace(messages=messages))

    turn = gen.chat("s", [{"role": "user", "content": "q"}], [SEARCH_SPEC], allow_tools=True)
    gen.chat("s", [{"role": "user", "content": "q"}], [], allow_tools=False)

    assert "tool_choice" not in messages.requests[0]
    assert messages.requests[0]["tools"][0]["name"] == "search_docs"
    assert "tools" not in messages.requests[1]
    assert "tool_choice" not in messages.requests[1]
    assert turn == Turn("", (), None, None)


def test_claude_chat_merges_tool_results_and_nudge_into_one_user_message():
    messages = StubToolMessages(SimpleNamespace(content=[], usage=None))
    gen = ClaudeGenerator("m", 100, client=SimpleNamespace(messages=messages))
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "text": "Checking two things.",
            "tool_calls": [
                ToolCall("a", "search_docs", '{"query": "x"}'),
                ToolCall("b", "search_docs", '{"query": "y"}'),
            ],
        },
        {"role": "tool", "id": "a", "content": "A"},
        {"role": "tool", "id": "b", "content": "B"},
        {"role": "user", "content": "Answer now."},
    ]

    gen.chat("s", history, [SEARCH_SPEC])

    sent = messages.requests[0]["messages"]
    assert sent[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Checking two things."},
            {"type": "tool_use", "id": "a", "name": "search_docs", "input": {"query": "x"}},
            {"type": "tool_use", "id": "b", "name": "search_docs", "input": {"query": "y"}},
        ],
    }
    assert sent[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "A"},
            {"type": "tool_result", "tool_use_id": "b", "content": "B"},
            {"type": "text", "text": "Answer now."},
        ],
    }
    assert len(sent) == 3


def test_anthropic_messages_tolerates_invalid_argument_json_in_history():
    history = [
        {"role": "assistant", "text": "", "tool_calls": [ToolCall("a", "search_docs", "{oops")]},
    ]

    assert _anthropic_messages(history) == [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "a", "name": "search_docs", "input": {}}],
        }
    ]


def test_anthropic_messages_drops_whitespace_only_text_block():
    history = [
        {"role": "assistant", "text": "\n", "tool_calls": [ToolCall("a", "search_docs", "{}")]},
    ]

    assert _anthropic_messages(history) == [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "a", "name": "search_docs", "input": {}}],
        }
    ]


@pytest.mark.parametrize("arguments", ["[1]", '"x"', "3"])
def test_anthropic_messages_non_object_arguments_become_empty_input(arguments):
    history = [
        {"role": "assistant", "text": "", "tool_calls": [ToolCall("a", "search_docs", arguments)]},
    ]

    block = _anthropic_messages(history)[0]["content"][0]

    assert block["input"] == {}


class StubOpenAIToolCompletions:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return self.response


def test_openai_chat_round_trip_with_tool_choice_none():
    tool_call = SimpleNamespace(
        id="t1",
        type="function",
        function=SimpleNamespace(name="search_docs", arguments='{"query": "pdb"}'),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Let me search.", tool_calls=[tool_call])
            )
        ],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=15),
    )
    completions = StubOpenAIToolCompletions(response)
    gen = OpenAICompatibleGenerator("m", 128, client=openai_client(completions))
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "text": "Searching.",
            "tool_calls": [ToolCall("t0", "search_docs", '{"query": "a"}')],
        },
        {"role": "tool", "id": "t0", "content": "[1] ..."},
    ]

    turn = gen.chat("sys", history, [SEARCH_SPEC], allow_tools=False)

    sent = completions.requests[0]
    assert sent["model"] == "m"
    assert sent["max_tokens"] == 128
    assert sent["tool_choice"] == "none"
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search the Kubernetes docs.",
                "parameters": SEARCH_SPEC["input_schema"],
            },
        }
    ]
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "Searching.",
            "tool_calls": [
                {
                    "id": "t0",
                    "type": "function",
                    "function": {"name": "search_docs", "arguments": '{"query": "a"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t0", "content": "[1] ..."},
    ]
    assert turn == Turn(
        "Let me search.", (ToolCall("t1", "search_docs", '{"query": "pdb"}'),), 120, 15
    )


def test_openai_chat_allow_tools_sends_tools_without_tool_choice():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=None))],
        usage=None,
    )
    completions = StubOpenAIToolCompletions(response)
    gen = OpenAICompatibleGenerator("m", 128, client=openai_client(completions))

    turn = gen.chat("s", [{"role": "user", "content": "q"}], [SEARCH_SPEC], allow_tools=True)

    assert "tool_choice" not in completions.requests[0]
    assert "tools" in completions.requests[0]
    assert turn == Turn("", (), None, None)


def test_openai_messages_empty_assistant_text_becomes_null_content():
    history = [
        {"role": "assistant", "text": "", "tool_calls": [ToolCall("a", "search_docs", "{}")]}
    ]

    assert _openai_messages("S", history)[1]["content"] is None


def test_make_tool_generator_falls_back_to_llm_backend_and_model():
    settings = Settings(llm_backend="openrouter", openrouter_model="m-or", _env_file=None)

    gen = make_tool_generator(settings)

    assert isinstance(gen, OpenAICompatibleGenerator)
    assert gen.model == "m-or"


def test_agent_model_overrides_only_the_agent_not_make_generator():
    from rag.generate.generator import make_tool_generator

    settings = Settings(
        llm_backend="openrouter",
        openrouter_model="m-or",
        agent_model="m-agent",
        _env_file=None,
    )

    assert make_tool_generator(settings).model == "m-agent"
    assert make_generator(settings).model == "m-or"


def test_agent_backend_anthropic_overrides_openrouter_llm_backend():
    from rag.generate.generator import make_tool_generator

    settings = Settings(
        llm_backend="openrouter",
        agent_backend="anthropic",
        llm_model="claude-x",
        _env_file=None,
    )

    gen = make_tool_generator(settings)

    assert isinstance(gen, ClaudeGenerator)
    assert gen.model == "claude-x"


def test_fake_tool_generator_scripts_a_search_then_a_cited_answer():
    from rag.generate.generator import make_tool_generator

    gen = make_tool_generator(Settings(agent_backend="fake", _env_file=None))

    first = gen.chat("sys", [{"role": "user", "content": "q"}], [])
    second = gen.chat("sys", [{"role": "user", "content": "q"}], [])

    assert isinstance(gen, ScriptedToolGenerator)
    assert first.tool_calls == (ToolCall("fake-1", "search_docs", '{"query": "rollback"}'),)
    assert first.text == ""
    assert second.tool_calls == ()
    assert "[1]" in second.text


# --- Token usage counters ---------------------------------------------------------------


def test_usage_add_sums_known_values_and_flags_reported():
    from rag.generate.generator import Usage

    usage = Usage()
    usage.add(None, None)
    assert usage == Usage(0, 0, False)

    usage.add(100, 20)
    usage.add(50, None)
    assert usage == Usage(150, 20, True)


def test_claude_generator_counts_usage_from_generate_and_chat():
    class Messages:
        def __init__(self):
            self.responses = [
                SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="Hi")],
                    usage=SimpleNamespace(input_tokens=100, output_tokens=20),
                ),
                SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="Yo")],
                    usage=SimpleNamespace(input_tokens=50, output_tokens=5),
                ),
            ]

        def create(self, **request):
            return self.responses.pop(0)

    gen = ClaudeGenerator(model="m", max_tokens=8, client=SimpleNamespace(messages=Messages()))

    gen.generate("S", "U")
    gen.chat("S", [{"role": "user", "content": "U"}], [])

    assert gen.usage.input_tokens == 150
    assert gen.usage.output_tokens == 25
    assert gen.usage.reported is True


def test_openai_generator_without_usage_block_stays_unreported():
    gen = OpenAICompatibleGenerator(
        model="m", max_tokens=64, client=openai_client(StubCompletions())
    )

    gen.generate("S", "U")

    assert gen.usage.reported is False
    assert gen.usage.input_tokens == 0


def test_openai_generator_counts_usage_from_generate():
    class Completions(StubCompletions):
        def create(self, **request):
            response = super().create(**request)
            response.usage = SimpleNamespace(prompt_tokens=30, completion_tokens=7)
            return response

    gen = OpenAICompatibleGenerator(model="m", max_tokens=64, client=openai_client(Completions()))

    gen.generate("S", "U")

    assert (gen.usage.input_tokens, gen.usage.output_tokens, gen.usage.reported) == (30, 7, True)


def test_usage_of_returns_none_for_generator_without_counters():
    from rag.generate.generator import usage_of

    assert usage_of(FakeGenerator()) is None


def test_usage_of_returns_counter_of_real_generator():
    from rag.generate.generator import usage_of

    gen = ClaudeGenerator(model="m", max_tokens=8, client=SimpleNamespace())

    assert usage_of(gen) is gen.usage
