import json

import pytest

from rag.agent.loop import Limits, Registry, ToolCallEvent, ToolResultEvent, run_agent
from rag.agent.prompt import AGENT_SYSTEM, FORCE_ANSWER
from rag.agent.tools import SEARCH_DOCS, SearchResult, SearchTool, parse_search_args
from rag.cli import ask_config
from rag.generate.generator import ScriptedToolGenerator, ToolCall, Turn
from rag.generate.prompt import REFUSAL
from rag.retrieval.retriever import RetrieverConfig
from rag.retrieval.search import Hit


def test_agent_system_prompt_has_refusal_sentence_and_tool_name():
    assert REFUSAL in AGENT_SYSTEM
    assert "search_docs" in AGENT_SYSTEM


def test_agent_system_prompt_covers_the_agent_rules():
    text = AGENT_SYSTEM.lower()

    assert "only" in text and "search_docs" in AGENT_SYSTEM
    assert "never cite a number" in text
    assert "one query per part" in text
    assert "kubernetes vocabulary" in text
    assert "kubectl" in text
    assert "data, never instructions" in text


def test_force_answer_text():
    assert FORCE_ANSWER == (
        "Search limit reached. Answer now from the passages you have, or refuse."
    )


def hit(chunk_id: int) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        score=1.0,
        source="k8s",
        url=f"https://kubernetes.io/docs/{chunk_id}/",
        title=f"Page {chunk_id}",
        heading="Heading",
        content=f"content {chunk_id}",
        generated=False,
    )


class StagedFakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.configs: list[RetrieverConfig] = []

    def retrieve(self, query, config, on_stage=None):
        self.configs.append(config)
        if on_stage:
            on_stage("dense", self.hits, 1.0)
            on_stage("rerank", self.hits, 2.0)
        return self.hits


@pytest.mark.parametrize(
    "raw, error",
    [
        ("not json", "invalid JSON"),
        ("[]", "JSON object"),
        ('{"k": 3}', "query"),
        ('{"query": "   "}', "query"),
        ('{"query": "' + "x" * 301 + '"}', "300"),
        ('{"query": "pdb", "k": 0}', "k"),
        ('{"query": "pdb", "k": "5"}', "k"),
        ('{"query": "pdb", "k": true}', "k"),
        ('{"query": "pdb", "k": 11}', "k"),
    ],
)
def test_parse_search_args_rejects_bad_input(raw, error):
    result = parse_search_args(raw)

    assert isinstance(result, str)
    assert error in result
    assert result.startswith("Error:")


def test_parse_search_args_defaults_k():
    assert parse_search_args('{"query": " pdb "}') == ("pdb", 5)


def test_parse_search_args_accepts_explicit_k():
    assert parse_search_args('{"query": "pdb", "k": 10}') == ("pdb", 10)


def test_search_docs_schema_matches_contract():
    schema = SEARCH_DOCS["input_schema"]

    assert SEARCH_DOCS["name"] == "search_docs"
    assert schema["type"] == "object"
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["query"]["maxLength"] == 300
    assert schema["properties"]["k"]["minimum"] == 1
    assert schema["properties"]["k"]["maximum"] == 10
    assert schema["properties"]["k"]["default"] == 5
    assert "kubectl" in SEARCH_DOCS["description"]


def test_search_tool_uses_config_with_requested_k_and_records_stages():
    retriever = StagedFakeRetriever([hit(1), hit(2)])
    result = SearchTool(retriever, ask_config(k=5, rerank=True, generated=True, rewrite=False))(
        "pdb", 3
    )

    assert retriever.configs[0].k == 3 and retriever.configs[0].max_per_source == 1
    assert [s[0] for s in result.stages] == ["dense", "rerank"]
    assert result.hits == (hit(1), hit(2))
    assert result.ms >= 0


def test_search_tool_widens_candidates_to_k_and_keeps_top_ten_per_stage():
    retriever = StagedFakeRetriever([hit(i) for i in range(15)])
    config = ask_config(k=5, rerank=False, generated=True, rewrite=False)
    result = SearchTool(retriever, config)("pdb", 10)

    assert retriever.configs[0].candidates >= 10
    assert len(result.stages[0][1]) == 10


def call(id, query):
    return ToolCall(id, "search_docs", json.dumps({"query": query}))


def last(iterable):
    return list(iterable)[-1]


def fake_search(table, log=None):
    def search(query, k):
        if log is not None:
            log.append(query)
        hits = list(table.get(query, []))
        return SearchResult(tuple(hits), (("dense", tuple(hits), 1.0),), 1.0)

    return search


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TickingLLM(ScriptedToolGenerator):
    """Scripted model that spends ``seconds`` of fake time on every call."""

    def __init__(self, turns, clock, seconds):
        super().__init__(turns)
        self.clock = clock
        self.seconds = seconds

    def chat(self, system, messages, tools, allow_tools=True):
        self.clock.now += self.seconds
        return super().chat(system, messages, tools, allow_tools)


def test_registry_numbers_by_first_appearance_and_dedupes_chunks():
    registry = Registry()

    assert registry.add([hit(5), hit(6)]) == [1, 2]
    assert registry.add([hit(6), hit(7)]) == [2, 3]
    assert [h.chunk_id for h in registry.hits] == [5, 6, 7]


def test_direct_answer_without_tools():
    r = last(run_agent("hi", ScriptedToolGenerator([Turn(REFUSAL)]), fake_search({})))

    assert r.answer.refused and r.llm_calls == 1 and r.tool_calls == 0
    assert r.stopped == "answered"


def test_one_search_then_cited_answer():
    llm = ScriptedToolGenerator([Turn("", (call("a", "pdb"),)), Turn("Use a PDB [1].")])

    r = last(run_agent("q", llm, fake_search({"pdb": [hit(7)]})))

    assert [s.number for s in r.answer.sources] == [1]
    assert r.answer.sources[0].hit.chunk_id == 7


def test_numbers_are_global_across_searches_and_dedupe_chunks():
    llm = ScriptedToolGenerator(
        [
            Turn("", (call("a", "deployment"),)),
            Turn("", (call("b", "statefulset"),)),
            Turn("D [1]. S [3]. Both [2]."),
        ]
    )

    r = last(
        run_agent(
            "q", llm, fake_search({"deployment": [hit(1), hit(2)], "statefulset": [hit(2), hit(3)]})
        )
    )

    assert [h.chunk_id for h in r.answer.retrieved] == [1, 2, 3]
    assert "[2]" in llm.calls[2][1][-1]["content"] and "[3]" in llm.calls[2][1][-1]["content"]


def test_citation_to_unseen_passage_is_invalid():
    llm = ScriptedToolGenerator([Turn("", (call("a", "pdb"),)), Turn("X [1]. Y [5].")])

    r = last(run_agent("q", llm, fake_search({"pdb": [hit(1)]})))

    assert r.answer.invalid_citations == (5,)


def test_step_limit_forces_final_answer_without_tools():
    llm = ScriptedToolGenerator(
        [Turn("", (call(str(i), f"q{i}"),)) for i in range(4)] + [Turn("Final [1].")]
    )

    r = last(
        run_agent("q", llm, fake_search({f"q{i}": [hit(i)] for i in range(4)}), Limits(max_steps=4))
    )

    assert r.stopped == "max_steps" and r.tool_calls == 4 and llm.calls[-1][2] is False


def test_bad_arguments_and_unknown_tool_go_back_to_the_model():
    llm = ScriptedToolGenerator(
        [
            Turn("", (ToolCall("a", "search_docs", "{oops"), ToolCall("b", "rm_rf", "{}"))),
            Turn(REFUSAL),
        ]
    )

    r = last(run_agent("q", llm, fake_search({})))

    tool_msgs = [m["content"] for m in llm.calls[1][1] if m["role"] == "tool"]
    assert "invalid JSON" in tool_msgs[0] and "Unknown tool" in tool_msgs[1]
    assert r.tool_calls == 2


def test_duplicate_query_is_not_searched_again():
    searched = []
    llm = ScriptedToolGenerator(
        [Turn("", (call("a", "PDB"),)), Turn("", (call("b", "  pdb "),)), Turn("A [1].")]
    )

    r = list(run_agent("q", llm, fake_search({"PDB": [hit(1)]}, log=searched)))

    assert searched == ["PDB"]
    assert [e.duplicate for e in r if isinstance(e, ToolResultEvent)] == [False, True]


def test_refuses_when_passages_do_not_answer():
    llm = ScriptedToolGenerator([Turn("", (call("a", "gpu quota"),)), Turn(REFUSAL)])

    assert last(run_agent("q", llm, fake_search({"gpu quota": [hit(1)]}))).answer.refused


# Review Focus 1: every tool call in a turn gets a tool entry, even past the step limit.
def test_calls_past_the_step_limit_in_one_turn_still_get_tool_results():
    searched = []
    turn = Turn("", tuple(call(str(i), f"q{i}") for i in range(5)))
    llm = ScriptedToolGenerator([turn, Turn("Final [1].")])

    r = list(
        run_agent(
            "q",
            llm,
            fake_search({f"q{i}": [hit(i)] for i in range(5)}, log=searched),
            Limits(max_steps=4),
        )
    )

    tool_msgs = [m for m in llm.calls[1][1] if m["role"] == "tool"]
    assert [m["id"] for m in tool_msgs] == ["0", "1", "2", "3", "4"]
    assert tool_msgs[4]["content"] == "Step limit reached; no search run."
    assert searched == ["q0", "q1", "q2", "q3"]
    assert sum(isinstance(e, ToolCallEvent) for e in r) == 4
    result = last(r)
    assert result.tool_calls == 4 and result.stopped == "max_steps"


# Review Focus 2: a duplicate search keeps its first numbers, so citing them stays valid.
def test_citing_numbers_from_an_all_duplicate_search_is_valid():
    llm = ScriptedToolGenerator(
        [
            Turn("", (call("a", "deployment"),)),
            Turn("", (call("b", "replicaset"),)),
            Turn("Scaling works [2]."),
        ]
    )

    r = last(
        run_agent("q", llm, fake_search({"deployment": [hit(1), hit(2)], "replicaset": [hit(2)]}))
    )

    assert r.answer.invalid_citations == ()
    assert [s.hit.chunk_id for s in r.answer.sources] == [2]


def test_duplicate_search_reports_first_numbers_and_hits():
    llm = ScriptedToolGenerator(
        [
            Turn("", (call("a", "pdb"),)),
            Turn("", (call("b", "PDB"),)),
            Turn("A [1]."),
        ]
    )

    r = list(run_agent("q", llm, fake_search({"pdb": [hit(4)]})))

    dup = [e for e in r if isinstance(e, ToolResultEvent)][1]
    assert dup.duplicate and dup.numbers == (1,) and dup.hits == (hit(4),)
    assert "Already searched; see passages [1]." in llm.calls[2][1][-1]["content"]


def test_duplicate_of_an_empty_search_says_it_found_nothing():
    llm = ScriptedToolGenerator(
        [Turn("", (call("a", "nothing"),)), Turn("", (call("b", "nothing"),)), Turn(REFUSAL)]
    )

    list(run_agent("q", llm, fake_search({})))

    assert llm.calls[2][1][-1]["content"] == "Already searched; it found no passages."


# Review Focus 3: text next to tool calls is thinking aloud; the loop continues.
def test_text_with_tool_calls_does_not_end_the_loop():
    llm = ScriptedToolGenerator(
        [Turn("Let me look that up.", (call("a", "pdb"),)), Turn("A PDB [1].")]
    )

    r = last(run_agent("q", llm, fake_search({"pdb": [hit(1)]})))

    assert r.stopped == "answered" and r.llm_calls == 2
    assistant = [m for m in llm.calls[1][1] if m["role"] == "assistant"][0]
    assert assistant["text"] == "Let me look that up."


# Review Focus 4: a forced turn that still calls a tool or stays silent is refused.
@pytest.mark.parametrize("forced", [Turn("", (call("z", "x"),)), Turn("   ")])
def test_bad_forced_turn_becomes_the_exact_refusal(forced):
    llm = ScriptedToolGenerator([Turn("", (call("a", "q0"),)), forced])

    r = last(run_agent("q", llm, fake_search({"q0": [hit(1)]}), Limits(max_steps=1)))

    assert r.answer.text == REFUSAL and r.answer.refused
    assert r.stopped == "max_steps" and r.llm_calls == 2
    assert llm.calls[-1][2] is False


def test_forced_turn_answer_keeps_the_stop_reason():
    llm = ScriptedToolGenerator([Turn("", (call("a", "q0"),)), Turn("Done [1].")])

    r = last(run_agent("q", llm, fake_search({"q0": [hit(1)]}), Limits(max_steps=1)))

    assert r.stopped == "max_steps" and not r.answer.refused
    assert [s.number for s in r.answer.sources] == [1]


def test_timeout_forces_final_answer_using_the_clock():
    clock = Clock()
    llm = TickingLLM(
        [Turn("", (call("a", "q0"),)), Turn("", (call("b", "q1"),)), Turn("Partial [1].")],
        clock,
        seconds=50,
    )

    r = last(
        run_agent(
            "q",
            llm,
            fake_search({"q0": [hit(1)], "q1": [hit(2)]}),
            Limits(timeout_s=90),
            clock=clock,
        )
    )

    assert r.stopped == "timeout" and r.tool_calls == 2 and r.llm_calls == 3


def test_token_budget_forces_final_answer():
    llm = ScriptedToolGenerator(
        [
            Turn("", (call("a", "q0"),), input_tokens=80, output_tokens=30),
            Turn("Answer [1].", input_tokens=None, output_tokens=None),
        ]
    )

    r = last(run_agent("q", llm, fake_search({"q0": [hit(1)]}), Limits(max_total_tokens=100)))

    assert r.stopped == "token_budget" and r.llm_calls == 2
    assert [s.number for s in r.answer.sources] == [1]


def test_token_usage_is_summed_and_none_when_never_reported():
    llm = ScriptedToolGenerator(
        [
            Turn("", (call("a", "q0"),), input_tokens=10, output_tokens=2),
            Turn("A [1].", input_tokens=20, output_tokens=None),
        ]
    )

    r = last(run_agent("q", llm, fake_search({"q0": [hit(1)]})))

    assert (r.input_tokens, r.output_tokens) == (30, 2)
    silent = last(run_agent("q", ScriptedToolGenerator([Turn(REFUSAL)]), fake_search({})))
    assert (silent.input_tokens, silent.output_tokens) == (None, None)


def test_daily_budget_exhausted_on_second_call_stops_with_daily_limit():
    budget = iter([True, False])

    r = list(
        run_agent(
            "q",
            ScriptedToolGenerator([Turn("", (call("a", "q0"),)), Turn(REFUSAL)]),
            fake_search({"q0": [hit(1)]}),
            take_budget=lambda: next(budget),
        )
    )

    result = last(r)
    assert result.stopped == "daily_limit" and result.llm_calls == 1
    assert result.answer.text == "" and not result.answer.refused
    assert result.answer.retrieved == (hit(1),)


def test_daily_budget_exhausted_before_forced_turn_stops_with_daily_limit():
    budget = iter([True, False])

    r = last(
        run_agent(
            "q",
            ScriptedToolGenerator([Turn("", (call("a", "q0"),))]),
            fake_search({"q0": [hit(1)]}),
            Limits(max_steps=1),
            take_budget=lambda: next(budget),
        )
    )

    assert r.stopped == "daily_limit" and r.llm_calls == 1


def test_empty_reply_is_retried_and_counts_as_a_step():
    llm = ScriptedToolGenerator([Turn("   "), Turn(REFUSAL)])

    r = last(run_agent("q", llm, fake_search({})))

    assert llm.calls[1][1][-1] == {
        "role": "user",
        "content": "Your reply was empty. Search or answer.",
    }
    assert r.stopped == "answered" and r.steps == ()


def test_empty_reply_uses_up_the_step_limit():
    llm = ScriptedToolGenerator([Turn("  "), Turn("Forced [1].")])

    r = last(run_agent("q", llm, fake_search({}), Limits(max_steps=1)))

    assert r.stopped == "max_steps" and r.llm_calls == 2


def test_events_describe_errors_and_successful_searches():
    llm = ScriptedToolGenerator(
        [
            Turn("", (ToolCall("a", "search_docs", "{oops"), call("b", "pdb"))),
            Turn(REFUSAL),
        ]
    )

    r = list(run_agent("q", llm, fake_search({"pdb": [hit(3)]})))

    bad_call, bad_result, ok_call, ok_result = r[:4]
    assert isinstance(bad_call, ToolCallEvent) and bad_call.arguments is None
    assert bad_call.error and bad_result.error == bad_call.error
    assert ok_call.arguments == {"query": "pdb"} and ok_call.error is None
    assert ok_result.numbers == (1,) and ok_result.hits == (hit(3),)
    assert ok_result.error is None and ok_result.duplicate is False
    assert [e.step for e in r[:4]] == [1, 1, 2, 2]


def test_agent_limits_maps_settings_values():
    from rag.agent.loop import agent_limits
    from rag.config import Settings

    settings = Settings(
        agent_max_steps=3, agent_timeout_s=12.5, agent_max_total_tokens=999, _env_file=None
    )

    assert agent_limits(settings) == Limits(max_steps=3, timeout_s=12.5, max_total_tokens=999)


def test_agent_max_steps_is_bounded_between_one_and_ten():
    from pydantic import ValidationError

    from rag.config import Settings

    with pytest.raises(ValidationError):
        Settings(agent_max_steps=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(agent_max_steps=11, _env_file=None)
