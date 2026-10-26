from itertools import chain, repeat

from langchain_core.messages import AIMessage

from rag.agent.loop import DEFAULT_LIMITS, Limits, ToolResultEvent
from rag.agent.tools import SearchResult
from rag.generate.prompt import REFUSAL
from rag.lc.agent import run_lc_agent
from rag.lc.llm import fake_tool_chat_model
from rag.retrieval.search import Hit


def hit(chunk_id, content):
    url = f"https://kubernetes.io/docs/x{chunk_id}/"
    return Hit(chunk_id, 1.0, f"x{chunk_id}.md", url, "Title", "Title > Part", content, False)


HITS = (hit(1, "Use kubectl rollout undo."), hit(2, "Revisions are kept."))


def search(query, k):
    return SearchResult(HITS, (), 1.0)


def calls_search(query, usage=None):
    return AIMessage(
        content="",
        tool_calls=[{"name": "search_docs", "args": {"query": query}, "id": f"id-{query}"}],
        usage_metadata=usage,
    )


def model(*messages):
    return fake_tool_chat_model(messages)


def run(*messages, limits=DEFAULT_LIMITS, **kwargs):
    return run_lc_agent("how to roll back?", model(*messages), search, limits, **kwargs)


def test_searches_then_answers_with_citation():
    result = run(calls_search("rollback"), AIMessage(content="Use undo [1]."))
    assert result.stopped == "answered"
    assert (result.llm_calls, result.tool_calls) == (2, 1)
    assert [s.number for s in result.answer.sources] == [1]
    assert result.answer.text == "Use undo [1]."


def test_answer_without_search_is_still_returned():
    result = run(AIMessage(content="Just an answer."))
    assert result.stopped == "answered"
    assert (result.llm_calls, result.tool_calls) == (1, 0)
    assert not result.answer.refused


def test_step_limit_stops_a_looping_agent():
    result = run(
        calls_search("q0"),
        calls_search("q1"),
        AIMessage(content="Use undo [1]."),  # the forced final turn, tools off
        limits=Limits(max_steps=2),
    )
    assert result.stopped == "max_steps"
    assert (result.llm_calls, result.tool_calls) == (3, 2)
    assert result.answer.text == "Use undo [1]."


def test_forced_final_turn_that_still_calls_tools_refuses():
    result = run(
        calls_search("q0"), calls_search("q1"), calls_search("q2"), limits=Limits(max_steps=2)
    )
    assert result.stopped == "max_steps"
    assert result.answer.refused and result.answer.text.startswith(REFUSAL)


def test_duplicate_query_reuses_numbers():
    result = run(
        calls_search("Rollback"), calls_search("rollback  "), AIMessage(content="Use undo [1].")
    )
    results = [e for e in result.steps if isinstance(e, ToolResultEvent)]
    assert [e.duplicate for e in results] == [False, True]
    assert results[0].numbers == results[1].numbers == (1, 2)


def test_token_budget_stops_run():
    usage = {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40}
    result = run(
        calls_search("q0", usage),
        AIMessage(content="Use undo [1]."),
        limits=Limits(max_total_tokens=30),
    )
    assert result.stopped == "token_budget"
    assert (result.input_tokens, result.output_tokens) == (30, 10)


def test_timeout_stops_run():
    ticks = chain([0], repeat(5))
    result = run(
        calls_search("q0"),
        AIMessage(content="Use undo [1]."),
        limits=Limits(timeout_s=1),
        clock=lambda: next(ticks),
    )
    assert result.stopped == "timeout"


def test_unparseable_tool_arguments_become_a_tool_error_not_a_crash():
    broken = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "search_docs",
                "args": "{bad",
                "id": "c1",
                "error": "bad json",
                "type": "invalid_tool_call",
            }
        ],
    )
    result = run(broken, AIMessage(content="Fallback answer."))
    errors = [e for e in result.steps if isinstance(e, ToolResultEvent) and e.error]
    assert len(errors) == 1 and result.stopped == "answered"


def test_empty_reply_gets_one_nudge_then_answer():
    result = run(AIMessage(content=""), AIMessage(content="Use undo [1]."))
    assert result.stopped == "answered" and result.llm_calls == 2


def test_parallel_calls_past_the_step_limit_are_not_run():
    three = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_docs", "args": {"query": f"q{i}"}, "id": f"id-{i}"} for i in range(3)
        ],
    )
    result = run(three, AIMessage(content="Use undo [1]."), limits=Limits(max_steps=2))
    assert result.tool_calls == 2 and result.stopped == "max_steps"
