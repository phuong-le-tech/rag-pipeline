"""Hand-written agent loop: one model turn at a time, search_docs calls, hard limits.

The model never sees a raw search result twice: every passage gets one global number the
first time it appears, so a later ``[n]`` always points at the same chunk.
"""

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

from rag.agent.prompt import AGENT_SYSTEM, FORCE_ANSWER
from rag.agent.tools import SEARCH_DOCS, SearchResult, parse_search_args
from rag.config import Settings
from rag.generate.answer import Answer, finish_answer
from rag.generate.generator import Message, ToolCall, ToolGenerator, Turn
from rag.generate.prompt import REFUSAL, format_numbered
from rag.retrieval.search import Hit

Stop = Literal["answered", "max_steps", "timeout", "token_budget", "daily_limit"]

SEARCH_NAME = SEARCH_DOCS["name"]
STEP_LIMIT_MSG = "Step limit reached; no search run."
EMPTY_REPLY_MSG = "Your reply was empty. Search or answer."
NO_PASSAGES_MSG = "No passages found."


class Registry:
    """Every passage seen in the run, numbered by first appearance (``hits[n-1]`` is ``[n]``)."""

    def __init__(self) -> None:
        self._hits: list[Hit] = []
        self._numbers: dict[int, int] = {}

    def add(self, hits: Sequence[Hit]) -> list[int]:
        numbers = []
        for hit in hits:
            if hit.chunk_id not in self._numbers:
                self._hits.append(hit)
                self._numbers[hit.chunk_id] = len(self._hits)
            numbers.append(self._numbers[hit.chunk_id])
        return numbers

    @property
    def hits(self) -> tuple[Hit, ...]:
        return tuple(self._hits)


@dataclass(frozen=True)
class ToolCallEvent:
    step: int
    name: str
    arguments: dict | None
    error: str | None


@dataclass(frozen=True)
class ToolResultEvent:
    step: int
    name: str
    query: str
    numbers: tuple[int, ...]
    hits: tuple[Hit, ...]
    stages: tuple[tuple[str, tuple[Hit, ...], float], ...]
    ms: float
    error: str | None
    duplicate: bool


AgentEvent = ToolCallEvent | ToolResultEvent


@dataclass(frozen=True)
class Limits:
    max_steps: int = 4
    timeout_s: float = 90.0
    max_total_tokens: int | None = 30000


DEFAULT_LIMITS = Limits()


def agent_limits(settings: Settings) -> Limits:
    return Limits(
        max_steps=settings.agent_max_steps,
        timeout_s=settings.agent_timeout_s,
        max_total_tokens=settings.agent_max_total_tokens,
    )


@dataclass(frozen=True)
class AgentResult:
    answer: Answer
    steps: tuple[AgentEvent, ...]
    llm_calls: int
    tool_calls: int
    input_tokens: int | None  # None when the provider reports nothing
    output_tokens: int | None
    stopped: Stop


@dataclass
class _State:
    messages: list[Message]
    registry: Registry = field(default_factory=Registry)
    seen: dict[str, tuple[int, ...]] = field(default_factory=dict)  # normalised query -> numbers
    events: list[AgentEvent] = field(default_factory=list)
    steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def tokens(self) -> int:
        return (self.input_tokens or 0) + (self.output_tokens or 0)


def run_agent(
    question: str,
    llm: ToolGenerator,
    search: Callable[[str, int], SearchResult],
    limits: Limits = DEFAULT_LIMITS,
    take_budget: Callable[[], bool] = lambda: True,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[AgentEvent | AgentResult]:
    start = clock()
    st = _State(messages=[{"role": "user", "content": question}])
    while True:
        stopped = _limit_reason(st, limits, clock() - start)
        if stopped is not None:
            break
        if not take_budget():
            yield _result(st, _daily_limit_answer(st), "daily_limit")
            return
        turn = _chat(llm, st)
        if not turn.tool_calls:
            if turn.text.strip():
                yield _result(st, finish_answer(turn.text, st.registry.hits), "answered")
                return
            st.messages.append({"role": "user", "content": EMPTY_REPLY_MSG})
            st.steps += 1
            continue
        yield from _run_tool_calls(turn, st, search, limits)

    # Limit reached: one last turn with tools switched off, so the model must answer or refuse.
    if not take_budget():
        yield _result(st, _daily_limit_answer(st), "daily_limit")
        return
    st.messages.append({"role": "user", "content": FORCE_ANSWER})
    turn = _chat(llm, st, allow_tools=False)
    text = REFUSAL if turn.tool_calls or not turn.text.strip() else turn.text
    yield _result(st, finish_answer(text, st.registry.hits), stopped)


def _limit_reason(st: _State, limits: Limits, elapsed: float) -> Stop | None:
    if st.steps >= limits.max_steps:
        return "max_steps"
    if elapsed > limits.timeout_s:
        return "timeout"
    if limits.max_total_tokens is not None and st.tokens > limits.max_total_tokens:
        return "token_budget"
    return None


def _chat(llm: ToolGenerator, st: _State, allow_tools: bool = True) -> Turn:
    turn = llm.chat(AGENT_SYSTEM, st.messages, [SEARCH_DOCS], allow_tools=allow_tools)
    st.llm_calls += 1
    st.input_tokens = _add(st.input_tokens, turn.input_tokens)
    st.output_tokens = _add(st.output_tokens, turn.output_tokens)
    return turn


def _add(total: int | None, value: int | None) -> int | None:
    return total if value is None else (total or 0) + value


def _run_tool_calls(
    turn: Turn, st: _State, search: Callable[[str, int], SearchResult], limits: Limits
) -> Iterator[AgentEvent]:
    # Every tool_use needs a tool_result in the next request, so calls past the limit still get one.
    st.messages.append(
        {"role": "assistant", "text": turn.text, "tool_calls": list(turn.tool_calls)}
    )
    for call in turn.tool_calls:
        if st.steps >= limits.max_steps:
            content, events = STEP_LIMIT_MSG, []
        else:
            content, events = _run_call(call, st, search)
        st.messages.append({"role": "tool", "id": call.id, "content": content})
        for event in events:
            st.events.append(event)
            yield event


def _run_call(
    call: ToolCall, st: _State, search: Callable[[str, int], SearchResult]
) -> tuple[str, list[AgentEvent]]:
    st.steps += 1
    st.tool_calls += 1
    step = st.steps
    arguments = _parse_arguments(call.arguments)
    if call.name != SEARCH_NAME:
        error = f"Unknown tool '{call.name}'. Available: {SEARCH_NAME}."
        return error, _failed(step, call.name, arguments, error)

    parsed = parse_search_args(call.arguments)
    if isinstance(parsed, str):
        return parsed, _failed(step, call.name, arguments, parsed)
    query, k = parsed
    head = ToolCallEvent(step, call.name, arguments, None)

    key = " ".join(query.casefold().split())
    if key in st.seen:
        numbers = st.seen[key]
        if numbers:
            refs = ", ".join(f"[{n}]" for n in numbers)
            content = f"Already searched; see passages {refs}."
        else:
            content = "Already searched; it found no passages."
        hits = tuple(st.registry.hits[n - 1] for n in numbers)
        result = ToolResultEvent(step, call.name, query, numbers, hits, (), 0.0, None, True)
        return content, [head, result]

    found = search(query, k)
    numbers = tuple(st.registry.add(found.hits))
    st.seen[key] = numbers
    content = (
        format_numbered(list(zip(numbers, found.hits, strict=True)))
        if found.hits
        else NO_PASSAGES_MSG
    )
    result = ToolResultEvent(
        step, call.name, query, numbers, found.hits, found.stages, found.ms, None, False
    )
    return content, [head, result]


def _failed(step: int, name: str, arguments: dict | None, error: str) -> list[AgentEvent]:
    return [
        ToolCallEvent(step, name, arguments, error),
        ToolResultEvent(step, name, "", (), (), (), 0.0, error, False),
    ]


def _parse_arguments(raw: str) -> dict | None:
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _daily_limit_answer(st: _State) -> Answer:
    return Answer("", (), (), False, st.registry.hits)


def _result(st: _State, answer: Answer, stopped: Stop) -> AgentResult:
    return AgentResult(
        answer,
        tuple(st.events),
        st.llm_calls,
        st.tool_calls,
        st.input_tokens,
        st.output_tokens,
        stopped,
    )
