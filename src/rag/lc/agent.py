"""LangGraph version of the agent loop in ``rag.agent.loop``.

The graph owns the control flow (model turn -> tool turn -> limit gate -> ...). Tool behaviour
(argument validation, query dedupe, global passage numbering) and stop rules are the
hand-written ones, imported from ``rag.agent.loop``, so only the orchestration differs.
"""

import json
import time
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from rag.agent.loop import (
    EMPTY_REPLY_MSG,
    STEP_LIMIT_MSG,
    AgentResult,
    Limits,
    Stop,
    _add,
    _limit_reason,
    _result,
    _run_call,
    _State,
)
from rag.agent.prompt import AGENT_SYSTEM, FORCE_ANSWER
from rag.agent.tools import SEARCH_DOCS, SearchResult
from rag.generate.answer import finish_answer
from rag.generate.generator import ToolCall
from rag.generate.prompt import REFUSAL

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": SEARCH_DOCS["name"],
        "description": SEARCH_DOCS["description"],
        "parameters": SEARCH_DOCS["input_schema"],
    },
}
GRAPH_STEPS_PER_SEARCH = 3  # agent + tools + gate nodes per tool round trip


def run_lc_agent(
    question: str,
    model: BaseChatModel,
    search: Callable[[str, int], SearchResult],
    limits: Limits,
    clock: Callable[[], float] = time.monotonic,
) -> AgentResult:
    st = _State(messages=[])  # the graph state holds the transcript; _State keeps the counters
    start = clock()
    outcome: dict[str, Any] = {}
    with_tools = model.bind_tools([SEARCH_TOOL])

    def agent(state: MessagesState) -> dict:
        ai = with_tools.invoke([SystemMessage(AGENT_SYSTEM), *state["messages"]])
        _count(st, ai)
        return {"messages": [ai]}

    def route(state: MessagesState) -> str:
        ai = state["messages"][-1]
        if ai.tool_calls or ai.invalid_tool_calls:
            return "tools"
        return "finish" if str(ai.text).strip() else "nudge"

    def tools(state: MessagesState) -> dict:
        ai = state["messages"][-1]
        calls = [ToolCall(c["id"], c["name"], json.dumps(c["args"])) for c in ai.tool_calls]
        # Arguments the model sent as invalid JSON: pass the raw text so the tool reports the error.
        calls += [ToolCall(c["id"], c["name"], c["args"] or "") for c in ai.invalid_tool_calls]
        replies = []
        for call in calls:
            # Every tool call needs a reply, so calls past the limit get one without running.
            if st.steps >= limits.max_steps:
                content, events = STEP_LIMIT_MSG, []
            else:
                content, events = _run_call(call, st, search)
            st.events.extend(events)
            replies.append(ToolMessage(content, tool_call_id=call.id))
        return {"messages": replies}

    def nudge(state: MessagesState) -> dict:
        st.steps += 1  # an empty reply costs a step, as in the hand-written loop
        return {"messages": [HumanMessage(EMPTY_REPLY_MSG)]}

    def gate(state: MessagesState) -> str:
        reason = _limit_reason(st, limits, clock() - start)
        if reason is None:
            return "agent"
        outcome["stopped"] = reason
        return "force"

    def finish(state: MessagesState) -> dict:
        text = str(state["messages"][-1].text)
        outcome["result"] = _result(st, finish_answer(text, st.registry.hits), "answered")
        return {}

    def force(state: MessagesState) -> dict:
        # Limit reached: one last turn with tools off, so the model must answer or refuse.
        ai = model.invoke(
            [SystemMessage(AGENT_SYSTEM), *state["messages"], HumanMessage(FORCE_ANSWER)]
        )
        _count(st, ai)
        stopped: Stop = outcome["stopped"]
        text = REFUSAL if ai.tool_calls or not str(ai.text).strip() else str(ai.text)
        outcome["result"] = _result(st, finish_answer(text, st.registry.hits), stopped)
        return {}

    graph = StateGraph(MessagesState)
    for name, node in [
        ("agent", agent),
        ("tools", tools),
        ("nudge", nudge),
        ("finish", finish),
        ("force", force),
    ]:
        graph.add_node(name, node)
    graph.add_conditional_edges(START, gate, ["agent", "force"])
    graph.add_conditional_edges("agent", route, ["tools", "nudge", "finish"])
    graph.add_conditional_edges("tools", gate, ["agent", "force"])
    graph.add_conditional_edges("nudge", gate, ["agent", "force"])
    graph.add_edge("finish", END)
    graph.add_edge("force", END)

    recursion_limit = GRAPH_STEPS_PER_SEARCH * (limits.max_steps + 2) + 5
    graph.compile().invoke(
        {"messages": [HumanMessage(question)]}, {"recursion_limit": recursion_limit}
    )
    return outcome["result"]


def _count(st: _State, ai: AIMessage | BaseMessage) -> None:
    st.llm_calls += 1
    usage = getattr(ai, "usage_metadata", None) or {}
    st.input_tokens = _add(st.input_tokens, usage.get("input_tokens"))
    st.output_tokens = _add(st.output_tokens, usage.get("output_tokens"))
