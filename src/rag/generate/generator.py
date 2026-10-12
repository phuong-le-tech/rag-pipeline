"""LLM wrappers behind one small protocol. Kept deliberately thin: no framework."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag.config import Settings
from rag.generate.prompt import REFUSAL

FAKE_ANSWER = "This is a fake answer that cites the first passage [1]."


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text; the loop parses it so bad JSON becomes a tool error


@dataclass(frozen=True)
class Turn:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None


# Transcript entries (plain dicts, provider-neutral):
#   {"role": "user", "content": str}
#   {"role": "assistant", "text": str, "tool_calls": list[ToolCall]}
#   {"role": "tool", "id": str, "content": str}
Message = dict[str, Any]


class Generator(Protocol):
    def generate(self, system: str, user: str) -> str: ...

    def stream(self, system: str, user: str) -> Iterator[str]: ...


class FakeGenerator:
    """Returns a canned answer and records prompts. For tests and offline runs."""

    def __init__(self, answer: str = FAKE_ANSWER) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answer

    def stream(self, system: str, user: str) -> Iterator[str]:
        yield self.generate(system, user)


class ClaudeGenerator:
    """Anthropic Messages API. Reads ``ANTHROPIC_API_KEY`` from the environment by default."""

    def __init__(
        self,
        model: str,
        max_tokens: int,
        api_key: str | None = None,
        timeout: float | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._api_key = api_key
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            # api_key=None falls back to the ANTHROPIC_API_KEY environment variable.
            self._client = Anthropic(api_key=self._api_key, timeout=self.timeout)
        return self._client

    def _request(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

    def generate(self, system: str, user: str) -> str:
        response = self.client.messages.create(**self._request(system, user))
        return "".join(block.text for block in response.content if block.type == "text")

    def stream(self, system: str, user: str) -> Iterator[str]:
        with self.client.messages.stream(**self._request(system, user)) as stream:
            yield from stream.text_stream

    def chat(
        self, system: str, messages: list[Message], tools: list[dict], allow_tools: bool = True
    ) -> Turn:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": _anthropic_messages(messages),
        }
        if tools:
            request["tools"] = _anthropic_tools(tools)
            if not allow_tools:
                request["tool_choice"] = {"type": "none"}
        response = self.client.messages.create(**request)
        text = "".join(block.text for block in response.content if block.type == "text")
        calls = tuple(
            ToolCall(block.id, block.name, json.dumps(block.input))
            for block in response.content
            if block.type == "tool_use"
        )
        usage = response.usage
        return Turn(
            text,
            calls,
            usage.input_tokens if usage else None,
            usage.output_tokens if usage else None,
        )


class OpenAICompatibleGenerator:
    """Any OpenAI-style Chat Completions endpoint: OpenRouter, Groq, Gemini, Ollama, vLLM...

    Only ``base_url`` and ``model`` change between providers.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = base_url
        self._api_key = api_key
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url, api_key=self._api_key, timeout=self.timeout
            )
        return self._client

    def _request(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(**self._request(system, user))
        return response.choices[0].message.content or ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        chunks = self.client.chat.completions.create(**self._request(system, user), stream=True)
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat(
        self, system: str, messages: list[Message], tools: list[dict], allow_tools: bool = True
    ) -> Turn:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": _openai_messages(system, messages),
        }
        if tools:
            request["tools"] = _openai_tools(tools)
            if not allow_tools:
                request["tool_choice"] = "none"
        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        calls = tuple(
            ToolCall(call.id, call.function.name, call.function.arguments)
            for call in message.tool_calls or []
        )
        usage = response.usage
        return Turn(
            message.content or "",
            calls,
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
        )


class ToolGenerator(Protocol):
    """Tool-calling chat. Tool specs are neutral ``{"name", "description", "input_schema"}``."""

    def chat(
        self, system: str, messages: list[Message], tools: list[dict], allow_tools: bool = True
    ) -> Turn: ...


class ScriptedToolGenerator:
    """Replays scripted turns in order, then refuses. For tests and offline runs."""

    def __init__(self, turns: Sequence[Turn]) -> None:
        self._turns = list(turns)
        self.calls: list[tuple[str, list[Message], bool]] = []

    def chat(
        self, system: str, messages: list[Message], tools: list[dict], allow_tools: bool = True
    ) -> Turn:
        self.calls.append((system, list(messages), allow_tools))
        if not self._turns:
            return Turn(REFUSAL)
        return self._turns.pop(0)


def _anthropic_tools(tools: list[dict]) -> list[dict]:
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        for t in tools
    ]


def _openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _anthropic_messages(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m["role"] == "assistant":
            blocks: list[dict] = [{"type": "text", "text": m["text"]}] if m["text"].strip() else []
            for call in m["tool_calls"]:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": _parse_arguments(call.arguments),
                    }
                )
            out.append({"role": "assistant", "content": blocks})
            continue
        # Tool results and nudges share one user message so roles keep alternating.
        if m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["id"], "content": m["content"]}
        else:
            block = {"type": "text", "text": m["content"]}
        if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
            out[-1]["content"].append(block)
        elif m["role"] == "tool":
            out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


def _openai_messages(system: str, messages: list[Message]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m["text"] or None}
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in m["tool_calls"]
            ]
            out.append(entry)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["id"], "content": m["content"]})
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


def _parse_arguments(arguments: str) -> Any:
    # History is ours, but a model can emit bad JSON; the loop reports that as a tool error.
    # Anthropic's tool_use.input must be an object, so any other JSON value also becomes {}.
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _secret(value: Any) -> str | None:
    return value.get_secret_value() if value else None


def make_generator(settings: Settings) -> Generator:
    match settings.llm_backend:
        case "fake":
            return FakeGenerator()
        case "openrouter":
            return OpenAICompatibleGenerator(
                model=settings.openrouter_model,
                max_tokens=settings.llm_max_tokens,
                base_url=settings.openrouter_base_url,
                api_key=_secret(settings.openrouter_api_key),
                timeout=settings.llm_timeout_s,
            )
    return ClaudeGenerator(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        api_key=_secret(settings.anthropic_api_key),
        timeout=settings.llm_timeout_s,
    )


def make_tool_generator(settings: Settings) -> ToolGenerator:
    backend = settings.agent_backend or settings.llm_backend
    match backend:
        case "fake":
            return ScriptedToolGenerator(
                [
                    Turn("", (ToolCall("fake-1", "search_docs", '{"query": "rollback"}'),)),
                    Turn("This is a fake agent answer [1]."),
                ]
            )
        case "openrouter":
            return OpenAICompatibleGenerator(
                model=settings.agent_model or settings.openrouter_model,
                max_tokens=settings.llm_max_tokens,
                base_url=settings.openrouter_base_url,
                api_key=_secret(settings.openrouter_api_key),
                timeout=settings.llm_timeout_s,
            )
    return ClaudeGenerator(
        model=settings.agent_model or settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        api_key=_secret(settings.anthropic_api_key),
        timeout=settings.llm_timeout_s,
    )
