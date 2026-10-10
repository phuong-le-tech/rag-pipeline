"""LLM wrappers behind one small protocol. Kept deliberately thin: no framework."""

from collections.abc import Iterator
from typing import Any, Protocol

from rag.config import Settings

FAKE_ANSWER = "This is a fake answer that cites the first passage [1]."


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
        self, model: str, max_tokens: int, api_key: str | None = None, client: Any = None
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            # api_key=None falls back to the ANTHROPIC_API_KEY environment variable.
            self._client = Anthropic(api_key=self._api_key)
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
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._api_key = api_key
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self._api_key)
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
            )
    return ClaudeGenerator(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        api_key=_secret(settings.anthropic_api_key),
    )
