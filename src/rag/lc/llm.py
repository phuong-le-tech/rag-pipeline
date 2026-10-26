"""LangChain chat model built from the same settings as ``make_generator``."""

import importlib.util
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from rag.config import Settings
from rag.generate.generator import FAKE_ANSWER

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


LANGCHAIN_BACKENDS = ("openrouter", "fake")  # langchain-anthropic cannot install here yet


def require_langchain() -> None:
    if importlib.util.find_spec("langchain_core") is None:
        raise RuntimeError("LangChain is not installed; run: uv sync --extra langchain")


def make_chat_model(settings: Settings, max_retries: int | None = None) -> "BaseChatModel":
    """``max_retries`` is the SDK's 429 backoff: evals raise it, the web path keeps the default."""
    require_langchain()
    match settings.llm_backend:
        case "fake":
            from langchain_core.language_models import FakeListChatModel

            return FakeListChatModel(responses=[FAKE_ANSWER])
        case "openrouter":
            from langchain_openai import ChatOpenAI

            key = settings.openrouter_api_key
            return ChatOpenAI(
                model=settings.require_openrouter_model(),
                base_url=settings.openrouter_base_url,
                api_key=key.get_secret_value() if key else None,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout_s,
                **({} if max_retries is None else {"max_retries": max_retries}),
            )
    # shortcut: langchain-anthropic needs anthropic<2 (project pins >=2.1), so no anthropic yet.
    raise ValueError(
        f"llm_backend {settings.llm_backend!r} is not available with impl=langchain; "
        f"use one of: {', '.join(LANGCHAIN_BACKENDS)}"
    )


def fake_tool_chat_model(messages: Iterable[Any]) -> "BaseChatModel":
    """Scripted chat model that accepts ``bind_tools``; offline runs and tests (like
    ``ScriptedToolGenerator``). Replies are the given ``AIMessage``s, one per model call."""
    require_langchain()
    from langchain_core.language_models import GenericFakeChatModel

    class FakeToolChatModel(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolChatModel":
            return self

    return FakeToolChatModel(messages=iter(messages))
