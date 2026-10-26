import importlib.util

import pytest

from rag.config import Settings


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_default_impl_is_handwritten():
    assert settings().impl == "handwritten"


def test_openrouter_backend_builds_chat_openai():
    from langchain_openai import ChatOpenAI

    from rag.lc.llm import make_chat_model

    s = settings(llm_backend="openrouter", openrouter_model="vendor/m", OPENROUTER_API_KEY="k")
    model = make_chat_model(s)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "vendor/m"
    assert model.openai_api_base == s.openrouter_base_url


def test_anthropic_backend_is_rejected_with_a_clear_message():
    from rag.lc.llm import make_chat_model

    with pytest.raises(ValueError, match="openrouter"):
        make_chat_model(settings(llm_backend="anthropic"))


def test_fake_backend_returns_fake_list_chat_model():
    from langchain_core.language_models import FakeListChatModel

    from rag.lc.llm import make_chat_model

    assert isinstance(make_chat_model(settings(llm_backend="fake")), FakeListChatModel)


def test_require_langchain_message(monkeypatch):
    from rag.lc.llm import require_langchain

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match=r"LangChain is not installed; run: uv sync --extra lan"):
        require_langchain()
