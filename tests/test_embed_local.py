import numpy as np
import pytest

from rag.config import Settings
from rag.embed import FakeEmbedder, LocalEmbedder, OpenAIEmbedder, make_embedder


class StubModel:
    """Mimics fastembed.TextEmbedding: generators of numpy arrays."""

    def __init__(self, dim):
        self.dim = dim
        self.passage_calls = []
        self.query_calls = []

    def passage_embed(self, texts, batch_size):
        self.passage_calls.append((list(texts), batch_size))
        for t in texts:
            yield np.full(self.dim, float(len(t)), dtype=np.float32)

    def query_embed(self, query):
        self.query_calls.append(query)
        yield np.full(self.dim, -1.0, dtype=np.float32)


def test_local_embedder_uses_passage_mode_and_returns_lists():
    model = StubModel(dim=3)
    emb = LocalEmbedder(dim=3, batch_size=16, model=model)

    vectors = emb.embed(["a", "bb"])

    assert vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    assert all(type(x) is float for x in vectors[0])
    assert model.passage_calls == [(["a", "bb"], 16)]


def test_local_embedder_query_uses_query_mode():
    model = StubModel(dim=3)

    vector = LocalEmbedder(dim=3, model=model).embed_query("how to roll back")

    assert vector == [-1.0, -1.0, -1.0]
    assert model.query_calls == ["how to roll back"]


def test_local_embedder_empty_input_skips_model():
    model = StubModel(dim=3)

    assert LocalEmbedder(dim=3, model=model).embed([]) == []
    assert model.passage_calls == []


def test_local_embedder_rejects_blank_text_and_dim_mismatch():
    with pytest.raises(ValueError, match="index 0"):
        LocalEmbedder(dim=3, model=StubModel(dim=3)).embed([" "])
    with pytest.raises(ValueError, match="expected 4"):
        LocalEmbedder(dim=4, model=StubModel(dim=3)).embed(["a"])


def test_fake_query_matches_passage_vector():
    emb = FakeEmbedder(dim=4)

    assert emb.embed_query("x") == emb.embed(["x"])[0]


@pytest.mark.parametrize(
    ("backend", "cls"),
    [("local", LocalEmbedder), ("openai", OpenAIEmbedder), ("fake", FakeEmbedder)],
)
def test_make_embedder_picks_backend(backend, cls):
    settings = Settings(embedding_backend=backend, embedding_dim=384)

    emb = make_embedder(settings)

    assert isinstance(emb, cls)
    assert emb.dim == 384


def test_make_embedder_passes_thread_cap_to_local_model():
    emb = make_embedder(Settings(embedding_backend="local", embedding_threads=6))

    assert emb.threads == 6


def test_settings_default_to_local_bge_small():
    settings = Settings(_env_file=None)

    assert settings.embedding_backend == "local"
    assert settings.local_embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.embedding_dim == 384
