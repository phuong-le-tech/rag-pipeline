import math
from types import SimpleNamespace

import pytest

from rag.embed import FakeEmbedder, OpenAIEmbedder


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def test_fake_embedder_is_deterministic_normalised_and_sized():
    emb = FakeEmbedder(dim=8)

    a1, b = emb.embed(["alpha", "beta"])
    (a2,) = emb.embed(["alpha"])

    assert a1 == a2
    assert a1 != b
    assert len(a1) == 8
    assert norm(a1) == pytest.approx(1.0)


def test_fake_embedder_rejects_bad_dim():
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)


class StubClient:
    """Mimics openai.OpenAI().embeddings.create, returning items shuffled."""

    def __init__(self, dim):
        self.dim = dim
        self.calls = []
        self.embeddings = self

    def create(self, model, input, dimensions):
        self.calls.append(list(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(len(t))] * self.dim)
            for i, t in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)))


def test_openai_embedder_batches_and_preserves_order():
    client = StubClient(dim=3)
    emb = OpenAIEmbedder(dim=3, batch_size=2, client=client)

    vectors = emb.embed(["a", "bb", "ccc"])

    assert client.calls == [["a", "bb"], ["ccc"]]
    assert vectors == [[1.0] * 3, [2.0] * 3, [3.0] * 3]


def test_openai_embedder_empty_input_makes_no_call():
    client = StubClient(dim=3)

    assert OpenAIEmbedder(dim=3, client=client).embed([]) == []
    assert client.calls == []


def test_openai_embedder_rejects_blank_text():
    emb = OpenAIEmbedder(dim=3, client=StubClient(dim=3))

    with pytest.raises(ValueError, match="index 1"):
        emb.embed(["ok", "   "])


def test_openai_embedder_detects_dimension_mismatch():
    emb = OpenAIEmbedder(dim=4, client=StubClient(dim=3))

    with pytest.raises(ValueError, match="expected 4"):
        emb.embed(["a"])
