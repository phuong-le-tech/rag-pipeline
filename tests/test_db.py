import pytest

from rag.db import render_schema


def test_render_schema_substitutes_embedding_dim():
    sql = render_schema(768)

    assert "VECTOR(768)" in sql
    assert "{embedding_dim}" not in sql
    assert "DEFAULT '{}'::jsonb" in sql


@pytest.mark.parametrize("bad", [0, -1, "1536", 1.5])
def test_render_schema_rejects_invalid_dim(bad):
    with pytest.raises(ValueError):
        render_schema(bad)
