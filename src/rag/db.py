from importlib.resources import files

import psycopg

from rag.config import Settings


def connect(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.database_url)


def render_schema(embedding_dim: int) -> str:
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be a positive int, got {embedding_dim!r}")
    template = files("rag").joinpath("schema.sql").read_text(encoding="utf-8")
    return template.format(embedding_dim=embedding_dim)


def init_db(settings: Settings) -> None:
    with connect(settings) as conn:
        conn.execute(render_schema(settings.embedding_dim))
