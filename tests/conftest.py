import os

import psycopg
import pytest
from pgvector.psycopg import register_vector

from rag.db import render_schema

TEST_DIM = 8
TEST_DB = "rag_test"
ADMIN_URL = os.environ.get("RAG_TEST_ADMIN_URL", "postgresql://rag:rag@localhost:5433/rag")
TEST_URL = os.environ.get("RAG_TEST_DATABASE_URL", f"postgresql://rag:rag@localhost:5433/{TEST_DB}")


@pytest.fixture(scope="session")
def test_database_url():
    """Create the throwaway rag_test database once; skip DB tests if Postgres is down."""
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3) as admin:
            exists = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,))
            if exists.fetchone() is None:
                admin.execute(f'CREATE DATABASE "{TEST_DB}"')
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable ({exc}); run: docker compose up -d")
    with psycopg.connect(TEST_URL) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, documents")
        conn.execute(render_schema(TEST_DIM))
    return TEST_URL


@pytest.fixture
def conn(test_database_url):
    with psycopg.connect(test_database_url) as connection:
        register_vector(connection)
        connection.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
        connection.commit()
        yield connection
