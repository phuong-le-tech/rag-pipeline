import psycopg
import pytest
from typer.testing import CliRunner

from rag import cli
from rag.config import Settings

runner = CliRunner()


def test_chunk_reports_counts_for_folder(tmp_path):
    (tmp_path / "a.md").write_text("# A\none\n\n# B\ntwo\n")
    (tmp_path / "skip.pdf").write_bytes(b"%PDF")

    result = runner.invoke(cli.app, ["chunk", str(tmp_path), "--preview"])

    assert result.exit_code == 0
    assert "a.md: 2 chunks" in result.output
    assert "[1] B | two" in result.output
    assert "1 docs, 2 chunks" in result.output


def test_chunk_exits_nonzero_when_no_supported_files(tmp_path):
    (tmp_path / "x.pdf").write_bytes(b"%PDF")

    result = runner.invoke(cli.app, ["chunk", str(tmp_path)])

    assert result.exit_code == 1


def test_init_db_reports_unreachable_database(monkeypatch):
    def boom(_settings):
        raise psycopg.OperationalError("refused")

    monkeypatch.setattr(cli, "init_db", boom)

    result = runner.invoke(cli.app, ["init-db"])

    assert result.exit_code == 1
    assert "docker compose up -d" in result.output


def test_init_db_success(monkeypatch):
    monkeypatch.setattr(cli, "init_db", lambda _settings: None)

    result = runner.invoke(cli.app, ["init-db"])

    assert result.exit_code == 0
    assert "Schema ready." in result.output


def test_pipeline_fingerprint_reflects_backend_model_dim_and_chunk_size():
    settings = Settings(embedding_backend="local", embedding_dim=384, chunk_max_chars=1500)

    assert settings.pipeline_fingerprint() == "local:BAAI/bge-small-en-v1.5:384:1500"
    assert Settings(embedding_backend="fake", embedding_dim=8).pipeline_fingerprint() == (
        "fake:fake:8:1500"
    )


@pytest.mark.integration
def test_ingest_command_indexes_then_skips(tmp_path, monkeypatch, test_database_url):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.md").write_text("---\ntitle: One\n---\nalpha\n\n## Sub\nbeta\n")
    monkeypatch.setenv("RAG_DATABASE_URL", test_database_url)
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "8")
    with psycopg.connect(test_database_url) as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")

    first = runner.invoke(cli.app, ["ingest", str(tmp_path)])
    second = runner.invoke(cli.app, ["ingest", str(tmp_path)])

    assert first.exit_code == 0, first.output
    assert "added 1, updated 0, skipped 0, deleted 0 (2 chunks)" in first.output
    assert "added 0, updated 0, skipped 1, deleted 0 (0 chunks)" in second.output


def test_ingest_reports_unreachable_database(tmp_path, monkeypatch):
    (tmp_path / "one.md").write_text("---\ntitle: One\n---\nalpha\n")
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")

    def boom(_settings):
        raise psycopg.OperationalError("refused")

    monkeypatch.setattr(cli, "connect", boom)

    result = runner.invoke(cli.app, ["ingest", str(tmp_path)])

    assert result.exit_code == 1
    assert "docker compose up -d" in result.output


def test_ingest_exits_when_no_documents(tmp_path):
    result = runner.invoke(cli.app, ["ingest", str(tmp_path)])

    assert result.exit_code == 1
    assert "No documents" in result.output
