import psycopg
import pytest
from typer.testing import CliRunner

from rag import cli
from rag.config import Settings
from rag.ingest.loaders import LOADER_VERSION

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


def test_pipeline_fingerprint_reflects_backend_model_dim_chunk_size_and_loader():
    settings = Settings(embedding_backend="local", embedding_dim=384, chunk_max_chars=1500)

    assert settings.pipeline_fingerprint() == (
        f"local:BAAI/bge-small-en-v1.5:384:1500:loader{LOADER_VERSION}"
    )
    assert Settings(embedding_backend="fake", embedding_dim=8).pipeline_fingerprint() == (
        f"fake:fake:8:1500:loader{LOADER_VERSION}"
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


@pytest.mark.integration
def test_search_command_prints_ranked_hits(tmp_path, monkeypatch, test_database_url):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "rollback.md").write_text(
        "---\ntitle: Rollback\n---\nUse kubectl rollout undo to revert a Deployment.\n"
    )
    monkeypatch.setenv("RAG_DATABASE_URL", test_database_url)
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "8")
    with psycopg.connect(test_database_url) as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    runner.invoke(cli.app, ["ingest", str(tmp_path)])

    result = runner.invoke(cli.app, ["search", "reverting", "--mode", "keyword", "-k", "3"])

    assert result.exit_code == 0, result.output
    assert "1. " in result.output
    assert "Rollback" in result.output
    assert "https://kubernetes.io/docs/tasks/rollback/" in result.output


def test_search_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")

    result = runner.invoke(cli.app, ["search", "anything", "--mode", "magic"])

    assert result.exit_code == 2
    assert "mode" in result.output


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
