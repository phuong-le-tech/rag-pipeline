import psycopg
from typer.testing import CliRunner

from rag import cli

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
