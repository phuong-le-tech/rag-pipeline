import contextlib
import json
from dataclasses import replace

import anthropic
import httpx
import openai
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


@pytest.mark.integration
def test_search_rerank_flag_uses_reranker(tmp_path, monkeypatch, test_database_url):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.md").write_text("---\ntitle: X\n---\nrollback steps\n")
    (tmp_path / "a" / "y.md").write_text("---\ntitle: Y\n---\nrollback rollback rollback\n")
    monkeypatch.setenv("RAG_DATABASE_URL", test_database_url)
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "8")
    with psycopg.connect(test_database_url) as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    runner.invoke(cli.app, ["ingest", str(tmp_path)])

    class ReverseReranker:
        def rerank(self, query, hits):
            return [replace(h, score=-i) for i, h in enumerate(reversed(hits))]

    monkeypatch.setattr(cli, "make_reranker", lambda _settings: ReverseReranker())

    plain = runner.invoke(cli.app, ["search", "rollback", "--mode", "keyword", "-k", "2"])
    reranked = runner.invoke(
        cli.app, ["search", "rollback", "--mode", "keyword", "-k", "2", "--rerank"]
    )

    assert reranked.exit_code == 0, reranked.output
    first = [line for line in plain.output.splitlines() if line.startswith("1. ")][0]
    first_reranked = [line for line in reranked.output.splitlines() if line.startswith("1. ")][0]
    assert first.split()[-1] != first_reranked.split()[-1]


@pytest.mark.integration
def test_ask_command_prints_answer_and_cited_sources(tmp_path, monkeypatch, test_database_url):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "rollback.md").write_text(
        "---\ntitle: Rollback\n---\nUse kubectl rollout undo to revert a Deployment.\n"
    )
    monkeypatch.setenv("RAG_DATABASE_URL", test_database_url)
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "8")
    monkeypatch.setenv("RAG_LLM_BACKEND", "fake")
    with psycopg.connect(test_database_url) as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    runner.invoke(cli.app, ["ingest", str(tmp_path)])

    result = runner.invoke(cli.app, ["ask", "how do I revert?", "--no-rerank"])

    assert result.exit_code == 0, result.output
    assert "fake answer" in result.output
    assert "Sources" in result.output
    assert "[1] Rollback" in result.output
    assert "https://kubernetes.io/docs/tasks/rollback/" in result.output


@pytest.mark.integration
def test_ask_agent_prints_steps_answer_and_counters(tmp_path, monkeypatch, test_database_url):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "rollback.md").write_text(
        "---\ntitle: Rollback\n---\nUse kubectl rollout undo to revert a Deployment.\n"
    )
    monkeypatch.setenv("RAG_DATABASE_URL", test_database_url)
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "8")
    monkeypatch.setenv("RAG_LLM_BACKEND", "fake")
    with psycopg.connect(test_database_url) as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    runner.invoke(cli.app, ["ingest", str(tmp_path)])

    result = runner.invoke(cli.app, ["ask", "how do I revert?", "--agent", "--no-rerank"])

    assert result.exit_code == 0, result.output
    assert 'search_docs("rollback", k=5)' in result.output
    assert "fake agent answer" in result.output
    assert "[1] Rollback" in result.output
    assert "stopped: answered" in result.output


def test_ask_agent_error_hint_names_the_agent_backend(monkeypatch):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = anthropic.AuthenticationError(
        "bad key", response=httpx.Response(401, request=request), body=None
    )

    def fail(*_args, **_kwargs):
        raise exc

    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_LLM_BACKEND", "openrouter")
    monkeypatch.setenv("RAG_AGENT_BACKEND", "anthropic")
    monkeypatch.setattr(cli, "connect", lambda _settings: contextlib.nullcontext(object()))
    monkeypatch.setattr(cli, "register_vector", lambda _conn: None)
    monkeypatch.setattr(cli, "Retriever", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "_run_agent_with_trace", fail)

    result = runner.invoke(cli.app, ["ask", "q", "--agent", "--no-rerank"])

    assert result.exit_code == 1
    assert "LLM call failed (anthropic)" in result.output
    assert "ANTHROPIC_API_KEY" in result.output
    assert "OPENROUTER_API_KEY" not in result.output


def test_ask_agent_rejects_rewrite():
    result = runner.invoke(cli.app, ["ask", "q", "--agent", "--rewrite"])

    assert result.exit_code == 2
    assert "cannot be combined with --rewrite" in result.output


def _openai_error(cls, status):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_openai_error(openai.RateLimitError, 429), "rate-limited"),
        (_openai_error(openai.AuthenticationError, 401), "OPENROUTER_API_KEY"),
        (openai.APIConnectionError(request=httpx.Request("POST", "https://x")), "network"),
    ],
)
def test_llm_error_hint_matches_failure_kind(exc, expected):
    settings = Settings(llm_backend="openrouter")

    assert expected in cli.llm_error_hint(exc, settings)


def test_llm_error_hint_daily_free_quota_does_not_suggest_another_free_model():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    exc = openai.RateLimitError(
        "Rate limit exceeded: free-models-per-day",
        response=httpx.Response(429, request=request),
        body={"error": {"message": "Rate limit exceeded: free-models-per-day"}},
    )

    hint = cli.llm_error_hint(exc, Settings(llm_backend="openrouter"))

    assert "daily" in hint
    assert "RAG_OPENROUTER_MODEL" not in hint


def test_llm_error_hint_rate_limit_suggests_switching_openrouter_model():
    hint = cli.llm_error_hint(
        _openai_error(openai.RateLimitError, 429), Settings(llm_backend="openrouter")
    )

    assert "RAG_OPENROUTER_MODEL" in hint


def test_llm_error_hint_agent_rate_limit_names_agent_model_not_llm_model():
    hint = cli.llm_error_hint(
        _openai_error(openai.RateLimitError, 429),
        Settings(llm_backend="openrouter"),
        agent=True,
    )

    assert "RAG_AGENT_MODEL" in hint
    assert "RAG_OPENROUTER_MODEL" not in hint


@pytest.mark.integration
def test_eval_command_prints_table_and_writes_results(tmp_path, monkeypatch, test_database_url):
    corpus = tmp_path / "corpus"
    (corpus / "tasks").mkdir(parents=True)
    (corpus / "tasks" / "rollback.md").write_text(
        "---\ntitle: Rollback\n---\nUse kubectl rollout undo to revert a Deployment.\n"
    )
    (corpus / "CORPUS_REF").write_text("deadbeef\n")
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id": "q1", "question": "revert a deployment", '
        '"relevant_sources": ["tasks/rollback.md"], "origin": "human"}\n'
    )
    monkeypatch.setenv("RAG_DATABASE_URL", test_database_url)
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "8")
    with psycopg.connect(test_database_url) as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    runner.invoke(cli.app, ["ingest", str(corpus)])
    out = tmp_path / "results"

    result = runner.invoke(
        cli.app,
        [
            "eval",
            "--golden",
            str(golden),
            "--config",
            "keyword",
            "--config",
            "dense",
            "--out",
            str(out),
            "--corpus",
            str(corpus),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "| keyword |" in result.output and "| dense |" in result.output
    (saved,) = out.glob("*.json")
    meta = json.loads(saved.read_text())["meta"]
    assert meta["corpus_ref"] == "deadbeef"
    assert meta["n_questions"] == 1
    assert meta["fingerprint"].startswith("fake:fake:8")


@pytest.mark.integration
def test_eval_supports_rewrite_configs(tmp_path, monkeypatch, test_database_url):
    corpus = tmp_path / "corpus"
    (corpus / "tasks").mkdir(parents=True)
    (corpus / "tasks" / "rollback.md").write_text(
        "---\ntitle: Rollback\n---\nUse kubectl rollout undo to revert a Deployment.\n"
    )
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id": "q1", "question": "revert a deployment", '
        '"relevant_sources": ["tasks/rollback.md"], "origin": "human"}\n'
    )
    monkeypatch.setenv("RAG_DATABASE_URL", test_database_url)
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "8")
    monkeypatch.setenv("RAG_LLM_BACKEND", "fake")
    with psycopg.connect(test_database_url) as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    runner.invoke(cli.app, ["ingest", str(corpus)])

    class StubReranker:
        def rerank(self, query, hits):
            return list(hits)

    monkeypatch.setattr(cli, "make_reranker", lambda _s: StubReranker())

    result = runner.invoke(
        cli.app,
        [
            "eval",
            "--golden",
            str(golden),
            "--config",
            "hybrid+rerank, rewrite",
            "--out",
            str(tmp_path / "r"),
            "--corpus",
            str(corpus),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "| hybrid+rerank, rewrite |" in result.output


@pytest.mark.integration
def test_eval_aborts_on_unknown_golden_sources(tmp_path, monkeypatch, test_database_url):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id": "q1", "question": "x", "relevant_sources": ["nope.md"], "origin": "human"}\n'
    )
    monkeypatch.setenv("RAG_DATABASE_URL", test_database_url)
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "8")

    result = runner.invoke(cli.app, ["eval", "--golden", str(golden), "--out", str(tmp_path)])

    assert result.exit_code == 1
    assert "nope.md" in result.output


@pytest.mark.integration
def test_draft_questions_writes_reviewable_jsonl(tmp_path, monkeypatch, test_database_url):
    corpus = tmp_path / "corpus"
    for i, section in enumerate(["concepts", "tasks", "reference"]):
        (corpus / section).mkdir(parents=True)
        body = f"Paragraph {i} about volumes and pods. " * 20
        (corpus / section / f"p{i}.md").write_text(f"---\ntitle: P{i}\n---\n{body}\n")
    monkeypatch.setenv("RAG_DATABASE_URL", test_database_url)
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "8")
    with psycopg.connect(test_database_url) as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    runner.invoke(cli.app, ["ingest", str(corpus)])

    class JsonGenerator:
        def generate(self, system, user):
            return '{"question": "How do scratch volumes behave?", "answer": "See passage."}'

    monkeypatch.setattr(cli, "make_generator", lambda _s: JsonGenerator())
    out = tmp_path / "drafts.jsonl"

    result = runner.invoke(cli.app, ["draft-questions", "--n", "2", "--out", str(out)])

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2
    assert {r["origin"] for r in rows} == {"llm"}
    assert "2 drafts" in result.output


@pytest.mark.integration
def test_eval_answers_command_writes_report(tmp_path, monkeypatch, test_database_url):
    corpus = tmp_path / "corpus"
    (corpus / "tasks").mkdir(parents=True)
    (corpus / "tasks" / "rollback.md").write_text(
        "---\ntitle: Rollback\n---\nUse kubectl rollout undo to revert a Deployment.\n"
    )
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id": "q1", "question": "revert a deployment", "relevant_sources": '
        '["tasks/rollback.md"], "reference_answer": "rollout undo", "origin": "human"}\n'
    )
    unanswerable = tmp_path / "u.jsonl"
    unanswerable.write_text('{"id": "u1", "question": "bake bread", "origin": "human"}\n')
    monkeypatch.setenv("RAG_DATABASE_URL", test_database_url)
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "8")
    monkeypatch.setenv("RAG_LLM_BACKEND", "fake")
    with psycopg.connect(test_database_url) as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    runner.invoke(cli.app, ["ingest", str(corpus)])
    out = tmp_path / "results"

    result = runner.invoke(
        cli.app,
        [
            "eval-answers",
            "--config",
            "keyword",
            "--golden",
            str(golden),
            "--unanswerable",
            str(unanswerable),
            "--out",
            str(out),
            "--corpus",
            str(corpus),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Faithfulness" in result.output
    (saved,) = out.glob("*-answers.json")
    data = json.loads(saved.read_text())
    assert data["meta"]["llm"] == "fake"
    assert data["meta"]["judge"] == "fake"
    assert data["summary"]["answerable_n"] == 1


def test_eval_rejects_unknown_config_name(tmp_path):
    result = runner.invoke(cli.app, ["eval", "--config", "magic", "--out", str(tmp_path)])

    assert result.exit_code == 2
    assert "magic" in result.output


def test_ask_config_uses_measured_defaults():
    config = cli.ask_config(k=5, rerank=True, generated=True, rewrite=False)

    assert config.mode == "hybrid"
    assert config.rerank is True
    assert config.max_per_source == 1  # +0.02 Hit@5 overall, +0.08 on hand-written (n=94)
    assert config.rewrite is False  # mixed evidence + one LLM call per query
    assert config.include_generated is True


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


@pytest.fixture
def answers_run(tmp_path, monkeypatch):
    """eval-answers with no database, network or real judging; records what it was given."""
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id": "q1", "question": "revert a deployment", "relevant_sources": '
        '["tasks/rollback.md"], "reference_answer": "rollout undo", "origin": "human"}\n'
    )
    unanswerable = tmp_path / "u.jsonl"
    unanswerable.write_text('{"id": "u1", "question": "bake bread", "origin": "human"}\n')
    seen: dict = {"meta": None, "agent": None, "evaluate": None}

    class Retriever:
        def __init__(self, *_args, **_kwargs):
            pass

    def spy_agent_answerer(make_llm, search, limits):
        seen["agent"] = (make_llm, search, limits)
        return lambda question: None

    def spy_evaluate(golden_items, refuse_items, answerer, judge, **kwargs):
        seen["evaluate"] = {"answerer": answerer, "judge": judge, **kwargs}
        return "report"

    def spy_write(out_dir, report, meta):
        seen["meta"] = meta
        return out_dir / "saved.json"

    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "8")
    monkeypatch.setenv("RAG_LLM_BACKEND", "fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(cli, "connect", lambda _settings: contextlib.nullcontext(object()))
    monkeypatch.setattr(cli, "register_vector", lambda _conn: None)
    monkeypatch.setattr(cli, "Retriever", Retriever)
    monkeypatch.setattr(cli, "make_reranker", lambda _settings: None)
    monkeypatch.setattr(cli, "make_embedder", lambda _settings: object())
    monkeypatch.setattr(cli, "agent_answerer", spy_agent_answerer)
    monkeypatch.setattr(cli, "evaluate_answers", spy_evaluate)
    monkeypatch.setattr(cli, "answers_markdown", lambda _report: "table")
    monkeypatch.setattr(cli, "write_answer_results", spy_write)

    def invoke(*extra):
        args = [
            "eval-answers",
            "--golden",
            str(golden),
            "--unanswerable",
            str(unanswerable),
            "--out",
            str(tmp_path / "results"),
            "--corpus",
            str(tmp_path),
            *extra,
        ]
        return runner.invoke(cli.app, args)

    return invoke, seen


def test_eval_answers_agent_mode_builds_agent_answerer_with_retrying_llm(answers_run):
    invoke, seen = answers_run

    result = invoke("--mode", "agent", "--k", "3")

    assert result.exit_code == 0, result.output
    make_llm, search, limits = seen["agent"]
    assert isinstance(make_llm(), cli.RetryingGenerator)
    assert limits == cli.agent_limits(Settings())
    assert seen["evaluate"]["answerer"] is not None
    assert seen["meta"]["mode"] == "agent"
    assert seen["meta"]["k"] is None
    assert seen["meta"]["agent_k"] == "model-chosen per call (default 5, max 10)"
    assert seen["meta"]["agent_llm"] == "fake"
    assert seen["meta"]["agent_limits"] == {
        "max_steps": 4,
        "timeout_s": 90.0,
        "max_total_tokens": 30000,
    }


def test_eval_answers_k_sets_the_pipeline_config_k(answers_run):
    invoke, seen = answers_run

    result = invoke("--k", "3")

    assert result.exit_code == 0, result.output
    assert seen["meta"]["mode"] == "pipeline"
    assert seen["meta"]["k"] == 3
    assert "agent_k" not in seen["meta"]


def test_eval_answers_pipeline_mode_is_the_default_and_has_no_agent_meta(answers_run):
    invoke, seen = answers_run

    result = invoke()

    assert result.exit_code == 0, result.output
    assert seen["agent"] is None
    assert seen["meta"]["mode"] == "pipeline"
    assert seen["meta"]["k"] == cli.ABLATIONS["hybrid+rerank"].k
    assert "agent_llm" not in seen["meta"]
    assert "agent_limits" not in seen["meta"]


def test_eval_answers_judge_model_replaces_the_judge_model_and_labels_it(answers_run):
    invoke, seen = answers_run

    result = invoke("--judge-backend", "openrouter", "--judge-model", "judge/x")

    assert result.exit_code == 0, result.output
    assert seen["meta"]["judge"] == "openrouter:judge/x"
    assert seen["evaluate"]["judge"].model == "judge/x"
    assert seen["meta"]["llm"] == "fake"


def test_eval_answers_records_prices_and_passes_them_to_evaluation(answers_run):
    invoke, seen = answers_run

    result = invoke("--price-in", "0.5", "--price-out", "2")

    assert result.exit_code == 0, result.output
    assert seen["meta"]["prices"] == {"in": 0.5, "out": 2.0}
    assert seen["evaluate"]["price_in"] == 0.5
    assert seen["evaluate"]["price_out"] == 2.0


def test_eval_answers_rejects_unknown_mode(answers_run):
    invoke, seen = answers_run

    result = invoke("--mode", "bogus")

    assert result.exit_code == 2
    assert "bogus" in result.output
    assert seen["evaluate"] is None


def test_eval_answers_resume_must_exist(answers_run, tmp_path):
    invoke, seen = answers_run

    result = invoke("--resume", str(tmp_path / "missing.partial.jsonl"))

    assert result.exit_code == 2
    assert seen["evaluate"] is None
