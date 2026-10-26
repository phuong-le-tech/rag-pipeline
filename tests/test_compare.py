import json
import math

import pytest
from typer.testing import CliRunner

from rag import cli
from rag.evals.compare import compare_markdown, count_lines

runner = CliRunner(env={"NO_COLOR": "1", "COLUMNS": "200"})


def result(impl, mode="pipeline", **summary):
    base = {
        "faithfulness": 0.88,
        "latency_p50_ms": 1000.0,
        "tokens_mean": 5000.0,
        "llm_calls_mean": 1.0,
    }
    meta = {"mode": mode, "llm": "openrouter:m", "retrieval": "hybrid+rerank"}
    if impl is not None:
        meta["impl"] = impl
    return {"meta": meta, "summary": {**base, **summary}}


LOC = {"handwritten": 700, "langchain": 450}


def test_table_has_a_row_per_metric_and_both_columns():
    table = compare_markdown(
        result("handwritten"), result("langchain", faithfulness=0.87, latency_p50_ms=1200.0), LOC
    )
    lines = table.splitlines()
    assert lines[0] == "| Metric | Hand-written | LangChain |"
    assert "| Faithfulness (supported claims) | 0.880 | 0.870 |" in lines
    assert "| Latency p50 (ms) | 1000 | 1200 |" in lines
    assert "| Lines of code | 700 | 450 |" in lines


def test_missing_token_counts_show_a_dash():
    table = compare_markdown(
        result("handwritten", tokens_mean=math.nan), result("langchain", tokens_mean=None), LOC
    )
    assert "| Tokens per question | - | - |" in table.splitlines()


def test_results_saved_before_impl_existed_count_as_handwritten():
    assert "Hand-written" in compare_markdown(result(None), result("langchain"), LOC)


def test_refuses_two_results_of_the_same_implementation():
    with pytest.raises(ValueError, match="one hand-written and one langchain"):
        compare_markdown(result("langchain"), result("langchain"), LOC)


def test_refuses_to_compare_different_modes():
    with pytest.raises(ValueError, match="mode"):
        compare_markdown(result("handwritten"), result("langchain", mode="agent"), LOC)


def test_count_lines_skips_blank_lines(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n\n\ny = 2\n")
    b.write_text("z = 3\n")
    assert count_lines([a, b]) == 3


def test_compare_command_prints_the_table_whatever_the_argument_order(tmp_path):
    hand, lc = tmp_path / "h.json", tmp_path / "l.json"
    hand.write_text(json.dumps(result("handwritten")))
    lc.write_text(json.dumps(result("langchain")))

    result_cli = runner.invoke(cli.app, ["compare", str(lc), str(hand)])

    assert result_cli.exit_code == 0, result_cli.output
    assert "| Metric | Hand-written | LangChain |" in result_cli.output
    assert "Lines of code" in result_cli.output
