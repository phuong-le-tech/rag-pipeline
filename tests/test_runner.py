import json

import pytest

from rag.evals.golden import GoldenItem
from rag.evals.runner import (
    ABLATIONS,
    breakdown_markdown,
    evaluate,
    to_markdown,
    write_results,
)
from rag.retrieval.retriever import RetrieverConfig
from rag.retrieval.search import Hit


def hit(source):
    return Hit(0, 1.0, source, "u", "T", "H", "c", False)


class ScriptedRetriever:
    """Returns a fixed ranking per question; records the configs it was called with."""

    def __init__(self, rankings):
        self.rankings = rankings
        self.configs = []

    def retrieve(self, query, config):
        self.configs.append(config)
        return [hit(s) for s in self.rankings[query]][: config.k]


ITEMS = [
    GoldenItem("q1", "first?", ("a.md",), None, ("howto",), "human"),
    GoldenItem("q2", "second?", ("b.md",), None, ("concept",), "llm"),
]


def test_evaluate_scores_each_question_and_averages():
    retriever = ScriptedRetriever({"first?": ["a.md", "x.md"], "second?": ["x.md", "y.md", "b.md"]})

    report = evaluate(retriever, ITEMS, RetrieverConfig(mode="dense", k=5), name="dense")

    assert report.name == "dense"
    assert [r.id for r in report.results] == ["q1", "q2"]
    assert report.results[0].ranked_sources == ("a.md", "x.md")
    assert report.summary["hit@1"] == pytest.approx(0.5)
    assert report.summary["hit@3"] == pytest.approx(1.0)
    assert report.summary["mrr"] == pytest.approx((1 + 1 / 3) / 2)
    assert report.summary["n"] == 2
    assert report.summary["latency_p50_ms"] >= 0


def test_evaluate_retrieves_deep_enough_for_largest_k():
    retriever = ScriptedRetriever({"first?": ["a.md"] * 20, "second?": ["b.md"] * 20})

    evaluate(retriever, ITEMS, RetrieverConfig(k=5, candidates=5), name="x")

    assert {c.k for c in retriever.configs} == {10}
    assert all(c.candidates >= 10 for c in retriever.configs)


def test_evaluate_rejects_empty_golden_set():
    with pytest.raises(ValueError):
        evaluate(ScriptedRetriever({}), [], RetrieverConfig(), name="x")


def test_ablations_cover_modes_rerank_and_generated_filter():
    configs = ABLATIONS.values()

    assert {c.mode for c in configs} == {"dense", "keyword", "hybrid"}
    assert any(c.rerank for c in configs)
    assert any(not c.include_generated for c in configs)


def test_write_results_saves_json_with_meta_and_per_question_rankings(tmp_path):
    retriever = ScriptedRetriever({"first?": ["a.md"], "second?": ["x.md", "b.md"]})
    report = evaluate(retriever, ITEMS, RetrieverConfig(mode="dense"), name="dense")
    meta = {"timestamp": "2026-10-10T18:30:00Z", "corpus_ref": "abc123"}

    path = write_results(tmp_path, [report], meta)

    data = json.loads(path.read_text())
    assert path.name == "2026-10-10T18-30-00Z.json"
    assert data["meta"] == meta
    (saved,) = data["reports"]
    assert saved["name"] == "dense"
    assert saved["config"]["mode"] == "dense"
    assert saved["summary"]["mrr"] == pytest.approx(0.75)
    assert saved["results"][1]["ranked_sources"] == ["x.md", "b.md"]
    assert (tmp_path / "2026-10-10T18-30-00Z.md").read_text().startswith("| Config |")


def test_breakdown_markdown_splits_metrics_by_group():
    retriever = ScriptedRetriever({"first?": ["a.md"], "second?": ["x.md", "b.md"]})
    report = evaluate(retriever, ITEMS, RetrieverConfig(mode="dense"), name="dense")

    table = breakdown_markdown([report], ITEMS, group=lambda item: item.origin)

    lines = table.strip().splitlines()
    assert lines[0] == "| Config | human Hit@5 | human MRR | llm Hit@5 | llm MRR |"
    assert lines[2] == "| dense | 1.000 | 1.000 | 1.000 | 0.500 |"


def test_to_markdown_renders_one_row_per_config():
    retriever = ScriptedRetriever({"first?": ["a.md"], "second?": ["b.md"]})
    reports = [
        evaluate(retriever, ITEMS, RetrieverConfig(mode="dense"), name="dense"),
        evaluate(retriever, ITEMS, RetrieverConfig(mode="keyword"), name="keyword"),
    ]

    table = to_markdown(reports)

    lines = table.strip().splitlines()
    assert lines[0].startswith("| Config |")
    assert "Recall@5" in lines[0] and "MRR" in lines[0]
    assert lines[2].startswith("| dense |") and "1.000" in lines[2]
    assert len(lines) == 4
