import json

import pytest

from rag.evals.answers import RetryingGenerator, answers_markdown, evaluate_answers
from rag.evals.golden import GoldenItem
from rag.evals.judge import UnanswerableItem
from rag.generate.prompt import REFUSAL
from rag.retrieval.retriever import RetrieverConfig
from rag.retrieval.search import Hit


def hit(i):
    return Hit(i, 1.0, f"s{i}.md", "u", "T", "H", f"passage {i}", False)


class StubRetriever:
    def __init__(self, hits_by_question):
        self.hits_by_question = hits_by_question

    def retrieve(self, query, config):
        return self.hits_by_question.get(query, [])


class ByQuestionGenerator:
    """Answers depend on which question appears in the user prompt."""

    def __init__(self, answers):
        self.answers = answers

    def generate(self, system, user):
        return next(a for q, a in self.answers.items() if q in user)

    def stream(self, system, user):
        yield self.generate(system, user)


class StubJudge:
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = 0

    def generate(self, system, user):
        self.calls += 1
        return next(v for q, v in self.verdicts.items() if q in user)


GOLDEN = [
    GoldenItem("g1", "roll back?", ("s1.md",), "rollout undo", (), "human"),
    GoldenItem("g2", "cpu usage?", ("s2.md",), "kubectl top pod", (), "human"),
    GoldenItem("g3", "drain?", ("s3.md",), "kubectl drain", (), "human"),
]
UNANSWERABLE = [
    UnanswerableItem("u1", "sourdough?", "human"),
    UnanswerableItem("u2", "lambda timeout?", "human"),
]


def run():
    retriever = StubRetriever(
        {q: [hit(1), hit(2)] for q in ["roll back?", "cpu usage?", "drain?", "lambda timeout?"]}
    )
    generator = ByQuestionGenerator(
        {
            "roll back?": "Use rollout undo [1]. Check status [2].",
            "cpu usage?": "Run kubectl top pod [1]. It lists all pods.",
            "drain?": REFUSAL,
            "lambda timeout?": "Set it in the console [7].",
        }
    )
    judge = StubJudge(
        {
            "roll back?": json.dumps(
                {"claims": [{"text": "a", "supported": True}] * 2, "correct": "yes"}
            ),
            "cpu usage?": json.dumps(
                {
                    "claims": [
                        {"text": "top pod", "supported": True},
                        {"text": "lists all pods", "supported": False},
                    ],
                    "correct": "partial",
                }
            ),
        }
    )
    report = evaluate_answers(
        GOLDEN, UNANSWERABLE, retriever, generator, judge, RetrieverConfig(), sleep=lambda _s: None
    )
    return report, judge


def test_summary_combines_judge_citations_and_refusals():
    report, judge = run()
    s = report.summary

    assert s["answerable_n"] == 3
    assert s["answer_rate"] == pytest.approx(2 / 3)
    assert s["false_refusal_rate"] == pytest.approx(1 / 3)
    assert s["faithfulness"] == pytest.approx((1.0 + 0.5) / 2)
    assert s["correct_yes"] == pytest.approx(1 / 2)
    assert s["correct_partial"] == pytest.approx(1 / 2)
    assert s["citation_coverage"] == pytest.approx((2 / 2 + 1 / 2) / 2)
    assert s["unanswerable_n"] == 2
    assert s["unanswerable_refusal_rate"] == pytest.approx(1 / 2)
    assert s["invalid_citation_rate"] == 0.0
    assert s["judge_failures"] == 0
    assert judge.calls == 2  # refused and unanswerable answers are not judged


def test_results_keep_unsupported_claims_for_failure_analysis():
    report, _ = run()

    cpu = next(r for r in report.results if r.id == "g2")
    assert cpu.unsupported == ("lists all pods",)
    assert cpu.kind == "answerable"
    lam = next(r for r in report.results if r.id == "u2")
    assert lam.refused is False
    assert lam.citations["invalid_citations"] == 1


class SequenceJudge:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, system, user):
        self.calls += 1
        return self.outputs.pop(0)


def test_judge_is_asked_again_when_its_output_has_no_usable_json():
    retriever = StubRetriever({"roll back?": [hit(1)]})
    generator = ByQuestionGenerator({"roll back?": "Undo [1]."})
    valid = json.dumps({"claims": [{"text": "undo", "supported": True}], "correct": "yes"})
    malformed = "Sorry, I cannot grade this answer."  # no JSON at all: repair can't help
    judge = SequenceJudge([malformed, valid])

    report = evaluate_answers(
        GOLDEN[:1], [], retriever, generator, judge, RetrieverConfig(), sleep=lambda _s: None
    )

    assert judge.calls == 2
    assert report.results[0].faithfulness == 1.0
    assert report.summary["judge_failures"] == 0


def test_judge_garbage_counts_as_failure_not_crash():
    retriever = StubRetriever({"roll back?": [hit(1)]})
    generator = ByQuestionGenerator({"roll back?": "Undo [1]."})
    judge = StubJudge({"roll back?": "I refuse to output JSON"})

    report = evaluate_answers(
        GOLDEN[:1], [], retriever, generator, judge, RetrieverConfig(), sleep=lambda _s: None
    )

    assert report.summary["judge_failures"] == 1
    assert report.results[0].faithfulness is None


class FlakyGenerator:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def generate(self, system, user):
        self.calls += 1
        if self.calls == 1:
            raise self.exc
        return "ok"

    def stream(self, system, user):
        yield self.generate(system, user)


def test_retrying_generator_retries_rate_limits():
    import httpx
    import openai

    request = httpx.Request("POST", "https://x")
    exc = openai.RateLimitError("429", response=httpx.Response(429, request=request), body=None)
    inner = FlakyGenerator(exc)
    sleeps = []

    assert RetryingGenerator(inner, sleep=sleeps.append).generate("s", "u") == "ok"
    assert inner.calls == 2
    assert len(sleeps) == 1


def test_answers_markdown_lists_key_metrics():
    report, _ = run()

    table = answers_markdown(report)

    assert "Faithfulness" in table and "Refusal on unanswerable" in table
    assert "0.750" in table


def test_usage_of_unwraps_retrying_generator():
    from rag.generate.generator import ClaudeGenerator, usage_of

    gen = ClaudeGenerator(model="m", max_tokens=8, client=object())

    assert usage_of(RetryingGenerator(gen)) is gen.usage
