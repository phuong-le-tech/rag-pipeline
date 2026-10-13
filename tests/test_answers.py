import json
from dataclasses import asdict, replace

import pytest

from rag.agent.loop import Limits
from rag.agent.tools import SearchResult
from rag.evals.answers import (
    Answered,
    AnswerResult,
    RetryingGenerator,
    _summary,
    agent_answerer,
    answers_markdown,
    evaluate_answers,
    pipeline_answerer,
)
from rag.evals.golden import GoldenItem
from rag.evals.judge import UnanswerableItem
from rag.generate.answer import Answer
from rag.generate.generator import ScriptedToolGenerator, ToolCall, Turn, Usage
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
        GOLDEN,
        UNANSWERABLE,
        pipeline_answerer(retriever, generator, RetrieverConfig()),
        judge,
        sleep=lambda _s: None,
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
        GOLDEN[:1],
        [],
        pipeline_answerer(retriever, generator, RetrieverConfig()),
        judge,
        sleep=lambda _s: None,
    )

    assert judge.calls == 2
    assert report.results[0].faithfulness == 1.0
    assert report.summary["judge_failures"] == 0


def test_judge_garbage_counts_as_failure_not_crash():
    retriever = StubRetriever({"roll back?": [hit(1)]})
    generator = ByQuestionGenerator({"roll back?": "Undo [1]."})
    judge = StubJudge({"roll back?": "I refuse to output JSON"})

    report = evaluate_answers(
        GOLDEN[:1],
        [],
        pipeline_answerer(retriever, generator, RetrieverConfig()),
        judge,
        sleep=lambda _s: None,
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


class FlakyToolGenerator:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def chat(self, system, messages, tools, allow_tools=True):
        self.calls += 1
        if self.calls == 1:
            raise self.exc
        return Turn("ok")


def test_retrying_generator_chat_retries_rate_limits():
    import httpx
    import openai

    request = httpx.Request("POST", "https://x")
    exc = openai.RateLimitError("429", response=httpx.Response(429, request=request), body=None)
    inner = FlakyToolGenerator(exc)
    sleeps = []

    turn = RetryingGenerator(inner, sleep=sleeps.append).chat("s", [], [])

    assert turn.text == "ok"
    assert inner.calls == 2
    assert len(sleeps) == 1


def search_stub(query, k):
    return SearchResult((hit(1), hit(2)), (), 1.0)


class RecordingJudge:
    def __init__(self, output):
        self.output = output
        self.prompts = []

    def generate(self, system, user):
        self.prompts.append(user)
        return self.output


def test_agent_answerer_records_calls_and_judges_every_passage_seen():
    made = []

    def make_llm():
        made.append(1)
        return ScriptedToolGenerator(
            [
                Turn("", (ToolCall("c1", "search_docs", '{"query": "rollback"}'),)),
                Turn("Use rollout undo [1]. Check status [2]."),
            ]
        )

    judge = RecordingJudge(
        json.dumps({"claims": [{"text": "a", "supported": True}] * 2, "correct": "yes"})
    )
    answerer = agent_answerer(make_llm, search_stub, Limits())

    report = evaluate_answers(GOLDEN[:1], UNANSWERABLE[:1], answerer, judge, sleep=lambda _s: None)

    assert len(made) == 2  # a fresh tool generator per question
    r = report.results[0]
    assert (r.tool_calls, r.llm_calls, r.stopped) == (1, 2, "answered")
    assert r.faithfulness == 1.0
    assert "passage 1" in judge.prompts[0] and "passage 2" in judge.prompts[0]
    assert report.results[1].stopped == "answered"


def result(**overrides):
    base = dict(
        id="x",
        kind="answerable",
        question="q",
        refused=True,  # refused rows need no citation stats, so the helper stays minimal
        answer="a",
        faithfulness=None,
        n_claims=0,
        unsupported=(),
        correct=None,
        citations={},
        latency_ms=1.0,
    )
    return AnswerResult(**(base | overrides))


def test_p95_is_nearest_rank_over_latencies():
    results = [result(latency_ms=float(i)) for i in range(1, 21)]

    assert _summary(results, judged_expected=False)["latency_p95_ms"] == 19.0


def answered(answer, llm_calls=1, tool_calls=0, tokens=(None, None), stopped="pipeline"):
    return Answered(answer, llm_calls, tool_calls, tokens[0], tokens[1], stopped)


ANSWER = Answer("Use it [1].", (), (), False, (hit(1),))
REFUSED = Answer(REFUSAL, (), (), True, ())


def test_cost_and_tokens_average_over_questions_that_reported_them():
    def fake(question):
        if question == "roll back?":
            return answered(ANSWER, tokens=(1000, 500))
        return answered(ANSWER)

    report = evaluate_answers(
        GOLDEN[:1],
        UNANSWERABLE[:1],
        fake,
        None,
        sleep=lambda _s: None,
        price_in=1.0,
        price_out=2.0,
    )

    s = report.summary
    assert s["tokens_mean"] == 1500.0
    assert s["cost_usd_per_question"] == pytest.approx(0.002)
    assert s["llm_calls_mean"] == 1.0
    assert s["tool_calls_mean"] == 0.0


def test_by_origin_splits_summary_and_markdown_by_question_origin():
    golden = [
        GoldenItem("g1", "roll back?", ("s1.md",), None, (), "human"),
        GoldenItem("g2", "llm q?", ("s2.md",), None, (), "llm"),
    ]
    unanswerable = [UnanswerableItem("u1", "bake?", "llm")]
    answers = {"roll back?": ANSWER, "llm q?": REFUSED}

    report = evaluate_answers(
        golden,
        unanswerable,
        lambda q: answered(answers.get(q, REFUSED)),
        None,
        sleep=lambda _s: None,
    )

    assert list(report.by_origin) == ["human", "llm"]
    assert report.by_origin["human"]["answer_rate"] == 1.0
    assert report.by_origin["llm"]["answer_rate"] == 0.0
    assert report.by_origin["llm"]["answerable_n"] == 1
    assert report.by_origin["llm"]["unanswerable_n"] == 1
    assert report.by_origin["human"]["unanswerable_n"] == 0
    table = answers_markdown(report)
    assert "By origin" in table and "| Metric | human | llm |" in table
    assert "p95" in table


class CountingGenerator:
    def __init__(self):
        self.usage = Usage()

    def generate(self, system, user):
        self.usage.add(100, 50)
        return "Use it [1]."

    def stream(self, system, user):
        yield self.generate(system, user)


def test_pipeline_answerer_reports_per_question_token_delta_and_llm_calls():
    retriever = StubRetriever({"roll back?": [hit(1)], "drain?": []})
    answerer = pipeline_answerer(retriever, CountingGenerator(), RetrieverConfig())

    first = answerer("roll back?")
    second = answerer("roll back?")
    refused = answerer("drain?")

    assert (first.input_tokens, first.output_tokens) == (100, 50)
    assert (second.input_tokens, second.output_tokens) == (100, 50)
    assert (first.llm_calls, first.tool_calls, first.stopped) == (1, 0, "pipeline")
    assert refused.answer.refused and refused.llm_calls == 0
    assert refused.input_tokens is None


def test_pipeline_answerer_tokens_are_none_when_provider_reports_nothing():
    retriever = StubRetriever({"roll back?": [hit(1)]})
    generator = ByQuestionGenerator({"roll back?": "Undo [1]."})

    assert (
        pipeline_answerer(retriever, generator, RetrieverConfig())("roll back?").input_tokens
        is None
    )


def _resume_stack():
    retriever = StubRetriever(
        {
            q: [hit(1), hit(2)]
            for q in ["roll back?", "cpu usage?", "drain?", "lambda timeout?", "sourdough?"]
        }
    )
    generator = ByQuestionGenerator(
        {
            "roll back?": "Use rollout undo [1]. Check status [2].",
            "cpu usage?": "Run kubectl top pod [1]. It lists all pods.",
            "drain?": REFUSAL,
            "lambda timeout?": "Set it in the console [7].",
            "sourdough?": REFUSAL,
        }
    )
    verdicts = {
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
    return retriever, generator, verdicts


def _without_latency(report):
    def keep(summary):
        return {k: v for k, v in summary.items() if "latency" not in k}

    results = [replace(r, latency_ms=0.0) for r in report.results]
    return results, keep(report.summary), {o: keep(s) for o, s in report.by_origin.items()}


def test_resume_after_interruption_answers_only_remaining_questions(tmp_path):
    checkpoint = tmp_path / "run.partial.jsonl"
    retriever, generator, verdicts = _resume_stack()
    answerer = pipeline_answerer(retriever, generator, RetrieverConfig())

    def crashing(question):
        if question == "drain?":
            raise RuntimeError("network down")
        return answerer(question)

    judge = StubJudge(verdicts)
    with pytest.raises(RuntimeError):
        evaluate_answers(
            GOLDEN,
            UNANSWERABLE,
            crashing,
            judge,
            sleep=lambda _s: None,
            checkpoint=checkpoint,
        )
    assert judge.calls == 2
    saved = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert [r["id"] for r in saved] == ["g1", "g2"]

    asked = []
    progress = []

    def recording(question):
        asked.append(question)
        return answerer(question)

    resumed_judge = StubJudge(verdicts)
    resumed = evaluate_answers(
        GOLDEN,
        UNANSWERABLE,
        recording,
        resumed_judge,
        sleep=lambda _s: None,
        on_progress=lambda done, total: progress.append((done, total)),
        checkpoint=checkpoint,
    )

    assert asked == ["drain?", "sourdough?", "lambda timeout?"]
    assert resumed_judge.calls == 0  # g1 and g2 were judged before the interruption
    assert progress[-1] == (5, 5)
    saved = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert [r["id"] for r in saved] == ["g1", "g2", "g3", "u1", "u2"]
    assert all(isinstance(r["unsupported"], list) for r in saved)

    uninterrupted = evaluate_answers(
        GOLDEN,
        UNANSWERABLE,
        pipeline_answerer(retriever, generator, RetrieverConfig()),
        StubJudge(verdicts),
        sleep=lambda _s: None,
    )
    assert [r.id for r in resumed.results] == ["g1", "g2", "g3", "u1", "u2"]
    assert _without_latency(resumed) == _without_latency(uninterrupted)


def test_checkpoint_lines_for_unknown_ids_are_ignored(tmp_path):
    checkpoint = tmp_path / "run.partial.jsonl"
    checkpoint.write_text(json.dumps(asdict(result(id="gone"))) + "\n\n")
    retriever, generator, verdicts = _resume_stack()

    report = evaluate_answers(
        GOLDEN[:1],
        [],
        pipeline_answerer(retriever, generator, RetrieverConfig()),
        StubJudge(verdicts),
        sleep=lambda _s: None,
        checkpoint=checkpoint,
    )

    assert [r.id for r in report.results] == ["g1"]
    ids = [json.loads(line)["id"] for line in checkpoint.read_text().splitlines() if line.strip()]
    assert ids == ["gone", "g1"]
