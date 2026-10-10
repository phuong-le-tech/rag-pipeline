"""End-to-end answer evaluation: generate, judge, check citations, measure refusals."""

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.evals.drafting import with_retry
from rag.evals.golden import GoldenItem
from rag.evals.judge import UnanswerableItem, citation_stats, judge_prompt, parse_judgement
from rag.evals.metrics import mean
from rag.generate.answer import SupportsRetrieve, answer_question
from rag.generate.generator import Generator
from rag.retrieval.retriever import RetrieverConfig

JUDGE_ATTEMPTS = 3


class RetryingGenerator:
    """Adds rate-limit backoff to any Generator (free tiers return 429 often)."""

    def __init__(self, inner: Generator, sleep: Callable[[float], None] = time.sleep) -> None:
        self.inner = inner
        self.sleep = sleep

    def generate(self, system: str, user: str) -> str:
        return with_retry(lambda: self.inner.generate(system, user), sleep=self.sleep)

    def stream(self, system: str, user: str) -> Iterator[str]:
        return self.inner.stream(system, user)


@dataclass(frozen=True)
class AnswerResult:
    id: str
    kind: str  # "answerable" | "unanswerable"
    question: str
    refused: bool
    answer: str
    faithfulness: float | None
    n_claims: int
    unsupported: tuple[str, ...]
    correct: str | None
    citations: dict[str, int]
    latency_ms: float


@dataclass(frozen=True)
class AnswerReport:
    results: tuple[AnswerResult, ...]
    summary: dict[str, float]


def _answer_one(
    item_id: str,
    kind: str,
    question: str,
    reference: str | None,
    retriever: SupportsRetrieve,
    generator: Generator,
    judge: Generator | None,
    config: RetrieverConfig,
) -> AnswerResult:
    started = time.perf_counter()
    answer = answer_question(question, retriever, generator, config)
    latency_ms = (time.perf_counter() - started) * 1000
    citations = {} if answer.refused else citation_stats(answer.text, len(answer.retrieved))
    judgement = None
    if judge is not None and not answer.refused:
        system, user = judge_prompt(question, answer.retrieved, answer.text, reference)
        # LLM judges occasionally emit invalid JSON (e.g. a stray bracket); a fresh sample
        # almost always parses, so re-ask instead of dropping the score.
        for _ in range(JUDGE_ATTEMPTS):
            judgement = parse_judgement(judge.generate(system, user))
            if judgement is not None:
                break
    return AnswerResult(
        id=item_id,
        kind=kind,
        question=question,
        refused=answer.refused,
        answer=answer.text,
        faithfulness=judgement.faithfulness if judgement else None,
        n_claims=judgement.n_claims if judgement else 0,
        unsupported=judgement.unsupported if judgement else (),
        correct=judgement.correct if judgement else None,
        citations=citations,
        latency_ms=latency_ms,
    )


def _share(items: Sequence[object], predicate: Callable[[object], bool]) -> float:
    return sum(1 for i in items if predicate(i)) / len(items) if items else 0.0


def _summary(results: Sequence[AnswerResult], judged_expected: bool) -> dict[str, float]:
    answerable = [r for r in results if r.kind == "answerable"]
    unanswerable = [r for r in results if r.kind == "unanswerable"]
    answered = [r for r in answerable if not r.refused]
    judged = [r for r in answered if r.faithfulness is not None]
    graded = [r for r in judged if r.correct is not None]
    with_sentences = [r for r in answered if r.citations.get("sentences")]
    return {
        "answerable_n": len(answerable),
        "answer_rate": _share(answerable, lambda r: not r.refused),
        "false_refusal_rate": _share(answerable, lambda r: r.refused),
        "faithfulness": mean(r.faithfulness for r in judged),
        "correct_yes": _share(graded, lambda r: r.correct == "yes"),
        "correct_partial": _share(graded, lambda r: r.correct == "partial"),
        "citation_coverage": mean(
            r.citations["cited_sentences"] / r.citations["sentences"] for r in with_sentences
        ),
        "invalid_citation_rate": _share(answered, lambda r: r.citations["invalid_citations"] > 0),
        "unanswerable_n": len(unanswerable),
        "unanswerable_refusal_rate": _share(unanswerable, lambda r: r.refused),
        "judge_failures": len(answered) - len(judged) if judged_expected else 0,
        "latency_p50_ms": sorted(r.latency_ms for r in results)[len(results) // 2]
        if results
        else 0.0,
    }


def evaluate_answers(
    golden: Sequence[GoldenItem],
    unanswerable: Sequence[UnanswerableItem],
    retriever: SupportsRetrieve,
    generator: Generator,
    judge: Generator | None,
    config: RetrieverConfig,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[int, int], None] | None = None,
) -> AnswerReport:
    gen = RetryingGenerator(generator, sleep)
    jud = RetryingGenerator(judge, sleep) if judge is not None else None
    jobs = [("answerable", g.id, g.question, g.reference_answer) for g in golden] + [
        ("unanswerable", u.id, u.question, None) for u in unanswerable
    ]
    results = []
    for done, (kind, item_id, question, reference) in enumerate(jobs, start=1):
        judge_for = jud if kind == "answerable" else None
        results.append(
            _answer_one(item_id, kind, question, reference, retriever, gen, judge_for, config)
        )
        if on_progress is not None:
            on_progress(done, len(jobs))
    return AnswerReport(tuple(results), _summary(results, judged_expected=judge is not None))


ROWS = (
    ("Answer rate (answerable)", "answer_rate"),
    ("False refusals", "false_refusal_rate"),
    ("Faithfulness (supported claims)", "faithfulness"),
    ("Correct: yes", "correct_yes"),
    ("Correct: partial", "correct_partial"),
    ("Citation coverage (sentences cited)", "citation_coverage"),
    ("Answers with invented citations", "invalid_citation_rate"),
    ("Refusal on unanswerable", "unanswerable_refusal_rate"),
)


def answers_markdown(report: AnswerReport) -> str:
    s = report.summary
    lines = ["| Metric | Value |", "|---|---:|"]
    lines += [f"| {label} | {s[key]:.3f} |" for label, key in ROWS]
    lines.append(f"| Judge parse failures | {int(s['judge_failures'])} |")
    lines.append(
        f"| Questions (answerable / unanswerable) | {s['answerable_n']} / {s['unanswerable_n']} |"
    )
    return "\n".join(lines) + "\n"


def write_answer_results(out_dir: Path, report: AnswerReport, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{str(meta['timestamp']).replace(':', '-')}-answers.json"
    payload = {
        "meta": meta,
        "summary": report.summary,
        "results": [asdict(r) for r in report.results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
