"""End-to-end answer evaluation: answer (pipeline or agent), judge, citations, refusals."""

import json
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag.agent.loop import AgentResult, Limits, run_agent
from rag.agent.tools import SearchResult
from rag.evals.drafting import with_retry
from rag.evals.golden import GoldenItem
from rag.evals.judge import UnanswerableItem, citation_stats, judge_prompt, parse_judgement
from rag.evals.metrics import mean
from rag.generate.answer import Answer, SupportsRetrieve, answer_question
from rag.generate.generator import Generator, Message, ToolGenerator, Turn, usage_of
from rag.retrieval.retriever import RetrieverConfig

JUDGE_ATTEMPTS = 3


class RetryingGenerator:
    """Adds rate-limit backoff to any Generator or ToolGenerator (free tiers return 429 often)."""

    def __init__(self, inner, sleep: Callable[[float], None] = time.sleep) -> None:
        self.inner = inner
        self.sleep = sleep

    def generate(self, system: str, user: str) -> str:
        return with_retry(lambda: self.inner.generate(system, user), sleep=self.sleep)

    def stream(self, system: str, user: str) -> Iterator[str]:
        return self.inner.stream(system, user)

    def chat(
        self, system: str, messages: list[Message], tools: list[dict], allow_tools: bool = True
    ) -> Turn:
        return with_retry(
            lambda: self.inner.chat(system, messages, tools, allow_tools=allow_tools),
            sleep=self.sleep,
        )


@dataclass(frozen=True)
class Answered:
    answer: Answer
    llm_calls: int
    tool_calls: int
    input_tokens: int | None
    output_tokens: int | None
    stopped: str  # "pipeline" for the fixed pipeline, AgentResult.stopped for the agent


def pipeline_answerer(
    retriever: SupportsRetrieve,
    generator: Generator,
    config: RetrieverConfig,
    answer_fn: Callable[..., Answer] = answer_question,  # rag.lc swaps in its LCEL chain
) -> Callable[[str], Answered]:
    def answer(question: str) -> Answered:
        usage = usage_of(generator)
        before = (usage.input_tokens, usage.output_tokens) if usage else (0, 0)
        result = answer_fn(question, retriever, generator, config)
        tokens = (None, None)  # no usage block, or this question moved no counter
        if (
            usage is not None
            and usage.reported
            and (usage.input_tokens, usage.output_tokens) != before
        ):
            tokens = (usage.input_tokens - before[0], usage.output_tokens - before[1])
        llm_calls = 0 if result.refused and not result.retrieved else 1
        return Answered(result, llm_calls, 0, *tokens, "pipeline")

    return answer


def agent_answerer(
    make_llm: Callable[[], ToolGenerator],
    search: Callable[[str, int], SearchResult],
    limits: Limits,
) -> Callable[[str], Answered]:
    def answer(question: str) -> Answered:
        *_, result = run_agent(question, make_llm(), search, limits)  # last item is the AgentResult
        return answered_from_agent(result)

    return answer


def answered_from_agent(result: AgentResult) -> Answered:
    return Answered(
        result.answer,
        result.llm_calls,
        result.tool_calls,
        result.input_tokens,
        result.output_tokens,
        result.stopped,
    )


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
    origin: str = ""
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    stopped: str = ""


@dataclass(frozen=True)
class AnswerReport:
    results: tuple[AnswerResult, ...]
    summary: dict[str, float]
    by_origin: dict[str, dict[str, float]] = field(default_factory=dict)


def _answer_one(
    item_id: str,
    kind: str,
    origin: str,
    question: str,
    reference: str | None,
    answerer: Callable[[str], Answered],
    judge: Generator | None,
) -> AnswerResult:
    started = time.perf_counter()
    done = answerer(question)
    latency_ms = (time.perf_counter() - started) * 1000
    answer = done.answer
    citations = {} if answer.refused else citation_stats(answer.text, len(answer.retrieved))
    judgement = None
    if judge is not None and not answer.refused:
        system, user = judge_prompt(question, answer.retrieved, answer.text, reference)
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
        origin=origin,
        llm_calls=done.llm_calls,
        tool_calls=done.tool_calls,
        input_tokens=done.input_tokens,
        output_tokens=done.output_tokens,
        stopped=done.stopped,
    )


def _share(items: Sequence[object], predicate: Callable[[object], bool]) -> float:
    return sum(1 for i in items if predicate(i)) / len(items) if items else 0.0


def _p95(latencies: Sequence[float]) -> float:
    """Nearest-rank 95th percentile."""
    if not latencies:
        return 0.0
    return sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1]


def _summary(
    results: Sequence[AnswerResult],
    judged_expected: bool,
    price_in: float = 0.0,
    price_out: float = 0.0,
) -> dict[str, float]:
    answerable = [r for r in results if r.kind == "answerable"]
    unanswerable = [r for r in results if r.kind == "unanswerable"]
    answered = [r for r in answerable if not r.refused]
    judged = [r for r in answered if r.faithfulness is not None]
    graded = [r for r in judged if r.correct is not None]
    with_sentences = [r for r in answered if r.citations.get("sentences")]
    reporting = [r for r in results if r.input_tokens is not None or r.output_tokens is not None]
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
        "latency_p95_ms": _p95([r.latency_ms for r in results]),
        "llm_calls_mean": mean(r.llm_calls for r in results),
        "tool_calls_mean": mean(r.tool_calls for r in results),
        "tokens_mean": mean((r.input_tokens or 0) + (r.output_tokens or 0) for r in reporting),
        "cost_usd_per_question": mean(
            ((r.input_tokens or 0) * price_in + (r.output_tokens or 0) * price_out) / 1_000_000
            for r in reporting
        ),
    }


def _load_checkpoint(path: Path) -> dict[str, AnswerResult]:
    """Results saved by an interrupted run, keyed by id; a missing file means nothing saved yet."""
    if not path.exists():
        return {}
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    loaded = [AnswerResult(**{**row, "unsupported": tuple(row["unsupported"])}) for row in rows]
    return {r.id: r for r in loaded}


def _append_checkpoint(path: Path, result: AnswerResult) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result)) + "\n")
        f.flush()


def evaluate_answers(
    golden: Sequence[GoldenItem],
    unanswerable: Sequence[UnanswerableItem],
    answerer: Callable[[str], Answered],
    judge: Generator | None,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[int, int], None] | None = None,
    price_in: float = 0.0,
    price_out: float = 0.0,
    checkpoint: Path | None = None,
) -> AnswerReport:
    jud = RetryingGenerator(judge, sleep) if judge is not None else None
    jobs = [("answerable", g.id, g.origin, g.question, g.reference_answer) for g in golden] + [
        ("unanswerable", u.id, u.origin, u.question, None) for u in unanswerable
    ]
    loaded = _load_checkpoint(checkpoint) if checkpoint is not None else {}
    results = []
    for done, (kind, item_id, origin, question, reference) in enumerate(jobs, start=1):
        if item_id in loaded:
            results.append(loaded[item_id])
        else:
            judge_for = jud if kind == "answerable" else None
            result = _answer_one(item_id, kind, origin, question, reference, answerer, judge_for)
            if checkpoint is not None:
                _append_checkpoint(checkpoint, result)
            results.append(result)
        if on_progress is not None:
            on_progress(done, len(jobs))
    judged_expected = judge is not None
    prices = {"price_in": price_in, "price_out": price_out}
    by_origin = {
        origin: _summary([r for r in results if r.origin == origin], judged_expected, **prices)
        for origin in sorted({r.origin for r in results})
    }
    summary = _summary(results, judged_expected, **prices)
    return AnswerReport(tuple(results), summary, by_origin)


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

COST_ROWS = (
    ("LLM calls per question", "llm_calls_mean", "{:.2f}"),
    ("Searches per question", "tool_calls_mean", "{:.2f}"),
    ("Tokens per question", "tokens_mean", "{:.0f}"),
    ("Latency p50 (ms)", "latency_p50_ms", "{:.0f}"),
    ("Latency p95 (ms)", "latency_p95_ms", "{:.0f}"),
    ("Cost per question (USD)", "cost_usd_per_question", "{:.5f}"),
)

ORIGIN_ROWS = (
    ("Answer rate (answerable)", "answer_rate", "answerable_n"),
    ("False refusals", "false_refusal_rate", "answerable_n"),
    ("Faithfulness (supported claims)", "faithfulness", "answerable_n"),
    ("Correct: yes", "correct_yes", "answerable_n"),
    ("Correct: partial", "correct_partial", "answerable_n"),
    ("Answers with invented citations", "invalid_citation_rate", "answerable_n"),
    ("Refusal on unanswerable", "unanswerable_refusal_rate", "unanswerable_n"),
)


def answers_markdown(report: AnswerReport) -> str:
    s = report.summary
    lines = ["| Metric | Value |", "|---|---:|"]
    lines += [f"| {label} | {s[key]:.3f} |" for label, key in ROWS]
    lines += [f"| {label} | {fmt.format(s[key])} |" for label, key, fmt in COST_ROWS]
    lines.append(f"| Judge parse failures | {int(s['judge_failures'])} |")
    lines.append(
        f"| Questions (answerable / unanswerable) | {s['answerable_n']} / {s['unanswerable_n']} |"
    )
    origins = list(report.by_origin)
    lines += ["", "### By origin", "", "| Metric | " + " | ".join(origins) + " |"]
    lines.append("|---|" + "---:|" * len(origins))
    for label, key, count in ORIGIN_ROWS:
        cells = []
        for origin in origins:
            o = report.by_origin[origin]
            cells.append(f"{o[key]:.3f}" if o[count] else "")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_answer_results(out_dir: Path, report: AnswerReport, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{str(meta['timestamp']).replace(':', '-')}-answers.json"
    payload = {
        "meta": meta,
        "summary": report.summary,
        "by_origin": report.by_origin,
        "results": [asdict(r) for r in report.results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
