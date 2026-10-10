"""Run a retrieval config over the golden set and summarise the metrics.

Every ``RetrieverConfig`` in ``ABLATIONS`` becomes one row of the comparison table, so each
retrieval decision in the docs is backed by a measured difference, not an anecdote.
"""

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from rag.evals.golden import GoldenItem
from rag.evals.metrics import hit_at_k, mean, ndcg_at_k, recall_at_k, reciprocal_rank
from rag.retrieval.retriever import RetrieverConfig
from rag.retrieval.search import Hit

KS = (1, 3, 5, 10)
DEPTH = max(KS)

ABLATIONS: dict[str, RetrieverConfig] = {
    "dense": RetrieverConfig(mode="dense"),
    "keyword": RetrieverConfig(mode="keyword"),
    "hybrid": RetrieverConfig(mode="hybrid"),
    "hybrid+rerank": RetrieverConfig(mode="hybrid", rerank=True),
    "hybrid, no generated": RetrieverConfig(mode="hybrid", include_generated=False),
    "hybrid+rerank, no generated": RetrieverConfig(
        mode="hybrid", rerank=True, include_generated=False
    ),
}


class SupportsRetrieve(Protocol):
    def retrieve(self, query: str, config: RetrieverConfig) -> list[Hit]: ...


@dataclass(frozen=True)
class QuestionResult:
    id: str
    ranked_sources: tuple[str, ...]
    metrics: dict[str, float]
    latency_ms: float


@dataclass(frozen=True)
class EvalReport:
    name: str
    config: RetrieverConfig
    results: tuple[QuestionResult, ...]
    summary: dict[str, float]


def _percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]


def _score(ranked: Sequence[str], relevant: set[str]) -> dict[str, float]:
    metrics = {f"hit@{k}": hit_at_k(ranked, relevant, k) for k in KS}
    metrics |= {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in KS}
    metrics["mrr"] = reciprocal_rank(ranked, relevant)
    metrics["ndcg@10"] = ndcg_at_k(ranked, relevant, 10)
    return metrics


def evaluate(
    retriever: SupportsRetrieve,
    items: Sequence[GoldenItem],
    config: RetrieverConfig,
    name: str,
) -> EvalReport:
    if not items:
        raise ValueError("golden set is empty")
    deep = replace(config, k=DEPTH, candidates=max(config.candidates, DEPTH))
    results = []
    for item in items:
        started = time.perf_counter()
        hits = retriever.retrieve(item.question, deep)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = tuple(h.source for h in hits)
        results.append(
            QuestionResult(item.id, ranked, _score(ranked, set(item.relevant_sources)), latency_ms)
        )
    keys = results[0].metrics.keys()
    summary: dict[str, float] = {key: mean(r.metrics[key] for r in results) for key in keys}
    latencies = [r.latency_ms for r in results]
    summary |= {
        "n": len(results),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
    }
    return EvalReport(name, config, tuple(results), summary)


COLUMNS = (
    ("Hit@1", "hit@1"),
    ("Hit@5", "hit@5"),
    ("Recall@5", "recall@5"),
    ("MRR", "mrr"),
    ("nDCG@10", "ndcg@10"),
)


def write_results(out_dir: Path, reports: Sequence[EvalReport], meta: dict) -> Path:
    """Save the full run (meta, summaries, per-question rankings) as JSON plus a Markdown table.

    Files are named after ``meta["timestamp"]`` so runs never overwrite each other.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(meta["timestamp"]).replace(":", "-")
    payload = {
        "meta": meta,
        "reports": [
            {
                "name": r.name,
                "config": asdict(r.config),
                "summary": r.summary,
                "results": [asdict(q) for q in r.results],
            }
            for r in reports
        ],
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(to_markdown(reports), encoding="utf-8")
    return json_path


def to_markdown(reports: Sequence[EvalReport]) -> str:
    header = "| Config | " + " | ".join(label for label, _ in COLUMNS) + " | p50 ms |"
    divider = "|---|" + "---:|" * (len(COLUMNS) + 1)
    rows = [
        f"| {r.name} | "
        + " | ".join(f"{r.summary[key]:.3f}" for _, key in COLUMNS)
        + f" | {r.summary['latency_p50_ms']:.0f} |"
        for r in reports
    ]
    return "\n".join([header, divider, *rows]) + "\n"
