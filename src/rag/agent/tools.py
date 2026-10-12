"""The search_docs tool: validates the model's arguments, then runs retrieval and records stages."""

import json
import time
from dataclasses import dataclass, replace
from typing import Protocol

from rag.retrieval.retriever import RetrieverConfig, StageHook
from rag.retrieval.search import Hit

MAX_QUERY_CHARS = 300
K_RANGE = (1, 10)
DEFAULT_K = 5
STAGE_HITS = 10  # per-stage hits kept for the trace (same as the API's STAGE_HITS)

SEARCH_DOCS: dict = {
    "name": "search_docs",
    "description": (
        "Search the Kubernetes documentation. Call it before answering any Kubernetes question, "
        "and once per part of a multi-part question. Use short keyword-style queries in "
        "Kubernetes vocabulary: object kinds (Deployment, PodDisruptionBudget), field names "
        "(readinessProbe, storageClassName), and kubectl verbs (rollout undo, drain). "
        "Returns numbered passages, each with its page title and URL; cite them by number."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "maxLength": MAX_QUERY_CHARS,
                "description": "Short keyword-style search query.",
            },
            "k": {
                "type": "integer",
                "minimum": K_RANGE[0],
                "maximum": K_RANGE[1],
                "default": DEFAULT_K,
                "description": "Number of passages to return.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def parse_search_args(raw: str) -> tuple[str, int] | str:
    """Return (query, k) from the model's JSON arguments, or a sentence the model can read."""
    try:
        args = json.loads(raw)
    except (ValueError, RecursionError):
        return "Error: arguments are invalid JSON. Send a JSON object with a query string."
    if not isinstance(args, dict):
        return "Error: arguments must be a JSON object with a query string."
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return "Error: query is required and must be a non-blank string."
    query = query.strip()
    if len(query) > MAX_QUERY_CHARS:
        return f"Error: query must be at most {MAX_QUERY_CHARS} characters."
    k = args.get("k", DEFAULT_K)
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(k, bool) or not isinstance(k, int) or not K_RANGE[0] <= k <= K_RANGE[1]:
        return f"Error: k must be an integer from {K_RANGE[0]} to {K_RANGE[1]}."
    return query, k


class SupportsStagedRetrieve(Protocol):
    def retrieve(
        self, query: str, config: RetrieverConfig, on_stage: StageHook | None = None
    ) -> list[Hit]: ...


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[Hit, ...]
    stages: tuple[tuple[str, tuple[Hit, ...], float], ...]  # (name, top hits, ms)
    ms: float


class SearchTool:
    def __init__(self, retriever: SupportsStagedRetrieve, config: RetrieverConfig) -> None:
        self.retriever = retriever
        self.config = config

    def __call__(self, query: str, k: int) -> SearchResult:
        config = replace(self.config, k=k, candidates=max(self.config.candidates, k))
        stages: list[tuple[str, tuple[Hit, ...], float]] = []

        def on_stage(name: str, hits: list[Hit], ms: float) -> None:
            stages.append((name, tuple(hits[:STAGE_HITS]), ms))

        start = time.perf_counter()
        hits = self.retriever.retrieve(query, config, on_stage=on_stage)
        ms = (time.perf_counter() - start) * 1000
        return SearchResult(tuple(hits), tuple(stages), ms)
