"""Golden set: questions with the pages that answer them, stored as JSON Lines.

One object per line::

    {"id": "rollback-1", "question": "How do I roll back a deployment?",
     "relevant_sources": ["concepts/workloads/controllers/deployment.md"],
     "reference_answer": "kubectl rollout undo deployment/<name>",
     "tags": ["howto", "kubectl"], "origin": "human"}

``relevant_sources`` are ``documents.source`` paths (relative to the corpus root). ``origin``
records who wrote the question, so LLM-drafted and hand-written questions can be scored
separately: LLM drafts tend to reuse the passage's own words, which flatters dense retrieval.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ORIGINS = ("human", "llm")


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    relevant_sources: tuple[str, ...]
    reference_answer: str | None
    tags: tuple[str, ...]
    origin: str


def _parse(row: dict, where: str) -> GoldenItem:
    question = str(row.get("question", "")).strip()
    sources = tuple(row.get("relevant_sources") or ())
    origin = row.get("origin")
    if not row.get("id"):
        raise ValueError(f"{where}: missing id")
    if not question:
        raise ValueError(f"{where}: question must not be empty")
    if not sources:
        raise ValueError(f"{where}: relevant_sources must list at least one page")
    if origin not in ORIGINS:
        raise ValueError(f"{where}: origin must be one of {ORIGINS}, got {origin!r}")
    return GoldenItem(
        id=str(row["id"]),
        question=question,
        relevant_sources=sources,
        reference_answer=row.get("reference_answer"),
        tags=tuple(row.get("tags") or ()),
        origin=origin,
    )


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:line {line_no}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: invalid JSON ({exc.msg})") from exc
        item = _parse(row, where)
        if item.id in seen:
            raise ValueError(f"{where}: duplicate id '{item.id}'")
        seen.add(item.id)
        items.append(item)
    return items


def unknown_sources(items: Iterable[GoldenItem], known: set[str]) -> dict[str, tuple[str, ...]]:
    """Golden sources missing from the index (typos, renamed pages), keyed by question id."""
    missing = {item.id: tuple(s for s in item.relevant_sources if s not in known) for item in items}
    return {item_id: sources for item_id, sources in missing.items() if sources}
