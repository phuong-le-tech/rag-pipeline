"""Answer-level evaluation: LLM-as-judge for faithfulness/correctness, plus mechanical checks.

Retrieval metrics say whether the right passages were found. They can't say whether the answer
stuck to them. The judge splits an answer into atomic claims and marks each one supported or
not *by the passages* (not by its own knowledge): faithfulness = supported / total. Citation
checks need no LLM at all.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.generate.citations import CITATION_RE, check_citations
from rag.generate.prompt import format_context
from rag.retrieval.search import Hit

VERDICTS = ("yes", "partial", "no")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

JUDGE_SYSTEM = """You are a strict grader for a retrieval-augmented question answering system.
You receive numbered context passages, a question, a candidate answer and, optionally, a
reference answer.

1. Split the candidate answer into atomic factual claims. Ignore citation markers, and ignore
   sentences that only say the documentation does not cover something.
2. For each claim, decide whether it is fully supported by the passages. Use only the
   passages, never your own knowledge: a true claim that the passages don't state is NOT
   supported.
3. If a reference answer is given, decide whether the candidate answer is correct: "yes" if it
   conveys the reference's key point, "partial" if it conveys part of it, "no" otherwise.
   Use null when no reference answer is given.

Reply with JSON only:
{"claims": [{"text": "...", "supported": true}], "correct": "yes" | "partial" | "no" | null}"""


@dataclass(frozen=True)
class Judgement:
    faithfulness: float  # supported claims / claims (1.0 when there are no claims)
    n_claims: int
    unsupported: tuple[str, ...]
    correct: str | None  # "yes" | "partial" | "no" | None


@dataclass(frozen=True)
class UnanswerableItem:
    id: str
    question: str
    origin: str


def judge_prompt(
    question: str, hits: Sequence[Hit], answer: str, reference: str | None
) -> tuple[str, str]:
    parts = [
        f"Context passages:\n\n{format_context(hits)}",
        f"Question: {question}",
        f"Candidate answer:\n{answer}",
    ]
    if reference:
        parts.append(f"Reference answer: {reference}")
    return JUDGE_SYSTEM, "\n\n---\n\n".join(parts)


def parse_judgement(text: str) -> Judgement | None:
    match = JSON_RE.search(text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    claims = data.get("claims")
    correct = data.get("correct")
    if not isinstance(claims, list) or (correct is not None and correct not in VERDICTS):
        return None
    supported = [bool(c.get("supported")) for c in claims if isinstance(c, dict)]
    unsupported = tuple(
        str(c.get("text", "")) for c in claims if isinstance(c, dict) and not c.get("supported")
    )
    faithfulness = sum(supported) / len(supported) if supported else 1.0
    return Judgement(faithfulness, len(supported), unsupported, correct)


def _sentences(answer: str) -> list[str]:
    prose = CODE_FENCE_RE.sub("\n", answer)
    sentences = []
    for line in prose.splitlines():
        line = BULLET_RE.sub("", line).strip()
        sentences.extend(s for s in SENTENCE_SPLIT_RE.split(line) if s.strip())
    return sentences


def citation_stats(answer: str, num_sources: int) -> dict[str, int]:
    """Mechanical citation checks: how many sentences carry a citation, how many are invented."""
    sentences = _sentences(answer)
    return {
        "sentences": len(sentences),
        "cited_sentences": sum(1 for s in sentences if CITATION_RE.search(s)),
        "invalid_citations": len(check_citations(answer, num_sources).invalid),
    }


def load_unanswerable(path: Path) -> list[UnanswerableItem]:
    """Questions the corpus can't answer; the right behaviour is to refuse."""
    items: list[UnanswerableItem] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in seen:
            raise ValueError(f"{path}:line {line_no}: duplicate id '{row['id']}'")
        seen.add(row["id"])
        items.append(UnanswerableItem(row["id"], row["question"].strip(), row.get("origin", "")))
    return items
