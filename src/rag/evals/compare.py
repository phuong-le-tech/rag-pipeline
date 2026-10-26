"""Side-by-side table of a hand-written and a LangChain answer-eval run."""

import math
from collections.abc import Iterable
from pathlib import Path

import rag
from rag.evals.answers import COST_ROWS, ROWS

PACKAGE = Path(rag.__file__).parent
# Orchestration and provider glue only: the LangChain side reuses the hand-written prompt,
# search tool and citation checks, so its count understates what it depends on.
LOC_PATHS = {
    "handwritten": [
        PACKAGE / "generate" / "answer.py",
        PACKAGE / "generate" / "generator.py",
        PACKAGE / "agent" / "loop.py",
    ],
    "langchain": sorted((PACKAGE / "lc").glob("*.py")),
}


def count_lines(paths: Iterable[Path]) -> int:
    return sum(1 for p in paths for line in p.read_text().splitlines() if line.strip())


def impl_of(result: dict) -> str:
    return result["meta"].get("impl", "handwritten")  # runs saved before --impl existed


def _cell(summary: dict, key: str, fmt: str) -> str:
    value = summary.get(key)
    return "-" if value is None or math.isnan(value) else fmt.format(value)


def compare_markdown(handwritten: dict, langchain: dict, loc: dict[str, int]) -> str:
    if (impl_of(handwritten), impl_of(langchain)) != ("handwritten", "langchain"):
        raise ValueError("need one hand-written and one langchain result")
    if handwritten["meta"]["mode"] != langchain["meta"]["mode"]:
        raise ValueError(
            f"mode differs ({handwritten['meta']['mode']} vs {langchain['meta']['mode']}): "
            "compare runs of the same mode"
        )
    lines = ["| Metric | Hand-written | LangChain |", "|---|---:|---:|"]
    rows = [(label, key, "{:.3f}") for label, key in ROWS] + list(COST_ROWS)
    for label, key, fmt in rows:
        cells = (_cell(r["summary"], key, fmt) for r in (handwritten, langchain))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(f"| Lines of code | {loc['handwritten']} | {loc['langchain']} |")
    return "\n".join(lines) + "\n"
