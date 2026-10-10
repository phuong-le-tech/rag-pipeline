"""Human review of LLM-drafted golden questions: accept, edit or reject, resumable.

Decisions are appended to a small ledger (``id<TAB>decision``) so a review can stop at any
point and resume later without showing a draft twice. Accepted rows go straight into the
golden set, tagged ``reviewed`` (and ``edited`` when the question was rewritten).
"""

import json
from pathlib import Path

DECISIONS = ("accepted", "rejected")


def _reviewed_ids(reviewed: Path) -> set[str]:
    if not reviewed.exists():
        return set()
    return {line.split("\t", 1)[0] for line in reviewed.read_text().splitlines() if line}


def pending_drafts(drafts: Path, reviewed: Path) -> list[dict]:
    done = _reviewed_ids(reviewed)
    rows = [json.loads(line) for line in drafts.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row["id"] not in done]


def record_decision(
    row: dict, decision: str, golden: Path, reviewed: Path, edited_question: str | None
) -> None:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
    if decision == "accepted":
        tags = [*row.get("tags", []), "reviewed"] + (["edited"] if edited_question else [])
        accepted = {**row, "question": edited_question or row["question"], "tags": tags}
        with golden.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(accepted) + "\n")
    with reviewed.open("a", encoding="utf-8") as ledger:
        ledger.write(f"{row['id']}\t{decision}\n")
