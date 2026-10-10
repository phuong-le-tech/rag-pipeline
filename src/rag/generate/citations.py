"""Parse and validate ``[n]`` citations in a generated answer.

LLMs sometimes cite passages that don't exist ("[7]" with five passages). That is a hallucinated
source, and the UI must never turn it into a link, so every number is checked against the
context size.
"""

import re
from dataclasses import dataclass

# "[1]", "[1, 3]", "[2][4]" but not "a[0]" (preceded by a word character) or "[x]".
CITATION_RE = re.compile(r"(?<!\w)\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class CitationReport:
    cited: tuple[int, ...]  # valid numbers, first-appearance order
    invalid: tuple[int, ...]  # numbers outside 1..num_sources


def parse_citations(text: str) -> list[int]:
    numbers: list[int] = []
    for match in CITATION_RE.finditer(text):
        for part in match.group(1).split(","):
            number = int(part)
            if number not in numbers:
                numbers.append(number)
    return numbers


def check_citations(text: str, num_sources: int) -> CitationReport:
    numbers = parse_citations(text)
    valid = tuple(n for n in numbers if 1 <= n <= num_sources)
    invalid = tuple(n for n in numbers if not 1 <= n <= num_sources)
    return CitationReport(cited=valid, invalid=invalid)
