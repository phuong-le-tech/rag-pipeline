"""Draft golden-set questions with an LLM, one per sampled chunk, for human review.

LLM-drafted questions scale cheaply but have a known bias: they echo the passage's wording,
which makes retrieval look easier than it is for real users. The prompt asks for user-style
phrasing, and ``lexical_overlap`` tags drafts that copy the passage anyway (``high-overlap``)
so the reviewer can rewrite or drop them, and results can be sliced by it.
"""

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import anthropic
import openai
import psycopg

from rag.generate.generator import Generator

HIGH_OVERLAP = 0.6
MIN_CHARS = 300
RATE_LIMIT_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
)
STOPWORD_TEXT = """
a an and are as at be by can could do does for from had has have how i if in into is it its me
my no not of on or our should so that the their then there these this to was we what when where
which who why will with would you your
"""
STOPWORDS = frozenset(STOPWORD_TEXT.split())
WORD_RE = re.compile(r"[a-z0-9]+")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = """You write evaluation questions for a Kubernetes documentation search engine.

Given one passage, write ONE question that a real Kubernetes user might type and that this
passage answers. Phrase it in the user's own words: describe their situation or goal. Do not
copy distinctive phrases or the section heading from the passage. Then give a one- or
two-sentence answer taken from the passage.

Reply with JSON only: {"question": "...", "answer": "..."}
If the passage cannot answer a realistic user question (navigation, boilerplate, changelog,
lists of links, licence text), reply with JSON only: {"skip": true}"""

SAMPLE_SQL = """
SELECT chunk_id, source, section, heading, content FROM (
    SELECT DISTINCT ON (d.source)
           c.id AS chunk_id, d.source, d.metadata->>'section' AS section, c.heading, c.content,
           md5(c.id::text || %(seed)s) AS shuffle
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE length(c.content) >= %(min_chars)s
      AND coalesce((d.metadata->>'generated')::boolean, false) = %(generated)s
    ORDER BY d.source, md5(c.id::text || %(seed)s)
) one_chunk_per_page
ORDER BY shuffle
LIMIT %(limit)s
"""


@dataclass(frozen=True)
class ChunkSample:
    chunk_id: int
    source: str
    section: str
    heading: str
    content: str


@dataclass(frozen=True)
class Draft:
    question: str
    answer: str


def sample_chunks(
    conn: psycopg.Connection, n: int, seed: str, generated_share: float = 0.15
) -> list[ChunkSample]:
    """Deterministic sample, one chunk per page, with a fixed share from generated pages."""
    n_generated = round(n * generated_share)
    rows = []
    for generated, limit in ((False, n - n_generated), (True, n_generated)):
        params = {"seed": seed, "min_chars": MIN_CHARS, "generated": generated, "limit": limit}
        rows.extend(conn.execute(SAMPLE_SQL, params).fetchall())
    return [ChunkSample(*row) for row in rows]


def draft_prompt(sample: ChunkSample) -> tuple[str, str]:
    return SYSTEM_PROMPT, f"Page section: {sample.heading}\n\nPassage:\n{sample.content}"


def parse_draft(text: str) -> Draft | None:
    match = JSON_RE.search(text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    question = str(data.get("question") or "").strip()
    answer = str(data.get("answer") or "").strip()
    if data.get("skip") or not question:
        return None
    return Draft(question, answer)


def _content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS}


def lexical_overlap(question: str, passage: str) -> float:
    """Share of the question's content words that also appear in the passage (0..1)."""
    words = _content_words(question)
    if not words:
        return 0.0
    return len(words & _content_words(passage)) / len(words)


def with_retry[T](
    fn: Callable[[], T],
    retry_on: tuple[type[Exception], ...] = RATE_LIMIT_ERRORS,
    attempts: int = 6,
    base_delay: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn``, retrying listed errors with exponential backoff (base, 2x base, 4x base...)."""
    for attempt in range(attempts):
        try:
            return fn()
        except retry_on:
            if attempt == attempts - 1:
                raise
            sleep(base_delay * 2**attempt)
    raise AssertionError("unreachable")


def draft_items(
    samples: Sequence[ChunkSample],
    generator: Generator,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[int, int], None] | None = None,
    skip_ids: set[str] | frozenset[str] = frozenset(),
    on_row: Callable[[dict], None] | None = None,
) -> list[dict]:
    """One golden-format row per usable sample; skipped or unparseable drafts are dropped.

    ``on_row`` receives each row as soon as it exists (write it to disk there), and ``skip_ids``
    lists ids already drafted, so an interrupted run resumes without repeating LLM calls.
    """
    rows = []
    for position, sample in enumerate(samples, start=1):
        if f"llm-{position:04d}" in skip_ids:
            continue
        system, user = draft_prompt(sample)
        text = with_retry(lambda s=system, u=user: generator.generate(s, u), sleep=sleep)
        draft = parse_draft(text)
        if on_progress is not None:
            on_progress(position, len(rows) + (draft is not None))
        if draft is None:
            continue
        tags = ["llm-draft", sample.section or "root"]
        if lexical_overlap(draft.question, sample.content) >= HIGH_OVERLAP:
            tags.append("high-overlap")
        row = {
            "id": f"llm-{position:04d}",
            "question": draft.question,
            "relevant_sources": [sample.source],
            "reference_answer": draft.answer,
            "tags": tags,
            "origin": "llm",
        }
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows
