"""Heading-aware Markdown chunking.

Splits a document at headings so each chunk stays inside one section, records the
heading path ("Guide > Install") as retrieval metadata, then packs whole paragraphs
up to ``max_chars``. Paragraphs longer than ``max_chars`` are hard-split.
"""

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
PARAGRAPH_SEP = "\n\n"
HEADING_SEP = " > "


@dataclass(frozen=True)
class Chunk:
    ord: int
    heading: str
    content: str


@dataclass(frozen=True)
class _Section:
    heading: str
    body: str


def _split_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: tuple[tuple[int, str], ...] = ()
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append(_Section(HEADING_SEP.join(t for _, t in stack), content))

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING_RE.match(line)
        if match is None:
            body.append(line)
            continue
        flush()
        body = []
        level = len(match.group(1))
        stack = tuple(entry for entry in stack if entry[0] < level) + ((level, match.group(2)),)

    flush()
    return sections


def _pack_paragraphs(body: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    buffer = ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", body)):
        if not para:
            continue
        if len(para) > max_chars:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(para[i : i + max_chars] for i in range(0, len(para), max_chars))
        elif buffer and len(buffer) + len(PARAGRAPH_SEP) + len(para) > max_chars:
            pieces.append(buffer)
            buffer = para
        else:
            buffer = f"{buffer}{PARAGRAPH_SEP}{para}" if buffer else para
    if buffer:
        pieces.append(buffer)
    return pieces


def chunk_markdown(text: str, max_chars: int) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    pieces = [
        (section.heading, piece)
        for section in _split_sections(text)
        for piece in _pack_paragraphs(section.body, max_chars)
    ]
    return [Chunk(ord=i, heading=h, content=c) for i, (h, c) in enumerate(pieces)]
