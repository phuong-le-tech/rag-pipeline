"""Incremental ingestion: documents -> chunks -> embeddings -> Postgres.

A document is skipped when both its ``content_hash`` and the pipeline ``fingerprint``
(embedding backend, model, dimension, chunk size) match what is stored. Changing either the
text or how it is processed triggers re-chunking and re-embedding. Each document is written in
its own transaction, so a crash never leaves a document half-indexed.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from rag.embed import Embedder
from rag.ingest.chunking import Chunk, chunk_markdown
from rag.ingest.loaders import Document

UPSERT_DOCUMENT = """
INSERT INTO documents (source, content_hash, metadata, collection, updated_at)
VALUES (%s, %s, %s, %s, now())
ON CONFLICT (source) DO UPDATE
    SET content_hash = EXCLUDED.content_hash, metadata = EXCLUDED.metadata, updated_at = now()
RETURNING id
"""
INSERT_CHUNK = """
INSERT INTO chunks (document_id, ord, heading, content, embedding) VALUES (%s, %s, %s, %s, %s)
"""


@dataclass(frozen=True)
class IngestStats:
    added: int
    updated: int
    skipped: int
    deleted: int
    chunks: int


def embedding_text(chunk: Chunk) -> str:
    """Text sent to the embedder: heading path gives the chunk its page/section context."""
    return f"{chunk.heading}\n\n{chunk.content}" if chunk.heading else chunk.content


def _stored_versions(
    conn: psycopg.Connection, collection: str
) -> dict[str, tuple[str, str | None]]:
    rows = conn.execute(
        "SELECT source, content_hash, metadata->>'pipeline' FROM documents WHERE collection = %s",
        (collection,),
    )
    return {source: (content_hash, pipeline) for source, content_hash, pipeline in rows}


def store_document(
    conn: psycopg.Connection,
    doc: Document,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    fingerprint: str,
    collection: str,
) -> None:
    metadata = {
        "url": doc.url,
        "title": doc.title,
        "section": doc.section,
        "generated": doc.generated,
        "pipeline": fingerprint,
    }
    with conn.transaction():
        doc_id = conn.execute(
            UPSERT_DOCUMENT, (doc.source, doc.content_hash, Jsonb(metadata), collection)
        ).fetchone()[0]
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                INSERT_CHUNK,
                [
                    (doc_id, c.ord, c.heading, c.content, np.asarray(v, dtype=np.float32))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )


def _prune(conn: psycopg.Connection, keep: set[str], collection: str) -> int:
    with conn.transaction():
        cur = conn.execute(
            "DELETE FROM documents WHERE collection = %s AND NOT (source = ANY(%s))",
            (collection, list(keep)),
        )
        return cur.rowcount


def ingest(
    conn: psycopg.Connection,
    docs: Iterable[Document],
    embedder: Embedder,
    max_chars: int,
    fingerprint: str,
    prune: bool = True,
    on_progress: Callable[[int], None] | None = None,
    collection: str = "k8s",
) -> IngestStats:
    stored = _stored_versions(conn, collection)
    conn.commit()  # end the read so each document gets its own real transaction

    added = updated = skipped = total_chunks = 0
    seen: set[str] = set()
    for i, doc in enumerate(docs, start=1):
        seen.add(doc.source)
        previous = stored.get(doc.source)
        if previous == (doc.content_hash, fingerprint):
            skipped += 1
        else:
            chunks = chunk_markdown(doc.text, max_chars)
            vectors = embedder.embed([embedding_text(c) for c in chunks])
            store_document(conn, doc, chunks, vectors, fingerprint, collection)
            total_chunks += len(chunks)
            if previous is None:
                added += 1
            else:
                updated += 1
        if on_progress is not None:
            on_progress(i)

    deleted = _prune(conn, seen, collection) if prune else 0
    return IngestStats(added, updated, skipped, deleted, total_chunks)
