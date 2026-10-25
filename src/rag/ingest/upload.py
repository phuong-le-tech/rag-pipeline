"""Visitor uploads: ingest one document into its own collection, reporting each stage.

The collection id is a random 128-bit token: whoever holds it can search the upload, so it is
the only access control. Uploads expire (``purge_expired``) and are never logged.
"""

import time
from collections.abc import Iterator

import psycopg

from rag.config import Settings
from rag.embed import Embedder
from rag.ingest.chunking import chunk_markdown
from rag.ingest.loaders import load_upload
from rag.ingest.pipeline import embedding_text, store_document

MAX_UPLOAD_BYTES = 1_000_000
MAX_UPLOAD_CHUNKS = 300
PREVIEW_CHUNKS = 3
PREVIEW_CHARS = 160

Event = tuple[str, dict]


def purge_expired(conn: psycopg.Connection, ttl_hours: int) -> int:
    """Delete upload collections older than the TTL (chunks cascade). Never touches 'k8s'."""
    with conn.transaction():
        cur = conn.execute(
            "DELETE FROM documents WHERE collection <> 'k8s' "
            "AND created_at < now() - make_interval(hours => %s)",
            (ttl_hours,),
        )
        return cur.rowcount


def ingest_upload(
    conn: psycopg.Connection,
    embedder: Embedder,
    settings: Settings,
    collection: str,
    filename: str,
    text: str,
) -> Iterator[Event]:
    """Yield ``stage`` events (load, clean, chunk, embed, store), then ``done``."""
    t = time.perf_counter()

    def stage(name: str, **data) -> Event:
        nonlocal t
        now = time.perf_counter()
        ms, t = round((now - t) * 1000), now
        return "stage", {"name": name, "ms": ms, **data}

    yield stage("load", chars=len(text), bytes=len(text.encode("utf-8")))
    doc = load_upload(text, filename, collection)
    yield stage("clean", chars=len(doc.text), title=doc.title)
    chunks = chunk_markdown(doc.text, settings.chunk_max_chars)
    if len(chunks) > MAX_UPLOAD_CHUNKS:
        raise ValueError(f"upload has {len(chunks)} chunks; the limit is {MAX_UPLOAD_CHUNKS}")
    previews = [
        {"heading": c.heading, "content": c.content[:PREVIEW_CHARS]}
        for c in chunks[:PREVIEW_CHUNKS]
    ]
    yield stage("chunk", chunks=len(chunks), previews=previews)
    vectors = embedder.embed([embedding_text(c) for c in chunks])
    yield stage("embed", chunks=len(chunks), dim=settings.embedding_dim)
    store_document(conn, doc, chunks, vectors, settings.pipeline_fingerprint(), collection)
    yield stage("store", docs=1, chunks=len(chunks))
    yield "done", {"collection": collection, "docs": 1, "chunks": len(chunks)}
