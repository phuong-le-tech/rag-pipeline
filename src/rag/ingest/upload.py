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
MAX_COLLECTION_DOCS = 20
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
    """Yield ``stage`` events (load, clean, chunk, embed, store), then ``done``.

    ``collection`` may already hold documents: the file is added to it (same name replaces)."""
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
    others = conn.execute(  # re-uploading a file with the same name replaces it, so not counted
        "SELECT count(*) FROM documents WHERE collection = %s AND source <> %s",
        (collection, doc.source),
    ).fetchone()[0]
    if others >= MAX_COLLECTION_DOCS:
        raise ValueError(f"a collection holds at most {MAX_COLLECTION_DOCS} documents")
    vectors = embedder.embed([embedding_text(c) for c in chunks])
    yield stage("embed", chunks=len(chunks), dim=settings.embedding_dim)
    store_document(conn, doc, chunks, vectors, settings.pipeline_fingerprint(), collection)
    yield stage("store", docs=1, chunks=len(chunks))
    docs, total = conn.execute(
        "SELECT count(DISTINCT d.id), count(c.id) FROM documents d "
        "LEFT JOIN chunks c ON c.document_id = d.id WHERE d.collection = %s",
        (collection,),
    ).fetchone()
    # Collection totals, so the UI can show the whole set after each added file.
    yield "done", {"collection": collection, "docs": docs, "chunks": total}
