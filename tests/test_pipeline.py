import pytest

from rag.embed import FakeEmbedder
from rag.ingest.chunking import Chunk
from rag.ingest.loaders import Document
from rag.ingest.pipeline import IngestStats, embedding_text, ingest

DIM = 8
FP = "fake:test:8:1500"


def doc(source, body, content_hash, generated=False):
    return Document(
        source=source,
        url=f"https://kubernetes.io/docs/{source.removesuffix('.md')}/",
        title=source,
        section=source.split("/")[0],
        text=f"# {source}\n\n{body}",
        content_hash=content_hash,
        generated=generated,
    )


def run(conn, docs, fingerprint=FP, prune=True):
    return ingest(
        conn, docs, FakeEmbedder(DIM), max_chars=1500, fingerprint=fingerprint, prune=prune
    )


def counts(conn):
    return conn.execute(
        "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
    ).fetchone()


def test_embedding_text_prefixes_heading_path():
    chunk = Chunk(ord=0, heading="Deployments > Rolling back", content="Use rollout undo.")

    assert embedding_text(chunk) == "Deployments > Rolling back\n\nUse rollout undo."
    assert embedding_text(Chunk(ord=0, heading="", content="x")) == "x"


@pytest.mark.integration
def test_first_ingest_stores_documents_chunks_and_metadata(conn):
    stats = run(conn, [doc("a/one.md", "alpha\n\n## Sub\nbeta", "h1", generated=True)])

    assert stats == IngestStats(added=1, updated=0, skipped=0, deleted=0, chunks=2)
    assert counts(conn) == (1, 2)
    meta = conn.execute("SELECT metadata FROM documents").fetchone()[0]
    assert meta == {
        "url": "https://kubernetes.io/docs/a/one/",
        "title": "a/one.md",
        "section": "a",
        "generated": True,
        "pipeline": FP,
    }
    rows = conn.execute("SELECT ord, heading, embedding FROM chunks ORDER BY ord").fetchall()
    assert [(r[0], r[1]) for r in rows] == [(0, "a/one.md"), (1, "a/one.md > Sub")]
    assert len(rows[0][2].to_list()) == DIM


@pytest.mark.integration
def test_reingest_unchanged_skips_and_keeps_rows(conn):
    run(conn, [doc("a/one.md", "alpha", "h1")])
    before = conn.execute("SELECT id FROM chunks").fetchall()

    stats = run(conn, [doc("a/one.md", "alpha", "h1")])

    assert stats == IngestStats(added=0, updated=0, skipped=1, deleted=0, chunks=0)
    assert conn.execute("SELECT id FROM chunks").fetchall() == before


@pytest.mark.integration
def test_changed_content_replaces_chunks(conn):
    run(conn, [doc("a/one.md", "alpha", "h1")])

    stats = run(conn, [doc("a/one.md", "alpha\n\n## New\ngamma", "h2")])

    assert stats.updated == 1
    assert counts(conn) == (1, 2)
    assert conn.execute("SELECT content_hash FROM documents").fetchone()[0] == "h2"


@pytest.mark.integration
def test_pipeline_fingerprint_change_forces_reembed(conn):
    run(conn, [doc("a/one.md", "alpha", "h1")])

    stats = run(conn, [doc("a/one.md", "alpha", "h1")], fingerprint="local:bge:384:1500")

    assert stats.updated == 1


@pytest.mark.integration
def test_missing_documents_are_pruned_with_their_chunks(conn):
    run(conn, [doc("a/one.md", "alpha", "h1"), doc("b/two.md", "beta", "h2")])

    stats = run(conn, [doc("a/one.md", "alpha", "h1")])

    assert stats.deleted == 1
    assert counts(conn) == (1, 1)


@pytest.mark.integration
def test_prune_disabled_keeps_missing_documents(conn):
    run(conn, [doc("a/one.md", "alpha", "h1"), doc("b/two.md", "beta", "h2")])

    stats = run(conn, [doc("a/one.md", "alpha", "h1")], prune=False)

    assert stats.deleted == 0
    assert counts(conn) == (2, 2)


@pytest.mark.integration
def test_embedding_failure_leaves_previous_version_intact(conn):
    run(conn, [doc("a/one.md", "alpha", "h1")])

    class Boom(FakeEmbedder):
        def embed(self, texts):
            raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError):
        ingest(conn, [doc("a/one.md", "changed", "h2")], Boom(DIM), 1500, FP)

    assert conn.execute("SELECT content_hash FROM documents").fetchone()[0] == "h1"
    assert counts(conn) == (1, 1)


@pytest.mark.integration
def test_keyword_index_is_populated(conn):
    run(conn, [doc("tasks/rollback.md", "Use kubectl rollout undo to revert.", "h1")])

    hit = conn.execute(
        "SELECT count(*) FROM chunks WHERE tsv @@ websearch_to_tsquery('english', 'reverting')"
    ).fetchone()[0]

    assert hit == 1
