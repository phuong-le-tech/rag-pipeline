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


@pytest.mark.integration
def test_prune_and_versions_are_scoped_to_the_collection(conn):
    ingest(
        conn,
        [doc("up/mine.md", "uploaded", "u1")],
        FakeEmbedder(DIM),
        max_chars=1500,
        fingerprint=FP,
        prune=False,
        collection="x",
    )

    stats = run(conn, [doc("a/one.md", "alpha", "h1")])  # k8s run with prune=True

    assert stats.deleted == 0
    assert conn.execute("SELECT source, collection FROM documents ORDER BY id").fetchall() == [
        ("up/mine.md", "x"),
        ("a/one.md", "k8s"),
    ]


@pytest.mark.integration
def test_upload_ingest_emits_stages_and_scopes_retrieval(conn):
    from rag.config import Settings
    from rag.ingest.upload import ingest_upload
    from rag.retrieval.retriever import Retriever, RetrieverConfig

    run(conn, [doc("a/one.md", "alpha pods", "h1")])
    settings = Settings(_env_file=None, embedding_backend="fake", embedding_dim=DIM)

    ev = list(
        ingest_upload(
            conn, FakeEmbedder(DIM), settings, "u" * 32, "n.md", "# Mine\n\nalpha pods too"
        )
    )

    assert [d["name"] for e, d in ev if e == "stage"] == [
        "load",
        "clean",
        "chunk",
        "embed",
        "store",
    ]
    assert ev[-1][0] == "done" and ev[-1][1]["chunks"] >= 1
    config = RetrieverConfig(mode="keyword", collection="u" * 32, rerank=False)
    hits = Retriever(conn, FakeEmbedder(DIM), None).retrieve("alpha pods", config)
    assert hits and all(h.source.startswith("upload:") for h in hits)


@pytest.mark.integration
def test_purge_expired_removes_old_uploads_and_never_k8s(conn):
    from rag.ingest.upload import purge_expired

    ingest(conn, [doc("up/old.md", "x", "u1")], FakeEmbedder(DIM), 1500, FP, False, collection="x")
    ingest(conn, [doc("up/new.md", "x", "u2")], FakeEmbedder(DIM), 1500, FP, False, collection="y")
    run(conn, [doc("a/one.md", "alpha", "h1")])
    conn.execute(
        "UPDATE documents SET created_at = now() - interval '25 hours' WHERE collection <> 'y'"
    )
    conn.commit()

    assert purge_expired(conn, 24) == 1  # 'x' only; old 'k8s' row is kept
    assert [r[0] for r in conn.execute("SELECT collection FROM documents ORDER BY id")] == [
        "y",
        "k8s",
    ]
    assert counts(conn)[1] == 2  # chunks of the deleted doc cascaded


def test_upload_over_chunk_limit_is_rejected_before_embedding():
    from rag.config import Settings
    from rag.embed import FakeEmbedder
    from rag.ingest.upload import MAX_UPLOAD_CHUNKS, ingest_upload

    text = "\n\n".join(f"# H{i}\n\n" + "word " * 20 for i in range(MAX_UPLOAD_CHUNKS + 5))
    events = ingest_upload(None, FakeEmbedder(8), Settings(), "a" * 32, "big.md", text)

    with pytest.raises(ValueError, match="chunks"):
        list(events)
