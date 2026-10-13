import pytest

from rag.embed import FakeEmbedder
from rag.ingest.loaders import Document
from rag.ingest.pipeline import ingest
from rag.retrieval.retriever import Retriever, RetrieverConfig
from rag.retrieval.search import dense_search, keyword_search

DIM = 8
EMB = FakeEmbedder(DIM)

DOCS = [
    ("tasks/rollback.md", "Rollback", "Use kubectl rollout undo to revert a Deployment.", False),
    ("concepts/service.md", "Service", "A Service exposes Pods behind a stable virtual IP.", False),
    ("reference/flags.md", "Flags", "--advertise-address string sets the apiserver IP.", True),
    (
        "concepts/rollout-notes.md",
        "Notes",
        "Rollback history is limited by revisionHistoryLimit.",
        False,
    ),
]


@pytest.fixture
def indexed(conn):
    docs = [
        Document(
            source=src,
            url=f"https://kubernetes.io/docs/{src.removesuffix('.md')}/",
            title=title,
            section=src.split("/")[0],
            text=f"# {title}\n\n{body}",
            content_hash=src,
            generated=gen,
        )
        for src, title, body, gen in DOCS
    ]
    ingest(conn, docs, EMB, max_chars=1500, fingerprint="fake")
    return conn


def sources(hits):
    return [h.source for h in hits]


@pytest.mark.integration
def test_dense_search_ranks_identical_text_first_with_metadata(indexed):
    query_vec = EMB.embed(["Service\n\nA Service exposes Pods behind a stable virtual IP."])[0]

    hits = dense_search(indexed, query_vec, limit=4)

    top = hits[0]
    assert top.source == "concepts/service.md"
    assert top.score == pytest.approx(1.0, abs=1e-4)
    assert top.url == "https://kubernetes.io/docs/concepts/service/"
    assert top.heading == "Service"
    assert len(hits) == 4


@pytest.mark.integration
def test_keyword_search_matches_any_term_with_stemming(indexed):
    hits = keyword_search(indexed, "reverting quickly", limit=10)

    assert sources(hits) == ["tasks/rollback.md"]


@pytest.mark.integration
def test_keyword_search_ranks_heading_match_above_body_match(indexed):
    hits = keyword_search(indexed, "rollback", limit=10)

    assert sources(hits)[0] == "tasks/rollback.md"
    assert set(sources(hits)) == {"tasks/rollback.md", "concepts/rollout-notes.md"}


@pytest.mark.integration
def test_keyword_search_with_only_stopwords_returns_nothing(indexed):
    assert keyword_search(indexed, "how do I", limit=10) == []


@pytest.mark.integration
def test_generated_pages_can_be_excluded(indexed):
    query_vec = EMB.embed_query("anything")

    dense = dense_search(indexed, query_vec, limit=10, include_generated=False)
    keyword = keyword_search(indexed, "apiserver", limit=10, include_generated=False)

    assert "reference/flags.md" not in sources(dense)
    assert keyword == []


@pytest.mark.integration
def test_hybrid_fuses_both_lists_and_respects_k(indexed):
    retriever = Retriever(indexed, EMB)

    hits = retriever.retrieve("rollback", RetrieverConfig(mode="hybrid", k=2, candidates=10))

    # Found by both lists => RRF >= 1/62 + 1/64, beating any dense-only hit (<= 1/61).
    assert set(sources(hits)) == {"tasks/rollback.md", "concepts/rollout-notes.md"}
    assert hits[0].score >= hits[1].score


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["dense", "keyword"])
def test_single_mode_returns_that_list(indexed, mode):
    hits = Retriever(indexed, EMB).retrieve("rollback", RetrieverConfig(mode=mode, k=1))

    assert len(hits) == 1


def test_retriever_rejects_blank_query():
    with pytest.raises(ValueError):
        Retriever(conn=None, embedder=EMB).retrieve("  ", RetrieverConfig())


@pytest.mark.parametrize(
    "kwargs", [{"k": 0}, {"k": 10, "candidates": 5}, {"rrf_k": 0}, {"mode": "magic"}]
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        RetrieverConfig(**kwargs)


@pytest.mark.integration
def test_retrieve_reports_each_stage(indexed):
    seen = []
    hits = Retriever(indexed, EMB).retrieve(
        "rollback",
        RetrieverConfig(mode="hybrid", k=2, candidates=4),
        on_stage=lambda name, stage_hits, ms: seen.append((name, len(stage_hits), ms >= 0)),
    )
    assert [s[0] for s in seen] == ["dense", "keyword", "fusion"]
    assert all(ok for *_, ok in seen)
    assert len(hits) == 2


@pytest.mark.integration
def test_other_collection_is_never_returned_by_default(indexed):
    other = Document(
        source="upload:x/rollback.md",
        url="",
        title="Mine",
        section="",
        text="# Mine\n\nRollback notes from an uploaded file.",
        content_hash="x",
        generated=False,
    )
    ingest(indexed, [other], EMB, max_chars=1500, fingerprint="fake", prune=False, collection="x")
    vec = EMB.embed(["Mine\n\nRollback notes from an uploaded file."])[0]

    default_dense = dense_search(indexed, vec, limit=10)
    default_keyword = keyword_search(indexed, "rollback uploaded", limit=10)
    scoped = dense_search(indexed, vec, limit=10, collection="x")

    assert "upload:x/rollback.md" not in sources(default_dense)
    assert "upload:x/rollback.md" not in sources(default_keyword)
    assert len(default_dense) == len(DOCS)  # filtered HNSW scan still returns every k8s chunk
    assert sources(scoped) == ["upload:x/rollback.md"]
    assert sources(keyword_search(indexed, "rollback", limit=10, collection="x")) == [
        "upload:x/rollback.md"
    ]
    cfg = RetrieverConfig(mode="dense", k=3, candidates=10, collection="x")
    assert sources(Retriever(indexed, EMB).retrieve("rollback", cfg)) == ["upload:x/rollback.md"]
