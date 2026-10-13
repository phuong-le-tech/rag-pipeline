"""The two first-stage searches: dense (pgvector) and keyword (Postgres full-text).

Both return ``Hit`` lists best-first, with scores on *different* scales (cosine similarity vs.
``ts_rank_cd``). Never compare those scores across methods; fuse by rank (see ``fusion.rrf``).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import psycopg
from psycopg import sql

HNSW_DEFAULT_EF_SEARCH = 40

SELECT_HIT = """
    c.id, d.source, d.metadata->>'url', d.metadata->>'title', c.heading, c.content,
    coalesce((d.metadata->>'generated')::boolean, false)
"""
GENERATED_FILTER = (
    "(%(include_generated)s OR NOT coalesce((d.metadata->>'generated')::boolean, false))"
)
COLLECTION_FILTER = "d.collection = %(collection)s"

DENSE_SQL = f"""
SELECT * FROM (
    SELECT {SELECT_HIT}, c.embedding <=> %(vec)s AS distance
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE {GENERATED_FILTER} AND {COLLECTION_FILTER}
    ORDER BY c.embedding <=> %(vec)s
    LIMIT %(limit)s
) candidates
ORDER BY distance
"""

# OR semantics: plainto_tsquery ANDs every word ('roll' & 'back' & 'deploy'); swapping & for |
# keeps recall for natural-language questions while ts_rank_cd still favours chunks matching
# more (and heading-weighted) terms.
KEYWORD_SQL = f"""
SELECT {SELECT_HIT}, ts_rank_cd(c.tsv, q) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id,
     to_tsquery('english', replace(plainto_tsquery('english', %(query)s)::text, ' & ', ' | ')) q
WHERE c.tsv @@ q AND {GENERATED_FILTER} AND {COLLECTION_FILTER}
ORDER BY score DESC, c.id
LIMIT %(limit)s
"""


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    score: float
    source: str
    url: str
    title: str
    heading: str
    content: str
    generated: bool


def _hit(row: tuple, score: float) -> Hit:
    chunk_id, source, url, title, heading, content, generated = row[:7]
    return Hit(chunk_id, score, source, url, title, heading, content, generated)


def _configure_hnsw(conn: psycopg.Connection, limit: int) -> None:
    # ef_search caps how many candidates HNSW returns: below LIMIT you silently get fewer rows.
    # relaxed_order iterative scans keep searching when a WHERE filter discards candidates.
    ef_search = max(HNSW_DEFAULT_EF_SEARCH, limit)
    conn.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(ef_search)))
    conn.execute("SET hnsw.iterative_scan = relaxed_order")


def dense_search(
    conn: psycopg.Connection,
    query_vector: Sequence[float],
    limit: int,
    include_generated: bool = True,
    collection: str = "k8s",
) -> list[Hit]:
    _configure_hnsw(conn, limit)
    params = {
        "vec": np.asarray(query_vector, dtype=np.float32),
        "limit": limit,
        "include_generated": include_generated,
        "collection": collection,
    }
    return [_hit(row, 1.0 - row[7]) for row in conn.execute(DENSE_SQL, params)]


def keyword_search(
    conn: psycopg.Connection,
    query: str,
    limit: int,
    include_generated: bool = True,
    collection: str = "k8s",
) -> list[Hit]:
    params = {
        "query": query,
        "limit": limit,
        "include_generated": include_generated,
        "collection": collection,
    }
    return [_hit(row, row[7]) for row in conn.execute(KEYWORD_SQL, params)]
