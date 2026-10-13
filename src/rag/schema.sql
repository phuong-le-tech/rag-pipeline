-- {embedding_dim} is substituted by rag.db.init_db (validated integer).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL UNIQUE,          -- path or URL, stable identity for re-ingestion
    content_hash TEXT NOT NULL,                 -- sha256 of raw content, skip unchanged docs
    metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Existing databases predate these columns; CREATE TABLE IF NOT EXISTS would skip them.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS collection TEXT NOT NULL DEFAULT 'k8s';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();
CREATE INDEX IF NOT EXISTS documents_collection ON documents (collection);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord         INT NOT NULL,
    heading     TEXT NOT NULL DEFAULT '',
    content     TEXT NOT NULL,
    embedding   VECTOR({embedding_dim}),
    tsv         TSVECTOR GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', heading), 'A') ||
                    setweight(to_tsvector('english', content), 'B')
                ) STORED,
    UNIQUE (document_id, ord)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin
    ON chunks USING gin (tsv);
