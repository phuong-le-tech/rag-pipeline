#!/usr/bin/env bash
# Copy the local index (documents + chunks, vectors included) into a remote Postgres.
# Usage: scripts/publish_index.sh "$TARGET_DATABASE_URL"   (replaces the target's index)
set -euo pipefail
TARGET="${1:?usage: publish_index.sh TARGET_DATABASE_URL}"
SOURCE_SERVICE="${SOURCE_SERVICE:-db}"  # docker compose service holding the local index
RAG_DATABASE_URL="$TARGET" uv run rag init-db
docker compose exec -T "$SOURCE_SERVICE" psql "$TARGET" -v ON_ERROR_STOP=1 \
  -c "TRUNCATE chunks, documents RESTART IDENTITY CASCADE"
# pg_dump/psql run inside the pgvector:pg17 container, so their version matches the server.
docker compose exec -T "$SOURCE_SERVICE" pg_dump -U rag --data-only --no-owner \
  --table=documents --table=chunks rag \
  | docker compose exec -T "$SOURCE_SERVICE" psql "$TARGET" -v ON_ERROR_STOP=1 -q
docker compose exec -T "$SOURCE_SERVICE" psql "$TARGET" -c \
  "SELECT (SELECT count(*) FROM documents) AS docs, (SELECT count(*) FROM chunks) AS chunks"
