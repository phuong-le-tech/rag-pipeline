FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy FASTEMBED_CACHE_PATH=/models
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra langchain --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra langchain
# Bake the embedding and reranker models into the image: no 150 MB download on cold start.
RUN uv run --no-sync python -c "from rag.config import Settings; from rag.embed import make_embedder; \
from rag.retrieval.rerank import make_reranker; s = Settings(_env_file=None); \
make_embedder(s).embed_query('x'); make_reranker(s).model"
RUN useradd --create-home app && chown -R app /app /models
USER app
# The rate limiter reads the client IP from the last X-Forwarded-For entry (rag.api.app.client_key).
# --no-access-log: access lines contain IPs; the app logs one JSON line per ask instead.
CMD ["sh", "-c", "exec uv run --no-sync uvicorn rag.api.app:main --factory --host 0.0.0.0 --port ${PORT:-8000} --no-access-log"]
