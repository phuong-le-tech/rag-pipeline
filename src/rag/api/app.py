"""HTTP API for the web UI: each retrieval stage and the answer stream as SSE events."""

import json
import logging
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from typing import Annotated, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from rag.api.guards import DailyBudget, RateLimiter
from rag.config import Settings
from rag.generate.answer import Answer, stream_answer
from rag.generate.generator import Generator
from rag.retrieval.retriever import RetrieverConfig
from rag.retrieval.search import Hit

log = logging.getLogger("rag.api")
CONTENT_LIMIT = 2000  # characters of passage text sent to the browser
STAGE_HITS = 10  # hits per stage shown in the UI
DAILY_LIMIT_MSG = (
    "Daily answer limit reached. Retrieval still works; answers resume tomorrow (UTC)."
)
LLM_FAILED_MSG = "The model call failed. Retrieval results are kept; retry in a moment."
SERVER_FAILED_MSG = "Something went wrong on the server. Please retry."


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    k: int = Field(default=5, ge=3, le=10)
    rerank: bool = True
    max_one_per_page: bool = True


@dataclass(frozen=True)
class Services:
    retriever_factory: Callable[[], AbstractContextManager[Any]]
    generator: Generator
    stats: Callable[[], dict]


def hit_json(h: Hit) -> dict:
    return {
        "id": h.chunk_id,
        "heading": h.heading,
        "title": h.title,
        "url": h.url,
        "content": h.content[:CONTENT_LIMIT],
        "score": round(h.score, 5),
    }


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def default_services(settings: Settings) -> Services:  # pragma: no cover - needs DB + models
    from pgvector.psycopg import register_vector

    from rag.db import connect
    from rag.embed import make_embedder
    from rag.generate.generator import make_generator
    from rag.retrieval.rerank import make_reranker
    from rag.retrieval.retriever import Retriever

    embedder, reranker = make_embedder(settings), make_reranker(settings)
    embedder.embed_query("warm up")  # load ONNX models at boot, not on the first request
    _ = reranker.model

    @contextmanager
    def retriever_factory():
        # shortcut: one connection per request; add psycopg_pool if traffic grows.
        with connect(settings) as conn:
            register_vector(conn)
            yield Retriever(conn, embedder, reranker)

    def stats() -> dict:
        with connect(settings) as conn:
            docs, chunks = conn.execute(
                "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
            ).fetchone()
        return {"docs": docs, "chunks": chunks}

    return Services(retriever_factory, make_generator(settings), stats)


def run_ask(
    question: str,
    config: RetrieverConfig,
    services: Services,
    budget: DailyBudget,
    release: Callable[[], None],
) -> Iterator[str]:
    """One ask as SSE: stages, tokens, then done or error. Always releases its slot."""
    rid, start = uuid.uuid4().hex[:8], time.perf_counter()
    timings: dict[str, int] = {}
    status, generating = "ok", False
    try:
        staged: list[str] = []

        def on_stage(name: str, hits: list[Hit], ms: float) -> None:
            timings[name] = round(ms)
            payload = {
                "name": name,
                "ms": round(ms),
                "hits": [hit_json(h) for h in hits][:STAGE_HITS],
            }
            staged.append(sse("stage", payload))

        with services.retriever_factory() as retriever:
            hits = tuple(retriever.retrieve(question, config, on_stage=on_stage))
        yield from staged
        # The passages the LLM will read (after rerank and the per-page cap).
        yield sse("context", {"hits": [hit_json(h) for h in hits]})
        if hits and not budget.take():
            status = "daily_limit"
            yield sse("error", {"code": "daily_limit", "message": DAILY_LIMIT_MSG})
            return
        gen_start, generating = time.perf_counter(), True
        result: Answer | None = None
        for item in stream_answer(question, hits, services.generator):
            if isinstance(item, Answer):
                result = item
            else:
                yield sse("token", {"text": item})
        generating = False
        timings["generate"] = round((time.perf_counter() - gen_start) * 1000)
        yield sse(
            "done",
            {
                "text": result.text,
                "refused": result.refused,
                "sources": [{"n": s.number, "id": s.hit.chunk_id} for s in result.sources],
                "invalid": list(result.invalid_citations),
                "ms": round((time.perf_counter() - start) * 1000),
            },
        )
    except Exception:  # the stream already started: report in-band instead of a broken response
        status = "llm_failed" if generating else "failed"
        log.exception("ask %s failed", rid)
        message = LLM_FAILED_MSG if generating else SERVER_FAILED_MSG
        yield sse("error", {"code": status, "message": message})
    finally:
        release()
        log.info(json.dumps({"id": rid, "status": status, "ms": timings}))


def client_key(request: Request) -> str:
    """Rate-limit key: the last X-Forwarded-For entry, the one the edge proxy (Railway) appends.
    Earlier entries are client-controlled and would let anyone mint fresh buckets."""
    forwarded = request.headers.get("x-forwarded-for", "")
    last = forwarded.rsplit(",", 1)[-1].strip()
    return last or (request.client.host if request.client else "?")


def slot_releaser(slots: threading.Semaphore) -> Callable[[], None]:
    """Idempotent release: the stream's ``finally`` and garbage collection may both call it."""
    lock, held = threading.Lock(), [True]

    def release() -> None:
        with lock:
            if held[0]:
                held[0] = False
                slots.release()

    return release


def create_app(settings: Settings, services: Services | None = None) -> FastAPI:
    from rag.cli import ask_config  # holds the measured `rag ask` defaults

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if services is None:  # tests inject services; prod loads DB + models at boot
            app.state.services = default_services(settings)
        yield

    app = FastAPI(
        title="Measured RAG", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    if services is not None:
        app.state.services = services
    app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
    app.state.budget = DailyBudget(settings.daily_llm_limit)
    app.state.slots = threading.Semaphore(settings.max_concurrent_asks)
    llm_model = (
        settings.openrouter_model if settings.llm_backend == "openrouter" else settings.llm_model
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        try:
            chunks = request.app.state.services.stats()["chunks"]
        except Exception:
            log.exception("readiness check failed")
            return JSONResponse({"status": "db_unreachable"}, status_code=503)
        return {"chunks": chunks} if chunks else JSONResponse({"chunks": 0}, status_code=503)

    @app.get("/api/collection")
    def collection(request: Request) -> dict:
        return {
            "name": "Kubernetes docs",
            **request.app.state.services.stats(),
            "llm_model": llm_model,
            "embedding_model": settings.embedding_model_name(),
        }

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request):
        wait = request.app.state.limiter.check(client_key(request))
        if wait:
            return JSONResponse(
                {"error": "rate_limited", "retry_after": wait},
                status_code=429,
                headers={"Retry-After": str(wait)},
            )
        slots = request.app.state.slots
        if not slots.acquire(blocking=False):
            return JSONResponse({"error": "busy"}, status_code=503)
        config = ask_config(k=body.k, rerank=body.rerank, generated=True, rewrite=False)
        if not body.max_one_per_page:
            config = replace(config, max_per_source=None)
        release = slot_releaser(slots)
        stream = run_ask(
            body.question, config, request.app.state.services, request.app.state.budget, release
        )
        # A client that disconnects before the first chunk makes Starlette drop the generator
        # unstarted, so its `finally` never runs; release the slot when it is collected.
        weakref.finalize(stream, release)
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def main() -> FastAPI:  # uvicorn factory: `uvicorn rag.api.app:main --factory`
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return create_app(Settings())
