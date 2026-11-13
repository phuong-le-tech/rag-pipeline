"""HTTP API for the web UI: each retrieval stage and the answer stream as SSE events."""

import functools
import importlib.util
import json
import logging
import re
import threading
import time
import traceback
import uuid
import weakref
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from email.parser import BytesParser
from email.policy import HTTP
from typing import Annotated, Any, Literal

import anthropic
import openai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, StringConstraints
from starlette.concurrency import run_in_threadpool

from rag.agent.loop import (
    DEFAULT_LIMITS,
    AgentResult,
    Limits,
    ToolCallEvent,
    ToolResultEvent,
    agent_limits,
    run_agent,
)
from rag.agent.tools import SearchTool
from rag.api.guards import DailyBudget, RateLimiter
from rag.config import Settings
from rag.generate.answer import Answer, stream_answer
from rag.generate.generator import Generator, ToolGenerator
from rag.generate.prompt import GENERIC_SYSTEM_PROMPT, SYSTEM_PROMPT
from rag.ingest.loaders import PDF_MAGIC, pdf_to_text
from rag.ingest.upload import MAX_UPLOAD_BYTES
from rag.lc.llm import LANGCHAIN_BACKENDS
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
COLLECTION_ID = r"^[0-9a-f]{32}$"
PURGE_EVERY_S = 600  # expired uploads are gone at most 10 min after their TTL
MULTIPART_OVERHEAD = 4096  # headers and boundaries around the file in a multipart body


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    k: int = Field(default=5, ge=3, le=10)
    rerank: bool = True
    max_one_per_page: bool = True
    collection: str | None = Field(default=None, pattern=COLLECTION_ID)  # None = Kubernetes docs
    impl: Literal["handwritten", "langchain"] = "handwritten"
    agent: bool = False  # the model decides what to search; several model calls per question


@dataclass(frozen=True)
class Services:
    retriever_factory: Callable[[], AbstractContextManager[Any]]
    generator: Generator
    stats: Callable[..., dict]  # stats(collection="k8s")
    # index_upload(collection, filename, text) -> (event, data) pairs; see ingest.upload
    index_upload: Callable[[str, str, str], Iterator[tuple[str, dict]]] | None = None
    # Builds the LangChain generator on first use; None when the extra or backend is unavailable.
    lc_generator: Callable[[], Generator] | None = None
    # Agent mode: a fresh tool-calling LLM (and LangChain chat model) per ask; None = unavailable.
    tool_generator: Callable[[], ToolGenerator] | None = None
    lc_agent_model: Callable[[], Any] | None = None
    agent_limits: Limits = DEFAULT_LIMITS

    def impls(self) -> list[str]:
        return ["handwritten", "langchain"] if self.lc_generator else ["handwritten"]

    def agent_available(self) -> bool:
        return self.tool_generator is not None


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
    from rag.generate.generator import make_generator, make_tool_generator
    from rag.ingest.upload import ingest_upload, purge_expired
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

    def stats(collection: str = "k8s") -> dict:
        with connect(settings) as conn:
            docs, chunks = conn.execute(
                "SELECT (SELECT count(*) FROM documents WHERE collection = %(c)s), "
                "(SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE d.collection = %(c)s)",
                {"c": collection},
            ).fetchone()
        return {"docs": docs, "chunks": chunks}

    def index_upload(collection: str, filename: str, text: str) -> Iterator[tuple[str, dict]]:
        with connect(settings) as conn:
            register_vector(conn)
            purge_expired(conn, settings.upload_ttl_hours)
            yield from ingest_upload(conn, embedder, settings, collection, filename, text)

    def purge_loop() -> None:
        # The privacy page promises deletion within the TTL even when nobody uploads, so expiry
        # can't wait for the next upload. Daemon thread: one replica, dies with the process.
        while True:
            time.sleep(PURGE_EVERY_S)
            try:
                with connect(settings) as conn:
                    purge_expired(conn, settings.upload_ttl_hours)
            except Exception:
                log.exception("upload purge failed")

    threading.Thread(target=purge_loop, name="upload-purge", daemon=True).start()

    lc_generator = lc_agent_model = None
    if importlib.util.find_spec("langchain_core") and settings.llm_backend in LANGCHAIN_BACKENDS:

        @functools.cache
        def lc_generator() -> Generator:
            from rag.lc.chain import LangChainGenerator
            from rag.lc.llm import make_chat_model

            return LangChainGenerator(make_chat_model(settings))

        def lc_agent_model():
            from rag.lc.factory import make_agent_model

            return make_agent_model(settings)

    tool_generator = None
    if settings.agent_enabled:

        def tool_generator() -> ToolGenerator:
            return make_tool_generator(settings)

    return Services(
        retriever_factory,
        make_generator(settings),
        stats,
        index_upload,
        lc_generator,
        tool_generator,
        lc_agent_model,
        agent_limits(settings),
    )


def retrieve(
    impl: str, retriever: Any, question: str, config: RetrieverConfig, on_stage: Any
) -> tuple[Hit, ...]:
    if impl == "langchain":  # same search, called through the LangChain retriever interface
        from rag.lc.retriever import HybridRetriever, hits_of

        docs = HybridRetriever(retriever=retriever, config=config, on_stage=on_stage).invoke(
            question
        )
        return hits_of(docs)
    return tuple(retriever.retrieve(question, config, on_stage=on_stage))


def run_ask(
    question: str,
    config: RetrieverConfig,
    services: Services,
    budget: DailyBudget,
    release: Callable[[], None],
    impl: str = "handwritten",
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
            hits = retrieve(impl, retriever, question, config, on_stage)
        yield from staged
        # The passages the LLM will read (after rerank and the per-page cap).
        yield sse("context", {"hits": [hit_json(h) for h in hits]})
        generator = (
            services.lc_generator() if impl == "langchain" else services.generator
        )  # built before the budget is spent
        if hits and not budget.take():
            status = "daily_limit"
            yield sse("error", {"code": "daily_limit", "message": DAILY_LIMIT_MSG})
            return
        gen_start, generating = time.perf_counter(), True
        result: Answer | None = None
        system = SYSTEM_PROMPT if config.collection == "k8s" else GENERIC_SYSTEM_PROMPT
        for item in stream_answer(question, hits, generator, system):
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
                "impl": impl,
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
        log.info(json.dumps({"id": rid, "impl": impl, "status": status, "ms": timings}))


def _agent_events(
    impl: str,
    question: str,
    search: SearchTool,
    services: Services,
    budget: DailyBudget,
    generic: bool = False,
) -> Iterator[ToolCallEvent | ToolResultEvent | AgentResult]:
    if impl == "langchain":
        from rag.lc.agent import run_lc_agent

        # The graph returns at the end, so its steps are replayed after the run.
        result = run_lc_agent(
            question,
            services.lc_agent_model(),
            search,
            services.agent_limits,
            take_budget=budget.take,
            generic=generic,
        )
        yield from result.steps
        yield result
    else:
        yield from run_agent(
            question,
            services.tool_generator(),
            search,
            services.agent_limits,
            budget.take,
            generic=generic,
        )


def run_ask_agent(
    question: str,
    config: RetrieverConfig,
    services: Services,
    budget: DailyBudget,
    release: Callable[[], None],
    impl: str = "handwritten",
) -> Iterator[str]:
    """One agent ask as SSE: stages and a step per search, then context and done or error.

    The answer is not streamed token by token. Every model call is charged to the daily budget.
    """
    rid, start = uuid.uuid4().hex[:8], time.perf_counter()
    timings: dict[str, int] = {}
    status, result = "ok", None
    try:
        calls: dict[int, ToolCallEvent] = {}
        with services.retriever_factory() as retriever:
            search = SearchTool(retriever, config)
            generic = config.collection != "k8s"
            for event in _agent_events(impl, question, search, services, budget, generic):
                if isinstance(event, ToolCallEvent):
                    calls[event.step] = event
                elif isinstance(event, ToolResultEvent):
                    for name, hits, ms in event.stages:
                        timings[name] = timings.get(name, 0) + round(ms)
                        hits_json = [hit_json(h) for h in hits][:STAGE_HITS]
                        yield sse(
                            "stage",
                            {"name": name, "ms": round(ms), "hits": hits_json, "step": event.step},
                        )
                    k = ((calls.get(event.step) or event).arguments or {}).get("k")
                    yield sse(
                        "step",
                        {
                            "step": event.step,
                            "query": event.query,
                            "k": k,
                            "numbers": list(event.numbers),
                            "duplicate": event.duplicate,
                            "error": event.error,
                            "ms": round(event.ms),
                        },
                    )
                else:
                    result = event
        # Every passage the agent saw, in citation-number order: hits[n - 1] is [n].
        yield sse("context", {"hits": [hit_json(h) for h in result.answer.retrieved]})
        if result.stopped == "daily_limit":
            status = "daily_limit"
            yield sse("error", {"code": "daily_limit", "message": DAILY_LIMIT_MSG})
            return
        answer = result.answer
        yield sse(
            "done",
            {
                "text": answer.text,
                "refused": answer.refused,
                "sources": [{"n": s.number, "id": s.hit.chunk_id} for s in answer.sources],
                "invalid": list(answer.invalid_citations),
                "impl": impl,
                "agent": True,
                "searches": result.tool_calls,
                "llm_calls": result.llm_calls,
                "stopped": result.stopped,
                "ms": round((time.perf_counter() - start) * 1000),
            },
        )
    except Exception as exc:
        llm = isinstance(exc, openai.OpenAIError | anthropic.AnthropicError)
        status = "llm_failed" if llm else "failed"
        log.exception("agent ask %s failed", rid)
        message = LLM_FAILED_MSG if llm else SERVER_FAILED_MSG
        yield sse("error", {"code": status, "message": message})
    finally:
        release()
        line = {"id": rid, "impl": impl, "agent": True, "status": status, "ms": timings}
        if result is not None:
            line.update(llm_calls=result.llm_calls, searches=result.tool_calls)
        log.info(json.dumps(line))


def run_upload(
    collection: str,
    filename: str,
    text: str,
    services: Services,
    release: Callable[[], None],
) -> Iterator[str]:
    """One upload as SSE: stage events, then done or error. Never logs content or filenames."""
    rid, start, status = uuid.uuid4().hex[:8], time.perf_counter(), "ok"
    try:
        for event, data in services.index_upload(collection, filename, text):
            yield sse(event, data)
    except ValueError as exc:  # our own limits and empty-input checks; the message is ours
        status = "rejected"
        yield sse("error", {"code": status, "message": str(exc)})
    except Exception as exc:
        status = "failed"
        # Type and stack only: DB error messages can quote values (the collection id, the
        # filename), and upload data must never reach the logs.
        stack = "".join(traceback.format_tb(exc.__traceback__))
        log.error("upload %s failed: %s\n%s", rid, type(exc).__name__, stack)
        yield sse("error", {"code": status, "message": SERVER_FAILED_MSG})
    finally:
        release()
        log.info(
            json.dumps(
                {"id": rid, "status": status, "ms": round((time.perf_counter() - start) * 1000)}
            )
        )


def parse_upload(content_type: str, body: bytes) -> tuple[str, bytes, str | None]:
    """(filename, raw bytes, collection) from a multipart body (``file`` part, optional
    ``collection`` part) or JSON ``{"text": ..., "collection"?: ...}``. The collection id is the
    only key to a collection, so it travels in the body: URLs end up in proxy access logs."""
    if content_type.startswith("application/json"):
        try:
            data = json.loads(body)
            text, collection = data["text"], data.get("collection")
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ValueError('JSON body must be {"text": "..."}') from None
        if not isinstance(text, str):
            raise ValueError('"text" must be a string')
        if collection is not None and not isinstance(collection, str):
            raise ValueError('"collection" must be a string')
        return "pasted.md", text.encode("utf-8"), collection
    if content_type.startswith("multipart/form-data"):
        # stdlib parser: python-multipart is not a dependency
        header = b"Content-Type: " + content_type.encode("latin-1", "replace") + b"\r\n\r\n"
        file, collection = None, None
        for part in BytesParser(policy=HTTP).parsebytes(header + body).iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "file" and file is None:
                file = (part.get_filename() or "upload.md", part.get_payload(decode=True) or b"")
            elif name == "collection":
                raw = part.get_payload(decode=True) or b""
                collection = raw.decode("ascii", "replace").strip()
        if file is None:
            raise ValueError('multipart body needs a "file" part')
        return *file, collection
    raise ValueError("send multipart/form-data with a file, or JSON {text}")


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

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        # JSON and event streams only: no sniffing, never framed, never referred from.
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    if services is not None:
        app.state.services = services
    app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
    app.state.upload_limiter = RateLimiter(settings.upload_rate_limit_per_minute)
    app.state.budget = DailyBudget(settings.daily_llm_limit)
    app.state.slots = threading.Semaphore(settings.max_concurrent_asks)
    llm_model = (
        settings.openrouter_model or ""
        if settings.llm_backend == "openrouter"
        else settings.llm_model
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
    def collection(request: Request, id: str | None = None):
        if id is not None and not re.fullmatch(COLLECTION_ID, id):
            return JSONResponse({"error": "bad_collection"}, status_code=422)
        stats = request.app.state.services.stats(*([id] if id else []))
        if id is not None and not stats["docs"]:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return {
            "name": "Your documents" if id else "Kubernetes docs",
            **stats,
            "llm_model": llm_model,
            "embedding_model": settings.embedding_model_name(),
            "impls": request.app.state.services.impls(),
            "agent": request.app.state.services.agent_available(),
        }

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request):
        if body.impl not in request.app.state.services.impls():
            return JSONResponse({"error": "impl_unavailable"}, status_code=422)
        if body.agent and not request.app.state.services.agent_available():
            return JSONResponse({"error": "agent_unavailable"}, status_code=422)
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
        if body.collection:
            config = replace(config, collection=body.collection)
        # An upload is often one file, so a per-file cap would leave a single passage.
        if body.collection or not body.max_one_per_page:
            config = replace(config, max_per_source=None)
        release = slot_releaser(slots)
        run = run_ask_agent if body.agent else run_ask
        stream = run(
            body.question,
            config,
            request.app.state.services,
            request.app.state.budget,
            release,
            body.impl,
        )
        # A client that disconnects before the first chunk makes Starlette drop the generator
        # unstarted, so its `finally` never runs; release the slot when it is collected.
        weakref.finalize(stream, release)
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/collections")
    async def upload(request: Request):
        """New collection, or with a ``collection`` id in the body add the file to that one."""
        wait = request.app.state.upload_limiter.check(client_key(request))
        if wait:
            return JSONResponse(
                {"error": "rate_limited", "retry_after": wait},
                status_code=429,
                headers={"Retry-After": str(wait)},
            )
        cap = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD
        if int(request.headers.get("content-length") or 0) > cap:
            return JSONResponse({"error": "too_large"}, status_code=413)
        body = bytearray()
        async for chunk in request.stream():  # stop reading as soon as the cap is crossed
            body += chunk
            if len(body) > cap:
                return JSONResponse({"error": "too_large"}, status_code=413)
        try:
            filename, raw, collection = parse_upload(
                request.headers.get("content-type", ""), bytes(body)
            )
        except ValueError as exc:
            return JSONResponse({"error": "bad_upload", "message": str(exc)}, status_code=422)
        if collection is not None and not re.fullmatch(COLLECTION_ID, collection):
            return JSONResponse({"error": "bad_collection"}, status_code=422)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "too_large"}, status_code=413)
        # Take the slot before parsing: PDF parsing is the costly step, so it must be capped too.
        slots = request.app.state.slots
        if not slots.acquire(blocking=False):
            return JSONResponse({"error": "busy"}, status_code=503)
        release = slot_releaser(slots)
        try:
            if raw.startswith(PDF_MAGIC):  # CPU-bound: keep it off the event loop
                text = await run_in_threadpool(pdf_to_text, raw)
            else:
                text = raw.decode("utf-8")
            services = request.app.state.services
            # Only add to a collection that exists: a client must not pick its own (guessable) id.
            if collection is not None and not services.stats(collection)["docs"]:
                release()
                return JSONResponse({"error": "not_found"}, status_code=404)
        except UnicodeDecodeError:
            release()
            return JSONResponse({"error": "not_utf8"}, status_code=422)
        except ValueError as exc:
            release()
            return JSONResponse({"error": "bad_upload", "message": str(exc)}, status_code=422)
        except BaseException:
            release()
            raise
        stream = run_upload(collection or uuid.uuid4().hex, filename, text, services, release)
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
