# Production v1 (glass-box RAG, Kubernetes docs) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Testers open a public URL, ask questions about the Kubernetes docs, and watch the *real* pipeline (dense, keyword, RRF, rerank, streamed cited answer) run, safely and at near-zero cost.

**Architecture:** A FastAPI service (`src/rag/api/`) wraps the existing `Retriever` and `Generator`, streams each stage as server-sent events, and enforces rate, size, concurrency and daily-LLM limits. It runs on Railway next to a pgvector Postgres restored from the local index (`pg_dump`, no re-embedding). The Next.js UI (`web/`) runs on Vercel and replaces its in-browser mock engine with an SSE client. Uploads stay off in v1.

**Tech Stack:** Python 3.12, FastAPI + uvicorn, psycopg3, fastembed (bge-small, MiniLM cross-encoder), OpenRouter free model; Next.js 16 / React 19; Railway (Docker + Postgres pgvector), Vercel.

**Spec:** owner answers 2026-10-11: real pipeline, k8s only; hosting Vercel or Railway; conformity = security basics + legal/GDPR + production quality. Decisions in `docs/decisions.md`.

## Global Constraints

- No API key ever reaches the browser; the UI only knows `NEXT_PUBLIC_API_URL`.
- `question`: 1–500 characters after trimming; `k`: 3–10; `rerank`, `max_one_per_page`: booleans.
- Defaults: `RAG_RATE_LIMIT_PER_MINUTE=10` per client IP, `RAG_DAILY_LLM_LIMIT=300` LLM calls/day (UTC), `RAG_MAX_CONCURRENT_ASKS=2`.
- CORS: only origins listed in `RAG_CORS_ORIGINS` (comma-separated).
- Logs never contain question text or IPs; one JSON line per request (id, status, stage timings, refused).
- Default LLM: `openrouter` / `nvidia/nemotron-3-super-120b-a12b:free`.
- Uploads disabled in v1: no endpoint accepts files.
- Kubernetes docs content is CC BY 4.0: attribution visible on every page.
- Python style: ruff line length 100, frozen dataclasses, `ValueError` naming the bad value; TDD.

## Review Focus

1. Two testers ask at once while a third request arrives: the third gets HTTP 503 `{"error":"busy"}` immediately, never a hung request (test in Task 3).
2. The LLM provider fails mid-answer (429, timeout): the stream ends with an `error` event, the UI shows the error box with Retry, never a frozen caret (tests in Task 3 and Task 5).
3. Blank, whitespace-only or 5,000-character question: 422 before any model runs (test in Task 3).
4. API down or CORS misconfigured: the UI shows "Service unavailable, try again later" in the Ask panel, not an empty page (Task 5).
5. Daily LLM budget exhausted: retrieval still works and is shown, the answer slot says the daily limit was reached; no LLM call is made (test in Task 3).

---

### Task 1: Retriever exposes each stage (`on_stage`)

**Files:**
- Modify: `src/rag/retrieval/retriever.py` (`_first_stage`, `retrieve`)
- Test: `tests/test_retrieval.py` (integration, uses the existing `indexed` fixture)

**Interfaces:**
- Produces: `Retriever.retrieve(query: str, config: RetrieverConfig, on_stage: StageHook | None = None) -> list[Hit]` where `StageHook = Callable[[str, list[Hit], float], None]`; stage names `"dense"`, `"keyword"`, `"fusion"` (hybrid only), `"rerank"` (when `config.rerank`); the float is elapsed milliseconds.

- [ ] **Step 1: Write the failing test** (append to `tests/test_retrieval.py`)

```python
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
```

- [ ] **Step 2: Run it, expect FAIL** (`TypeError: unexpected keyword argument 'on_stage'`)

Run: `uv run pytest tests/test_retrieval.py::test_retrieve_reports_each_stage -v`

- [ ] **Step 3: Implement**

In `retriever.py` add `import time` and `from collections.abc import Callable`, then:

```python
StageHook = Callable[[str, list[Hit], float], None]


def _timed(name: str, on_stage: StageHook | None, fn: Callable[[], list[Hit]]) -> list[Hit]:
    start = time.perf_counter()
    hits = fn()
    if on_stage:
        on_stage(name, hits, (time.perf_counter() - start) * 1000)
    return hits
```

Rewrite `_ranked_lists` / `_first_stage` / `retrieve` to route every step through `_timed`:

```python
def _ranked_lists(self, query, config, on_stage):
    lists = []
    if config.mode in ("dense", "hybrid"):
        lists.append(
            _timed(
                "dense",
                on_stage,
                lambda: dense_search(
                    self.conn,
                    self.embedder.embed_query(query),
                    config.candidates,
                    config.include_generated,
                ),
            )
        )
    if config.mode in ("keyword", "hybrid"):
        lists.append(
            _timed(
                "keyword",
                on_stage,
                lambda: keyword_search(
                    self.conn, query, config.candidates, config.include_generated
                ),
            )
        )
    return lists


def _first_stage(self, query, config, limit, on_stage=None):
    lists = self._ranked_lists(query, config, on_stage)
    if config.mode != "hybrid":
        return lists[0][:limit]
    by_id = {hit.chunk_id: hit for hits in lists for hit in hits}

    def fuse() -> list[Hit]:
        fused = rrf([[h.chunk_id for h in hits] for hits in lists], config.rrf_k, limit)
        return [replace(by_id[chunk_id], score=score) for chunk_id, score in fused]

    return _timed("fusion", on_stage, fuse)
```

and in `retrieve(self, query, config, on_stage: StageHook | None = None)`: pass `on_stage` to `_first_stage`, and wrap the rerank call: `hits = _timed("rerank", on_stage, lambda: self.reranker.rerank(query, hits))` (bind `hits` first: `first = hits` then `lambda: self.reranker.rerank(query, first)`).

- [ ] **Step 4: Run the retrieval tests, expect PASS** (needs `docker compose up -d`)

Run: `uv run pytest tests/test_retrieval.py -v`

- [ ] **Step 5: Commit** — `git commit -m "feat(retrieval): report each stage's hits and timing via on_stage"`

---

### Task 2: Generation from already-retrieved hits, as a token stream

**Files:**
- Modify: `src/rag/generate/answer.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Produces: `stream_answer(question: str, hits: tuple[Hit, ...], generator: Generator) -> Iterator[str | Answer]` (yields each token string as it arrives, then the final `Answer` last; empty `hits` yields only the refusal `Answer`, without calling the LLM). `answer_question` keeps its signature and is rebuilt on top of it.

- [ ] **Step 1: Failing test**

```python
from rag.generate.answer import Answer, stream_answer

HIT = Hit(1, 0.9, "a.md", "https://x/a", "A", "A > B", "Use rollout undo.", False)


def test_stream_answer_yields_tokens_then_answer():
    gen = FakeGenerator("Use rollout undo [1].")
    items = list(stream_answer("How to roll back?", (HIT,), gen))
    assert items[0] == "Use rollout undo [1]."
    final = items[-1]
    assert isinstance(final, Answer) and not final.refused and final.sources[0].hit == HIT


def test_stream_answer_without_hits_refuses_without_llm():
    gen = FakeGenerator()
    items = list(stream_answer("q", (), gen))
    assert len(items) == 1 and items[0].refused and gen.calls == []
```

(`Hit` and `FakeGenerator` are already imported at the top of `tests/test_generate.py`; add the imports if not.)

- [ ] **Step 2: Run, expect FAIL** (`ImportError: stream_answer`)

Run: `uv run pytest tests/test_generate.py -q`

- [ ] **Step 3: Implement** in `answer.py` (add `from collections.abc import Iterator`; add `"stream_answer"` to `__all__`):

```python
def stream_answer(
    question: str, hits: tuple[Hit, ...], generator: Generator
) -> Iterator[str | Answer]:
    """Yield answer tokens as they arrive, then the validated ``Answer`` last."""
    if not hits:
        yield _refusal()  # nothing relevant: refuse without paying for an LLM call
        return
    system, user = build_prompt(question, hits)
    pieces = []
    for piece in generator.stream(system, user):
        pieces.append(piece)
        yield piece
    text = "".join(pieces)
    if text.strip().startswith(REFUSAL):
        yield Answer(text.strip(), (), (), True, hits)
        return
    report = check_citations(text, num_sources=len(hits))
    yield Answer(
        text, tuple(Source(n, hits[n - 1]) for n in report.cited), report.invalid, False, hits
    )


def answer_question(question, retriever, generator, config, on_text=None) -> Answer:
    """Answer ``question``. With ``on_text``, tokens are streamed to it as they arrive."""
    hits = tuple(retriever.retrieve(question, config))
    if on_text is None and hits:
        # Non-streaming path keeps using generate(): evals rely on it.
        system, user = build_prompt(question, hits)
        text = generator.generate(system, user)
        if text.strip().startswith(REFUSAL):
            return Answer(text.strip(), (), (), True, hits)
        report = check_citations(text, num_sources=len(hits))
        return Answer(
            text, tuple(Source(n, hits[n - 1]) for n in report.cited), report.invalid, False, hits
        )
    for item in stream_answer(question, hits, generator):
        if isinstance(item, Answer):
            return item
        on_text(item)
    raise AssertionError("stream_answer always ends with an Answer")
```

Keep the original type annotations on `answer_question`'s parameters.

- [ ] **Step 4: Run** `uv run pytest tests/test_generate.py tests/test_answers.py tests/test_cli.py -q` → PASS (existing `answer_question` tests must stay green).
- [ ] **Step 5: Commit** — `git commit -m "feat(generate): stream_answer yields tokens then the validated answer"`

---

### Task 3: FastAPI service with SSE and guards

**Files:**
- Create: `src/rag/api/__init__.py` (empty), `src/rag/api/guards.py`, `src/rag/api/app.py`
- Modify: `src/rag/config.py` (four settings), `pyproject.toml` (deps)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Retriever.retrieve(..., on_stage=)` (Task 1), `stream_answer` (Task 2), `ask_config` from `rag.cli`.
- Produces: `create_app(settings: Settings, services: Services | None = None) -> FastAPI`; `Services(retriever_factory: Callable[[], ContextManager[SupportsStagedRetrieve]], generator: Generator, stats: Callable[[], dict])`. HTTP: `GET /health` → `{"status":"ok"}`; `GET /ready` → 200 `{"chunks": n}` or 503; `GET /api/collection` → `{"name","docs","chunks","llm_model","embedding_model"}`; `POST /api/ask` JSON `{"question","k","rerank","max_one_per_page"}` → `text/event-stream` with events `stage {name, ms, hits:[{id, heading, title, url, content, score}]}`, `token {text}`, `done {text, refused, sources:[{n,id}], invalid:[int], ms}`, `error {code, message}`; 422 invalid input, 429 `{"error":"rate_limited","retry_after":s}`, 503 `{"error":"busy"}`.

- [ ] **Step 1: Dependencies**

`uv add fastapi "uvicorn[standard]"` and `uv add --dev httpx`.

Add to `Settings` (after `openrouter_api_key`):

```python
    cors_origins: str = "http://localhost:3000"  # comma-separated
    rate_limit_per_minute: int = Field(default=10, gt=0)
    daily_llm_limit: int = Field(default=300, ge=0)
    max_concurrent_asks: int = Field(default=2, gt=0)
```

- [ ] **Step 2: Failing tests for guards** (`tests/test_api.py`)

```python
from rag.api.guards import DailyBudget, RateLimiter


def test_rate_limiter_blocks_after_limit_and_recovers():
    t = [0.0]
    rl = RateLimiter(per_minute=2, clock=lambda: t[0])
    assert rl.check("a") == 0 and rl.check("a") == 0
    assert rl.check("a") > 0  # third call within the minute: seconds to wait
    assert rl.check("b") == 0  # other clients unaffected
    t[0] = 61.0
    assert rl.check("a") == 0


def test_daily_budget_resets_each_utc_day():
    day = ["2026-10-12"]
    b = DailyBudget(limit=1, today=lambda: day[0])
    assert b.take() and not b.take()
    day[0] = "2026-10-13"
    assert b.take()
```

- [ ] **Step 3: Run, expect FAIL**, then implement `guards.py`:

```python
"""In-process abuse guards. shortcut: per-process memory, move to Redis if we run >1 replica."""

import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime


class RateLimiter:
    """Sliding one-minute window per key. ``check`` returns 0 if allowed, else seconds to wait."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute, self.clock = per_minute, clock
        self.calls: dict[str, deque[float]] = {}

    def check(self, key: str) -> int:
        now = self.clock()
        q = self.calls.setdefault(key, deque())
        while q and now - q[0] >= 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return int(60 - (now - q[0])) + 1
        q.append(now)
        return 0


class DailyBudget:
    """Caps LLM calls per UTC day, so a free tier or a credit balance can't be drained."""

    def __init__(self, limit: int, today: Callable[[], str] | None = None) -> None:
        self.limit = limit
        self.today = today or (lambda: datetime.now(UTC).date().isoformat())
        self.day, self.used = self.today(), 0

    def take(self) -> bool:
        if self.today() != self.day:
            self.day, self.used = self.today(), 0
        if self.used >= self.limit:
            return False
        self.used += 1
        return True
```

Run `uv run pytest tests/test_api.py -q` → PASS.

- [ ] **Step 4: Failing tests for the app** (append to `tests/test_api.py`)

```python
import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from rag.api.app import Services, create_app
from rag.config import Settings
from rag.generate.generator import FakeGenerator
from rag.retrieval.search import Hit

HIT = Hit(
    7,
    0.9,
    "d.md",
    "https://kubernetes.io/docs/d/",
    "Deploy",
    "Deploy > Roll back",
    "Run kubectl rollout undo.",
    False,
)


class StagedFake:
    def retrieve(self, query, config, on_stage=None):
        for name in ("dense", "keyword", "fusion", "rerank"):
            on_stage and on_stage(name, [HIT], 1.0)
        return [HIT]


def make_client(gen=None, **overrides):
    @contextmanager
    def factory():
        yield StagedFake()

    settings = Settings(_env_file=None, **overrides)
    services = Services(
        factory, gen or FakeGenerator("Use rollout undo [1]."), lambda: {"docs": 1, "chunks": 1}
    )
    return TestClient(create_app(settings, services))


def events(resp):
    out = []
    for block in resp.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_ask_streams_stages_tokens_and_done():
    ev = events(make_client().post("/api/ask", json={"question": "roll back?"}))
    names = [e for e, _ in ev]
    assert names[:4] == ["stage"] * 4 and names[-1] == "done"
    assert ev[-1][1]["sources"] == [{"n": 1, "id": 7}] and not ev[-1][1]["refused"]


@pytest.mark.parametrize("q", ["", "   ", "x" * 501])
def test_ask_rejects_bad_questions(q):
    assert make_client().post("/api/ask", json={"question": q}).status_code == 422


def test_rate_limit_returns_429():
    c = make_client(rate_limit_per_minute=1)
    assert c.post("/api/ask", json={"question": "a"}).status_code == 200
    r = c.post("/api/ask", json={"question": "a"})
    assert r.status_code == 429 and r.json()["error"] == "rate_limited"


def test_budget_exhausted_skips_llm_but_shows_retrieval():
    gen = FakeGenerator("x [1]")
    ev = events(make_client(gen, daily_llm_limit=0).post("/api/ask", json={"question": "a"}))
    assert ev[-1] == ("error", {"code": "daily_limit", "message": ev[-1][1]["message"]})
    assert [e for e, _ in ev].count("stage") == 4 and gen.calls == []


def test_llm_failure_ends_with_error_event():
    class Boom(FakeGenerator):
        def stream(self, system, user):
            yield "partial "
            raise TimeoutError("upstream timeout")

    ev = events(make_client(Boom()).post("/api/ask", json={"question": "a"}))
    assert ev[-1][0] == "error" and ev[-1][1]["code"] == "llm_failed"


def test_busy_when_concurrency_exhausted():
    c = make_client(max_concurrent_asks=1)
    c.app.state.slots.acquire()  # simulate one ask in flight
    r = c.post("/api/ask", json={"question": "a"})
    assert r.status_code == 503 and r.json() == {"error": "busy"}


def test_health_and_collection():
    c = make_client()
    assert c.get("/health").json() == {"status": "ok"}
    body = c.get("/api/collection").json()
    assert body["chunks"] == 1 and body["llm_model"].endswith(":free")
```

- [ ] **Step 5: Run, expect FAIL** (`ModuleNotFoundError: rag.api.app`)

- [ ] **Step 6: Implement `src/rag/api/app.py`**

```python
"""HTTP API for the glass-box UI: each retrieval stage and the answer stream as SSE events."""

import json
import logging
import threading
import time
import uuid
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
from rag.retrieval.search import Hit

log = logging.getLogger("rag.api")
CONTENT_LIMIT = 2000  # characters of passage text sent to the browser
STAGE_HITS = 10  # hits per stage shown in the UI
DAILY_LIMIT_MSG = (
    "Daily answer limit reached. Retrieval still works; answers resume tomorrow (UTC)."
)
LLM_FAILED_MSG = "The model call failed. Retrieval results are kept; retry in a moment."


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


def create_app(settings: Settings, services: Services | None = None) -> FastAPI:
    from rag.cli import ask_config  # cli imports heavy deps; keep it out of module import

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if services is None:  # tests inject services; prod loads DB + models at boot
            app.state.services = default_services(settings)
        yield

    app = FastAPI(
        title="glass-box RAG", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
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
        wait = request.app.state.limiter.check(request.client.host if request.client else "?")
        if wait:
            return JSONResponse(
                {"error": "rate_limited", "retry_after": wait},
                status_code=429,
                headers={"Retry-After": str(wait)},
            )
        slots = request.app.state.slots
        if not slots.acquire(blocking=False):
            return JSONResponse({"error": "busy"}, status_code=503)
        services: Services = request.app.state.services
        config = ask_config(k=body.k, rerank=body.rerank, generated=True, rewrite=False)
        if not body.max_one_per_page:
            config = replace(config, max_per_source=None)
        return StreamingResponse(
            run_ask(body.question, config, services, request.app.state.budget, slots),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def run_ask(
    question, config, services: Services, budget: DailyBudget, slots: threading.Semaphore
) -> Iterator[str]:
    """One ask as SSE: stages, tokens, then done or error. Always releases its slot."""
    rid, start, timings, status, generating = (
        uuid.uuid4().hex[:8],
        time.perf_counter(),
        {},
        "ok",
        False,
    )
    try:
        staged: list[str] = []

        def on_stage(name: str, hits: list[Hit], ms: float) -> None:
            timings[name] = round(ms)
            staged.append(
                sse(
                    "stage",
                    {
                        "name": name,
                        "ms": round(ms),
                        "hits": [hit_json(h) for h in hits[:STAGE_HITS]],
                    },
                )
            )

        with services.retriever_factory() as retriever:
            hits = tuple(retriever.retrieve(question, config, on_stage=on_stage))
        yield from staged
        if hits and not budget.take():
            status = "daily_limit"
            yield sse("error", {"code": "daily_limit", "message": DAILY_LIMIT_MSG})
            return
        gen_start, generating = time.perf_counter(), True
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
        yield sse(
            "error",
            {
                "code": status,
                "message": LLM_FAILED_MSG
                if generating
                else "Something went wrong on the server. Please retry.",
            },
        )
    finally:
        slots.release()
        log.info(json.dumps({"id": rid, "status": status, "ms": timings}))


def main() -> FastAPI:  # uvicorn factory: `uvicorn rag.api.app:main --factory`
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return create_app(Settings())
```

Notes: `StreamingResponse` iterates a sync generator in a worker thread, so blocking DB and
model calls don't stall the event loop. If the client disconnects, Starlette stops iterating;
the `finally` still releases the slot.

- [ ] **Step 7: Run** `uv run pytest tests/test_api.py tests/test_generate.py -q` → PASS; `uv run ruff check --no-cache .` → clean.

- [ ] **Step 8: Local smoke test against the real index** (DB up, index loaded)

```bash
RAG_CORS_ORIGINS=http://localhost:3000 uv run uvicorn rag.api.app:main --factory --port 8000
curl -s localhost:8000/ready
curl -sN -X POST localhost:8000/api/ask -H 'content-type: application/json' \
  -d '{"question":"How do I roll back a deployment?"}' | head -c 1500
```

Expected: `{"chunks": 17730}`; `event: stage` blocks for dense/keyword/fusion/rerank, then `event: token`..., then `event: done` with sources.

- [ ] **Step 9: Commit** — `git commit -m "feat(api): FastAPI SSE service with rate, budget and concurrency guards"`

---

### Task 4: Container, index publishing, Railway config

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `railway.json`, `scripts/publish_index.sh`

**Interfaces:**
- Consumes: `rag.api.app:main` (Task 3).
- Produces: image serving on `$PORT`; `scripts/publish_index.sh TARGET_DATABASE_URL` copies local `documents` + `chunks` into the target.

- [ ] **Step 1: `Dockerfile`** (models baked in so cold starts don't download 150 MB)

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy FASTEMBED_CACHE_PATH=/models
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
RUN uv run python -c "from rag.config import Settings; from rag.embed import make_embedder; \
from rag.retrieval.rerank import make_reranker; s=Settings(_env_file=None); \
make_embedder(s).embed_query('x'); make_reranker(s).rerank('x', [])"
RUN useradd --create-home app && chown -R app /app /models
USER app
CMD ["sh", "-c", "uv run --no-sync uvicorn rag.api.app:main --factory --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
```

`--proxy-headers` makes `request.client.host` the tester's IP behind Railway's proxy, which the rate limiter needs.

- [ ] **Step 2: `.dockerignore`**

```
.venv
data
evals
web
docs
tests
.git
.env
**/__pycache__
```

- [ ] **Step 3: `railway.json`**

```json
{
  "$schema": "https://railway.com/railway.schema.json",
  "build": { "builder": "DOCKERFILE", "dockerfilePath": "Dockerfile" },
  "deploy": { "healthcheckPath": "/ready", "healthcheckTimeout": 120, "restartPolicyType": "ON_FAILURE" }
}
```

- [ ] **Step 4: `scripts/publish_index.sh`**

```bash
#!/usr/bin/env bash
# Copy the local index (documents + chunks, vectors included) into a remote Postgres.
# Usage: scripts/publish_index.sh "$RAILWAY_DATABASE_URL"
set -euo pipefail
TARGET="${1:?usage: publish_index.sh TARGET_DATABASE_URL}"
SOURCE="${RAG_DATABASE_URL:-postgresql://rag:rag@localhost:5433/rag}"
RAG_DATABASE_URL="$TARGET" uv run rag init-db
psql "$TARGET" -v ON_ERROR_STOP=1 -c "TRUNCATE chunks, documents RESTART IDENTITY CASCADE"
pg_dump "$SOURCE" --data-only --table=documents --table=chunks --no-owner \
  | psql "$TARGET" -v ON_ERROR_STOP=1 -q
psql "$TARGET" -c "SELECT (SELECT count(*) FROM documents) docs, (SELECT count(*) FROM chunks) chunks"
```

`chmod +x scripts/publish_index.sh`. Check `pg_dump`'s major version is ≥ 17 locally (`pg_dump --version`); if not, run it via `docker compose exec db pg_dump -U rag ...` instead.

- [ ] **Step 5: Verify the image locally**

```bash
docker build -t rag-api .
docker run --rm -p 8000:8000 --network host -e RAG_DATABASE_URL=postgresql://rag:rag@localhost:5433/rag \
  -e OPENROUTER_API_KEY="$(uv run python -c 'from rag.config import Settings;print(Settings().openrouter_api_key.get_secret_value())')" rag-api
curl -s localhost:8000/ready
```

(macOS: replace `--network host` and `localhost` in the URL with `host.docker.internal`.)
Expected: `{"chunks": 17730}`.

- [ ] **Step 6: Commit** — `git commit -m "build: Docker image with baked models, Railway config, index publish script"`

---

### Task 5: UI talks to the real API

**Files:**
- Create: `web/lib/api.ts`, `web/lib/api.test.ts`, `web/.env.example`
- Modify: `web/components/GlassBox.tsx`, `web/components/Panels.tsx`, `web/components/Overlays.tsx`, `web/components/store.ts`, `web/lib/types.ts`
- Delete: `web/lib/mockEngine.ts`, `web/lib/mockEngine.test.ts`, `web/lib/corpus.ts` (keep `outlineFrom`/`idOf` only if still imported; expected: not)

**Interfaces:**
- Consumes: HTTP contract of Task 3.
- Produces: `parseSSE(buffer: string): { events: {event: string; data: any}[]; rest: string }`; `getCollection(): Promise<CollectionInfo>`; `askStream(body, onEvent: (e) => void, signal?: AbortSignal): Promise<void>` throwing `ApiError {status, code, retryAfter?}` for non-200.

- [ ] **Step 1: Failing test `web/lib/api.test.ts`**

```ts
import assert from "node:assert/strict";
import { test } from "node:test";
import { parseSSE, toChunk } from "./api.ts";

test("parseSSE splits complete events and keeps the partial tail", () => {
  const { events, rest } = parseSSE('event: token\ndata: {"text":"Hi"}\n\nevent: done\ndata: {"te');
  assert.deepEqual(events, [{ event: "token", data: { text: "Hi" } }]);
  assert.equal(rest, 'event: done\ndata: {"te');
});

test("toChunk maps an API hit to the UI chunk shape", () => {
  const c = toChunk({ id: 7, heading: "Deploy > Roll back", title: "Deploy", url: "https://k8s/d", content: "abc", score: 0.5 });
  assert.deepEqual(c, { id: "7", page: "https://k8s/d", url: "https://k8s/d", path: ["Deploy", "Roll back"], text: "abc", chars: 3 });
});
```

- [ ] **Step 2: Run** `cd web && npm test` → FAIL (`Cannot find module ./api.ts`)

- [ ] **Step 3: Implement `web/lib/api.ts`**

```ts
import type { Chunk } from "./types.ts";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export type ApiHit = { id: number; heading: string; title: string; url: string; content: string; score: number };
export type CollectionInfo = { name: string; docs: number; chunks: number; llm_model: string; embedding_model: string };
export type AskBody = { question: string; k: number; rerank: boolean; max_one_per_page: boolean };
export type SSE = { event: string; data: any };

export class ApiError extends Error {
  constructor(public status: number, public code: string, public retryAfter?: number) { super(code); }
}

export function toChunk(h: ApiHit): Chunk {
  const path = h.heading ? h.heading.split(" > ") : [h.title];
  return { id: String(h.id), page: h.url, url: h.url, path, text: h.content, chars: h.content.length };
}

export function parseSSE(buffer: string): { events: SSE[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events = blocks.filter(Boolean).map((b) => {
    const f = Object.fromEntries(b.split("\n").map((l) => [l.slice(0, l.indexOf(": ")), l.slice(l.indexOf(": ") + 2)]));
    return { event: f.event, data: JSON.parse(f.data) };
  });
  return { events, rest };
}

export async function getCollection(): Promise<CollectionInfo> {
  const r = await fetch(`${API_URL}/api/collection`);
  if (!r.ok) throw new ApiError(r.status, "unavailable");
  return r.json();
}

export async function askStream(body: AskBody, onEvent: (e: SSE) => void, signal?: AbortSignal) {
  const r = await fetch(`${API_URL}/api/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
  if (!r.ok || !r.body) {
    const j = await r.json().catch(() => ({}));
    throw new ApiError(r.status, j.error || "unavailable", j.retry_after);
  }
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    const { events, rest } = parseSSE(buf + value);
    buf = rest;
    events.forEach(onEvent);
  }
}
```

`web/.env.example`: `NEXT_PUBLIC_API_URL=http://localhost:8000`

Run `npm test` → PASS.

- [ ] **Step 4: Rewire `GlassBox.tsx`** (keep the view-model; swap the data source)

1. On mount (inside the existing boot effect, replacing the hash-driven demo boot): call `getCollection()`. On success set `S.col = { id: "k8s", name: info.name, chunks: [], outline: [], total: info.chunks, docs: info.docs }`, `S.info = info`, mark the four ingest stages `done` with keys `${docs} docs`, `${chunks} chunks`, `"384-dim vectors"`, `"indexed in pgvector + full-text"` and no timings (`ms` undefined renders "—"), set `S.phase = "ready"`. On failure set `S.phase = "empty"`, `S.apiDown = true`.
2. Replace `ask()`'s body after the guard with: build `S.q` exactly as now but with an empty `R` (`dense: [], kw: [], fusion: [], rerank: [], ctx: [], best: 0, refused: false, qt: [], k`), then `await askStream({question: text, k: S.set.k, rerank: S.set.rerank, max_one_per_page: S.set.one}, onEvent)` inside try/catch.
3. `onEvent` mapping (all chunks pass through `toChunk`; keep a `byId` map on `S.q` so citations resolve to chunks):
   - `stage dense|keyword`: `R.dense|R.kw = hits.map(h => ({c: toChunk(h), s: h.score}))`, `setQ(name === "keyword" ? "keyword" : "dense", "done", ms)`; mark `fusion` running.
   - `stage fusion`: build `R.fusion` from hits in order with `rank = i+1`, `rrf = score`, `d`/`k` = 1-based position of the same id in `R.dense` / `R.kw` (or `null`); `setQ("fusion","done",ms)`; set `rerank` running if `S.set.rerank` else `skipped`.
   - `stage rerank`: `R.rerank = hits.map((h, i) => ({c, s: 1 / (1 + Math.exp(-h.score)), from: fusionRank(id) ?? i+1, to: i+1, tag: ""}))`; `setQ("rerank","done",ms)`.
   - first `token`: `R.ctx` = the first `k` chunks of the last stage list received (rerank if present, else fusion), tag them `ctx` in `R.rerank`/`R.fusion`; `setQ("generate","running")`; then append `data.text` to the answer message's current text part, `S.q.tok++`, `render()`.
   - `done`: replace `m.parts` by splitting `data.text` on `/\[(\d+)\]/` into text parts and `{t:"c", n, id: String(sources.find(s => s.n === n)?.id)}` parts (unknown n → keep the literal `[n]` text, it is an invalid citation); `m.cites = data.sources.map(s => ({n: s.n, id: String(s.id)}))`; `m.status = data.refused ? "refusal" : "done"`; `setQ("generate", data.refused ? "skipped" : "done")`, `setQ("cite", data.refused ? "skipped" : "done")`; `S.stats = {total: data.ms, retrieve: dense+keyword+fusion ms, rerank: rerank ms, gen: data.ms - those, tin: 0, tout: S.q.tok}`.
   - `error`: `m.status = "error"`, `m.code = data.code`, `m.message = data.message`, `setQ("generate","error")`.
   - catch `ApiError`: 429 → `m.status="error"`, `m.retryAt = Date.now() + retryAfter*1000`, message "Too many questions, retry in N s"; 503 busy → "Two answers are already running, retry in a few seconds"; other/network → "Service unavailable, try again later".
4. Delete `S.set.inject` and its settings row and palette entry; delete upload handling (`handleFile`, drag/drop listeners, file input, `U` shortcut, "Upload Markdown file" commands); delete the `#ingesting`/`#answer` hash boot.

- [ ] **Step 5: Copy changes in `Panels.tsx` / `Overlays.tsx`**
- Source panel: replace the drop zone with a static note: "Uploads are coming soon. This demo answers from the Kubernetes documentation." Keep the Kubernetes collection button (pressed). Hide the outline block (no outline data in v1).
- Ingest caption: "Kubernetes docs · indexed ahead of time"; hide "Show chunks" (`chunks` is empty).
- Rerank formula line: `cross-encoder relevance (sigmoid of the logit) · higher is more relevant`.
- Refusal note: "The model found no passage that answers this question, so it declined instead of guessing."
- Error box uses `m.message`; show Retry only when `retryAt` is past or absent.
- Footer `Model` and `Embedding` show `S.info.llm_model` / `S.info.embedding_model` (fallback "—").
- When `S.apiDown`: Ask panel empty state title "Service unavailable", text "The API is not reachable right now. Try again later."
- Empty-ask suggestions for k8s stay; drop the uploaded-file branch.

- [ ] **Step 6: Verify in the browser** (API from Task 3 running on :8000)

`cd web && npm run dev`, open http://localhost:3000: stages fill with real hits, tokens stream, citations open real passages, "Who won the 2018 World Cup?" is refused by the model. Stop the API and reload: "Service unavailable" shows. `npx tsc --noEmit && npm test && npm run build` → all pass.

- [ ] **Step 7: Commit** — `git commit -m "feat(web): drive the glass-box UI from the real API over SSE"`

---

### Task 6: Legal pages, attribution, security headers

**Files:**
- Create: `web/app/privacy/page.tsx`, `web/app/legal/page.tsx`, `web/next.config.ts`
- Modify: `web/components/Panels.tsx` (composer notice, footer links), `web/app/globals.css` (two small classes)

**Interfaces:** none consumed beyond Task 5. Produces routes `/privacy`, `/legal`.

- [ ] **Step 1: Ask the owner** for the publisher name and a contact email for the legal notice (French LCEN requires the site editor's identity and the host's details). Do not invent them.

- [ ] **Step 2: `web/app/privacy/page.tsx`** — plain server component, same tokens (`className="legal"`), sections:
  - *What is processed*: the question text, sent to the API and to the LLM provider (OpenRouter, free model `nvidia/nemotron-3-super-120b-a12b:free`) to produce the answer. Free model providers may log prompts. **Do not enter personal or confidential information.**
  - *What is stored*: nothing. Questions and answers are not saved; server logs record only request ids, timings and status codes, not question text or IP addresses. The client IP is held in memory for one minute for rate limiting, then discarded.
  - *Browser storage*: only your theme choice (`localStorage`, key `gb-theme`); no cookies, no analytics, no trackers.
  - *Processors*: Vercel (web hosting), Railway (API and database), OpenRouter and the model provider it routes to (answer generation); links to each privacy policy.
  - *Your rights*: since nothing is stored, there is nothing to access or delete; contact address from Step 1 for questions.
  - Last updated date.
- [ ] **Step 3: `web/app/legal/page.tsx`** — publisher (Step 1), hosting providers with addresses (Vercel Inc., 440 N Barranca Ave #4133, Covina, CA 91723, USA; Railway Corporation, San Francisco, CA, USA), and content credits: "Answers quote the Kubernetes documentation, © The Kubernetes Authors, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/kubernetes/website. Passages are cleaned and split into chunks for retrieval." Plus "This is a personal demo project, provided as is; answers may be wrong."
- [ ] **Step 4: UI links** — under the composer, `<p className="fine">Questions are sent to a third-party AI model. Don't include personal data. <a href="/privacy">Privacy</a></p>`; in the run-stats footer, right side: `Docs © The Kubernetes Authors · CC BY 4.0 · <a href="/legal">Legal</a> · <a href="/privacy">Privacy</a>`. CSS: `.fine{font-size:var(--fs-xs);color:var(--tx3)} .legal{max-width:720px;margin:0 auto;padding:48px 16px;line-height:1.6} .legal h1{font-family:var(--display);font-size:28px;margin-bottom:16px} .legal h2{font-size:var(--fs-l);margin:24px 0 8px}`.
- [ ] **Step 5: `web/next.config.ts`** security headers

```ts
import type { NextConfig } from "next";

const api = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const csp = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'", // inline theme script runs before paint
  "style-src 'self' 'unsafe-inline'",
  "font-src 'self'",
  "img-src 'self' data:",
  `connect-src 'self' ${api}`,
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join("; ");

const config: NextConfig = {
  poweredByHeader: false,
  async headers() {
    return [{
      source: "/(.*)",
      headers: [
        { key: "Content-Security-Policy", value: csp },
        { key: "X-Content-Type-Options", value: "nosniff" },
        { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        { key: "Strict-Transport-Security", value: "max-age=31536000; includeSubDomains" },
      ],
    }];
  },
};
export default config;
```

`next dev` uses React dev tooling that needs `'unsafe-eval'`; if the dev overlay breaks, add `'unsafe-eval'` to `script-src` only when `process.env.NODE_ENV !== "production"`.

- [ ] **Step 6: Verify** — `npm run build && npm start`, then `curl -sI localhost:3000 | grep -i content-security` shows the header; `/privacy` and `/legal` render in both themes; the browser console has no CSP violations while asking a question against the local API.
- [ ] **Step 7: Commit** — `git commit -m "feat(web): privacy and legal pages, CC BY attribution, security headers"`

---

### Task 7: CI and documentation

**Files:**
- Modify: `.github/workflows/ci.yml`, `README.md`, `CLAUDE.md`, `docs/07-serving.md`, `docs/README.md`, `docs/decisions.md`, `docs/glossary.md`, `.claude/plans/rag-pipeline.plan.md`

- [ ] **Step 1: CI web job** (add under `jobs:`)

```yaml
  web:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    defaults:
      run:
        working-directory: web
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm
          cache-dependency-path: web/package-lock.json
      - run: npm ci
      - run: npx tsc --noEmit
      - run: npm test
      - run: npm run build
```

(Node 22 runs `.ts` tests only with `--experimental-strip-types`; set the test script to `node --experimental-strip-types --test lib/*.test.ts` so both Node 22 and 26 work.)

- [ ] **Step 2: Docs**
  - `docs/07-serving.md`: replace the mock-engine sections with the real flow: event contract table (stage/token/done/error), guards (why each limit exists, numbers), deployment diagram (Vercel → Railway API → Railway Postgres; OpenRouter), how the index is published (`pg_dump`, why not re-embed), what "conform" covers (privacy, attribution, headers) and what it doesn't (no auth, single replica).
  - `docs/decisions.md`: entry "Production v1 on Vercel + Railway, k8s only" (context: owner wants testers; decision: guards + no storage + free model; consequence: single replica in-memory guards, ~$5/month Railway).
  - `docs/glossary.md`: SSE, CORS, CSP, rate limiting, readiness probe.
  - `README.md`: Status line (Phase 5 live demo), "Deploy" section with the owner's manual steps below, Configuration table rows for the four new settings and `NEXT_PUBLIC_API_URL`, Layout (`src/rag/api/`, `Dockerfile`, `railway.json`), Roadmap.
  - `CLAUDE.md`: commands (`uvicorn rag.api.app:main --factory`, `scripts/publish_index.sh`), layout (`api/`), gotcha (in-memory guards assume one replica).
  - Plan file: tick Phase 5 boxes delivered.
- [ ] **Step 3: Owner's deploy runbook** (goes in README "Deploy"; the owner runs it, Claude never handles the keys):
  1. Railway: new project → add **Postgres** using the `pgvector/pgvector:pg17` image (Railway template "pgvector"); copy its public `DATABASE_URL`.
  2. Locally: `scripts/publish_index.sh "<public DATABASE_URL>"` → prints 1,652 docs / 17,730 chunks.
  3. Railway: add service from this GitHub repo (uses `railway.json`). Variables: `RAG_DATABASE_URL=${{Postgres.DATABASE_URL}}` (private URL), `OPENROUTER_API_KEY`, `RAG_CORS_ORIGINS=https://<vercel-domain>`, `RAG_EMBEDDING_THREADS=2`. Give it 2 GB RAM. Generate a public domain.
  4. Vercel: import the repo, root directory `web`, env `NEXT_PUBLIC_API_URL=https://<railway-api-domain>`. Deploy.
  5. Check: `curl https://<api>/ready` → chunk count; open the Vercel URL, ask a question; ask 11 questions in a minute → the 11th is rate-limited.
- [ ] **Step 4: Full verification** — `uv run ruff format --check . && uv run ruff check --no-cache . && uv run pytest -m "not integration" --cov=rag -q && (cd web && npx tsc --noEmit && npm test && npm run build)`; all green, coverage ≥ 80%.
- [ ] **Step 5: Commit** — `git commit -m "docs, ci: production v1 runbook, serving chapter, web CI job"`
