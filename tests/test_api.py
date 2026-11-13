import dataclasses
import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from rag.api.app import Services, create_app
from rag.api.guards import DailyBudget, RateLimiter
from rag.config import Settings
from rag.generate.generator import FakeGenerator
from rag.retrieval.search import Hit


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
            if on_stage:
                on_stage(name, [HIT], 1.0)
        return [HIT]


def make_client(gen=None, lc_generator=None, tool_generator=None, lc_agent_model=None, **overrides):
    @contextmanager
    def factory():
        yield StagedFake()

    settings = Settings(_env_file=None, **overrides)
    services = Services(
        factory,
        gen or FakeGenerator("Use rollout undo [1]."),
        lambda: {"docs": 1, "chunks": 1},
        lc_generator=lc_generator,
        tool_generator=tool_generator,
        lc_agent_model=lc_agent_model,
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
    assert names[:5] == ["stage"] * 4 + ["context"] and "token" in names and names[-1] == "done"
    assert ev[4][1]["hits"][0]["id"] == 7
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
    assert ev[-1][0] == "error" and ev[-1][1]["code"] == "daily_limit"
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


def test_slot_released_after_each_ask():
    c = make_client(max_concurrent_asks=1)
    for _ in range(3):
        assert c.post("/api/ask", json={"question": "a"}).status_code == 200


def test_health_and_collection():
    c = make_client(openrouter_model="x/y:free")
    assert c.get("/health").json() == {"status": "ok"}
    body = c.get("/api/collection").json()
    assert body["chunks"] == 1 and body["llm_model"] == "x/y:free"


def test_cors_allows_only_configured_origin():
    c = make_client(cors_origins="https://demo.example")
    ok = c.options(
        "/api/ask",
        headers={"Origin": "https://demo.example", "Access-Control-Request-Method": "POST"},
    )
    bad = c.options(
        "/api/ask",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "https://demo.example"
    assert "access-control-allow-origin" not in bad.headers


def test_slot_released_when_client_disconnects_before_stream_starts():
    import asyncio
    import gc

    client = make_client(max_concurrent_asks=1)
    app = client.app
    body = json.dumps({"question": "a"}).encode()
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        pass

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/ask",
        "raw_path": b"/api/ask",
        "query_string": b"",
        "root_path": "",
        "client": ("1.2.3.4", 1),
        "server": ("t", 80),
        "headers": [(b"content-type", b"application/json"), (b"host", b"t")],
    }
    asyncio.run(app(scope, receive, send))
    gc.collect()
    assert app.state.slots.acquire(blocking=False), "slot leaked after early disconnect"


def test_rate_limit_ignores_client_forged_forwarded_for():
    c = make_client(rate_limit_per_minute=1)
    # The edge proxy appends the real client IP last; earlier entries are client-controlled.
    first = c.post(
        "/api/ask", json={"question": "a"}, headers={"X-Forwarded-For": "10.0.0.1, 203.0.113.9"}
    )
    forged = c.post(
        "/api/ask", json={"question": "a"}, headers={"X-Forwarded-For": "10.0.0.2, 203.0.113.9"}
    )
    other = c.post("/api/ask", json={"question": "a"}, headers={"X-Forwarded-For": "203.0.113.10"})
    assert first.status_code == 200 and forged.status_code == 429
    assert other.status_code == 200  # a different real client has its own budget


def test_rate_limiter_forgets_idle_clients():
    t = [0.0]
    rl = RateLimiter(per_minute=1, clock=lambda: t[0])
    rl.check("a")
    t[0] = 61.0
    rl.check("b")
    assert "a" not in rl.calls


# --- uploads ---------------------------------------------------------------------------------

CID = "a" * 32


def fake_upload(collection, filename, text):
    yield "stage", {"name": "load", "ms": 0}
    yield "stage", {"name": "store", "ms": 1}
    yield "done", {"collection": collection, "docs": 1, "chunks": 2}


def upload_client(**overrides):
    @contextmanager
    def factory():
        yield StagedFake()

    settings = Settings(_env_file=None, **overrides)
    services = Services(
        factory,
        FakeGenerator("x"),
        lambda c="k8s": {"docs": {"k8s": 5, CID: 1}.get(c, 0), "chunks": 2},
        fake_upload,
    )
    return TestClient(create_app(settings, services))


def test_upload_file_streams_stages_then_done():
    resp = upload_client().post(
        "/api/collections", files={"file": ("notes.md", b"# Notes\n\nhello", "text/markdown")}
    )

    ev = events(resp)
    assert [e for e, _ in ev] == ["stage", "stage", "done"]
    assert len(ev[-1][1]["collection"]) == 32 and ev[-1][1]["chunks"] == 2


def test_upload_json_text():
    resp = upload_client().post("/api/collections", json={"text": "# Hi\n\nthere"})
    assert events(resp)[-1][0] == "done"


def test_upload_rejects_oversize_with_413():
    resp = upload_client().post("/api/collections", json={"text": "x" * 1_100_000})
    assert resp.status_code == 413


def test_upload_rejects_non_utf8_with_422():
    resp = upload_client().post(
        "/api/collections", files={"file": ("a.md", b"\xff\xfe\x00bad", "text/markdown")}
    )
    assert resp.status_code == 422 and resp.json()["error"] == "not_utf8"


def test_upload_rejects_unknown_content_type():
    resp = upload_client().post(
        "/api/collections", content=b"hi", headers={"content-type": "text/plain"}
    )
    assert resp.status_code == 422


def test_upload_is_rate_limited_and_uses_a_slot():
    c = upload_client(upload_rate_limit_per_minute=1)
    assert c.post("/api/collections", json={"text": "a"}).status_code == 200
    assert c.post("/api/collections", json={"text": "a"}).status_code == 429
    busy = upload_client(max_concurrent_asks=1)
    busy.app.state.slots.acquire()
    assert busy.post("/api/collections", json={"text": "a"}).status_code == 503


def test_upload_limit_error_is_reported_in_band():
    def too_big(collection, filename, text):
        raise ValueError("upload has 301 chunks; the limit is 300")
        yield  # pragma: no cover

    c = upload_client()
    c.app.state.services = Services(
        c.app.state.services.retriever_factory, FakeGenerator("x"), lambda: {}, too_big
    )
    ev = events(c.post("/api/collections", json={"text": "a"}))
    assert ev == [
        ("error", {"code": "rejected", "message": "upload has 301 chunks; the limit is 300"})
    ]


def test_ask_passes_collection_and_rejects_bad_ids():
    seen = []

    class Spy(StagedFake):
        def retrieve(self, query, config, on_stage=None):
            seen.append((config.collection, config.max_per_source))
            return super().retrieve(query, config, on_stage)

    @contextmanager
    def factory():
        yield Spy()

    services = Services(factory, FakeGenerator("Use [1]."), lambda: {"docs": 1, "chunks": 1})
    c = TestClient(create_app(Settings(_env_file=None), services))

    c.post("/api/ask", json={"question": "q", "collection": CID})
    c.post("/api/ask", json={"question": "q"})
    bad = c.post("/api/ask", json={"question": "q", "collection": "../k8s"})

    # Uploads skip the one-chunk-per-file cap; the Kubernetes docs keep it.
    assert seen == [(CID, None), ("k8s", 1)] and bad.status_code == 422


def test_collection_stats_by_id():
    c = upload_client()
    assert c.get(f"/api/collection?id={CID}").json()["docs"] == 1
    assert c.get(f"/api/collection?id={'b' * 32}").status_code == 404
    assert c.get("/api/collection?id=nope").status_code == 422
    assert c.get("/api/collection").json()["name"] == "Kubernetes docs"


def lc_client(reply="Use rollout undo [1].", **overrides):
    from langchain_core.language_models import FakeListChatModel

    from rag.lc.chain import LangChainGenerator

    generator = LangChainGenerator(FakeListChatModel(responses=[reply]))
    return make_client(lc_generator=lambda: generator, **overrides)


def test_ask_defaults_to_handwritten():
    ev = events(make_client().post("/api/ask", json={"question": "roll back?"}))
    assert ev[-1][1]["impl"] == "handwritten"


def test_ask_impl_langchain_streams_the_same_events():
    hand = events(make_client().post("/api/ask", json={"question": "roll back?"}))
    lc = events(lc_client().post("/api/ask", json={"question": "roll back?", "impl": "langchain"}))
    assert [e for e, _ in lc if e != "token"] == [e for e, _ in hand if e != "token"]
    assert "token" in [e for e, _ in lc]
    done = lc[-1][1]
    assert done["impl"] == "langchain" and done["sources"] == [{"n": 1, "id": 7}]
    assert done["text"] == hand[-1][1]["text"]


def test_unknown_impl_is_422():
    r = make_client().post("/api/ask", json={"question": "a", "impl": "magic"})
    assert r.status_code == 422


def test_langchain_unavailable_is_a_clear_422_and_takes_no_slot():
    c = make_client(max_concurrent_asks=1)
    r = c.post("/api/ask", json={"question": "a", "impl": "langchain"})
    assert r.status_code == 422 and r.json() == {"error": "impl_unavailable"}
    assert c.post("/api/ask", json={"question": "a"}).status_code == 200  # slot still free


def test_langchain_failure_ends_with_error_event():
    from langchain_core.language_models import FakeListChatModel

    from rag.lc.chain import LangChainGenerator

    class Boom(LangChainGenerator):
        def stream(self, system, user):
            yield "partial "
            raise TimeoutError("upstream timeout")

    boom = Boom(FakeListChatModel(responses=["x"]))
    c = make_client(lc_generator=lambda: boom)
    ev = events(c.post("/api/ask", json={"question": "a", "impl": "langchain"}))
    assert ev[-1][0] == "error" and ev[-1][1]["code"] == "llm_failed"


def test_langchain_answers_are_charged_to_the_same_daily_budget():
    c = lc_client(daily_llm_limit=1)
    body = {"question": "a", "impl": "langchain"}
    assert events(c.post("/api/ask", json=body))[-1][0] == "done"
    assert events(c.post("/api/ask", json=body))[-1][1]["code"] == "daily_limit"


def test_collection_lists_the_available_implementations():
    assert make_client().get("/api/collection").json()["impls"] == ["handwritten"]
    assert lc_client().get("/api/collection").json()["impls"] == ["handwritten", "langchain"]


def test_failing_generator_build_does_not_spend_the_daily_budget():
    from langchain_core.language_models import FakeListChatModel

    from rag.lc.chain import LangChainGenerator

    attempts = []

    def build():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("no model configured")
        return LangChainGenerator(FakeListChatModel(responses=["Use rollout undo [1]."]))

    c = make_client(lc_generator=build, daily_llm_limit=1)
    body = {"question": "a", "impl": "langchain"}
    assert events(c.post("/api/ask", json=body))[-1][1]["code"] == "failed"
    assert events(c.post("/api/ask", json=body))[-1][0] == "done"  # the slot was not spent


@pytest.mark.parametrize("impl", ["handwritten", "langchain"])
def test_ask_log_line_names_the_implementation(impl, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="rag.api")
    client = lc_client()
    events(client.post("/api/ask", json={"question": "roll back?", "impl": impl}))
    logged = [json.loads(r.message) for r in caplog.records if r.name == "rag.api"]
    assert logged[-1]["impl"] == impl and logged[-1]["status"] == "ok"


# ---- agent mode ----


def scripted_agent(*extra_turns):
    from rag.generate.generator import ScriptedToolGenerator, ToolCall, Turn

    turns = [
        Turn("", (ToolCall("c1", "search_docs", '{"query": "rollback"}'),)),
        Turn("Use rollout undo [1]."),
        *extra_turns,
    ]
    return lambda: ScriptedToolGenerator(turns)


def lc_scripted_agent():
    from langchain_core.messages import AIMessage

    from rag.lc.llm import fake_tool_chat_model

    call = {"name": "search_docs", "args": {"query": "rollback"}, "id": "c1"}
    messages = [
        AIMessage(content="", tool_calls=[call]),
        AIMessage(content="Use rollout undo [1]."),
    ]
    return lambda: fake_tool_chat_model(messages)


def ask_agent(client, **body):
    return client.post("/api/ask", json={"question": "roll back?", "agent": True, **body})


def test_agent_streams_stages_steps_context_and_done():
    ev = events(ask_agent(make_client(tool_generator=scripted_agent())))
    names = [e for e, _ in ev]
    assert names == ["stage"] * 4 + ["step", "context", "done"]
    step = dict(ev)["step"]
    assert (step["step"], step["query"], step["numbers"], step["duplicate"]) == (
        1,
        "rollback",
        [1],
        False,
    )
    done = ev[-1][1]
    assert done["text"] == "Use rollout undo [1]." and done["sources"] == [{"n": 1, "id": 7}]
    assert (done["agent"], done["impl"], done["searches"], done["llm_calls"]) == (
        True,
        "handwritten",
        1,
        2,
    )
    assert dict(ev)["context"]["hits"][0]["id"] == 7


def test_agent_with_langchain_streams_the_same_events():
    client = make_client(
        lc_generator=lambda: None,
        lc_agent_model=lc_scripted_agent(),
        tool_generator=scripted_agent(),
    )
    ev = events(ask_agent(client, impl="langchain"))
    assert [e for e, _ in ev] == ["stage"] * 4 + ["step", "context", "done"]
    assert ev[-1][1]["impl"] == "langchain" and ev[-1][1]["searches"] == 1


def test_agent_unavailable_is_a_clear_422_and_takes_no_slot():
    c = make_client(max_concurrent_asks=1)
    r = ask_agent(c)
    assert r.status_code == 422 and r.json() == {"error": "agent_unavailable"}
    assert c.post("/api/ask", json={"question": "a"}).status_code == 200


def test_agent_charges_the_daily_budget_per_model_call():
    c = make_client(tool_generator=scripted_agent(), daily_llm_limit=2)
    assert events(ask_agent(c))[-1][0] == "done"  # two model calls used the whole budget
    assert events(ask_agent(c))[-1][1]["code"] == "daily_limit"


def test_agent_budget_running_out_mid_run_keeps_the_search_steps():
    c = make_client(tool_generator=scripted_agent(), daily_llm_limit=1)
    ev = events(ask_agent(c))
    names = [e for e, _ in ev]
    assert "step" in names and ev[-1][0] == "error" and ev[-1][1]["code"] == "daily_limit"


def test_agent_llm_failure_ends_with_llm_failed():
    import httpx
    import openai

    class Boom:
        def chat(self, system, messages, tools, allow_tools=True):
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://x"))

    ev = events(ask_agent(make_client(tool_generator=lambda: Boom())))
    assert ev[-1][0] == "error" and ev[-1][1]["code"] == "llm_failed"


def test_agent_server_failure_ends_with_failed():
    class Boom:
        def chat(self, system, messages, tools, allow_tools=True):
            raise RuntimeError("bug")

    ev = events(ask_agent(make_client(tool_generator=lambda: Boom())))
    assert ev[-1][0] == "error" and ev[-1][1]["code"] == "failed"


def test_collection_reports_whether_agent_mode_is_available():
    assert make_client().get("/api/collection").json()["agent"] is False
    assert make_client(tool_generator=scripted_agent()).get("/api/collection").json()["agent"]


def test_agent_log_line_has_agent_and_counts(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="rag.api")
    events(ask_agent(make_client(tool_generator=scripted_agent())))
    logged = [json.loads(r.message) for r in caplog.records if r.name == "rag.api"][-1]
    assert logged["agent"] is True and logged["llm_calls"] == 2 and logged["status"] == "ok"


def test_agent_mode_is_on_by_default_in_settings():
    assert Settings(_env_file=None).agent_enabled is True


def _pdf(text: str | None) -> bytes:
    import io

    import pypdf
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    w = pypdf.PdfWriter()
    page = w.add_blank_page(200, 200)
    if text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = w._add_object(stream)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_upload_pdf_is_extracted_to_text():
    seen = {}

    def capture(collection, filename, text):
        seen["text"] = text
        yield "done", {"collection": collection, "docs": 1, "chunks": 1}

    c = upload_client()
    c.app.state.services = dataclasses.replace(c.app.state.services, index_upload=capture)
    resp = c.post(
        "/api/collections", files={"file": ("a.pdf", _pdf("Hello PDF"), "application/pdf")}
    )
    assert events(resp)[-1][0] == "done" and "Hello PDF" in seen["text"]


def test_upload_rejects_broken_pdf_with_422():
    resp = upload_client().post(
        "/api/collections", files={"file": ("a.pdf", b"%PDF-1.7 garbage", "application/pdf")}
    )
    assert resp.status_code == 422 and resp.json()["error"] == "bad_upload"


def test_upload_adds_to_an_existing_collection():
    c = upload_client()
    resp = c.post(
        "/api/collections",
        files={"file": ("b.md", b"# B\n\nbee", "text/markdown")},
        data={"collection": CID},
    )
    assert events(resp)[-1][1]["collection"] == CID
    pasted = c.post("/api/collections", json={"text": "more", "collection": CID})
    assert events(pasted)[-1][1]["collection"] == CID


@pytest.mark.parametrize(("cid", "status"), [("nope", 422), ("b" * 32, 404), ("k8s", 422)])
def test_upload_into_bad_or_unknown_collection_is_refused(cid, status):
    resp = upload_client().post("/api/collections", json={"text": "hi", "collection": cid})
    assert resp.status_code == status


def test_uploaded_collection_gets_topic_neutral_prompt_and_k8s_keeps_its_own():
    from rag.generate.prompt import GENERIC_SYSTEM_PROMPT, SYSTEM_PROMPT

    gen = FakeGenerator("ok [1]")

    def ask(collection):
        c = upload_client()
        c.app.state.services = dataclasses.replace(c.app.state.services, generator=gen)
        body = {"question": "what?", **({"collection": collection} if collection else {})}
        events(c.post("/api/ask", json=body))

    ask(None)
    ask(CID)
    assert [system for system, _ in gen.calls] == [SYSTEM_PROMPT, GENERIC_SYSTEM_PROMPT]
    assert "Kubernetes" not in GENERIC_SYSTEM_PROMPT


def test_pdf_is_not_parsed_when_no_slot_is_free(monkeypatch):
    import rag.api.app as api

    def boom(raw):
        raise AssertionError("parsed without a slot")

    monkeypatch.setattr(api, "pdf_to_text", boom)
    c = upload_client(max_concurrent_asks=1)
    c.app.state.slots.acquire()  # the only slot is busy
    resp = c.post("/api/collections", files={"file": ("a.pdf", _pdf("x"), "application/pdf")})
    assert resp.status_code == 503


@pytest.mark.parametrize(
    "file",
    [("a.pdf", b"%PDF-1.7 garbage"), ("a.md", b"\xff\xfe bad")],  # 422 after taking the slot
)
def test_rejected_upload_gives_its_slot_back(file):
    c = upload_client(max_concurrent_asks=1)
    assert c.post("/api/collections", files={"file": file}).status_code == 422
    unknown = {"text": "hi", "collection": "b" * 32}
    assert c.post("/api/collections", json=unknown).status_code == 404
    assert events(c.post("/api/collections", json={"text": "# ok"}))[-1][0] == "done"


def test_pdf_text_over_the_cap_is_refused(monkeypatch):
    from rag.ingest import loaders

    monkeypatch.setattr(loaders, "MAX_PDF_TEXT_CHARS", 5)
    with pytest.raises(ValueError, match="PDF text is over"):
        loaders.pdf_to_text(_pdf("more than five characters"))


def test_uploads_and_questions_have_separate_rate_limits():
    c = upload_client(rate_limit_per_minute=1, upload_rate_limit_per_minute=3)
    for _ in range(3):  # a batch of files leaves the question allowance untouched
        assert c.post("/api/collections", json={"text": "a"}).status_code == 200
    assert c.post("/api/collections", json={"text": "a"}).status_code == 429
    assert c.post("/api/ask", json={"question": "what?"}).status_code == 200
    assert c.post("/api/ask", json={"question": "what?"}).status_code == 429


def test_api_sends_security_headers_on_json_and_streams():
    c = upload_client()
    for resp in (c.get("/health"), c.post("/api/collections", json={"text": "# a"})):
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["x-frame-options"] == "DENY"


def test_failed_upload_logs_no_upload_data(caplog):
    def fail(collection, filename, text):
        raise RuntimeError(f"duplicate key upload:{collection}/{filename}")
        yield  # pragma: no cover

    c = upload_client()
    c.app.state.services = dataclasses.replace(c.app.state.services, index_upload=fail)
    files = {"file": ("secret-plan.md", b"# x", "text/markdown")}
    with caplog.at_level("INFO", logger="rag.api"):
        assert events(c.post("/api/collections", files=files))[-1][0] == "error"
    assert "RuntimeError" in caplog.text and "secret-plan" not in caplog.text
