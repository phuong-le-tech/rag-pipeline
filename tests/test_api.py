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
    c = make_client()
    assert c.get("/health").json() == {"status": "ok"}
    body = c.get("/api/collection").json()
    assert body["chunks"] == 1 and body["llm_model"].endswith(":free")


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
