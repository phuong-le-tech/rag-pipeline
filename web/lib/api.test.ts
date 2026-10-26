import assert from "node:assert/strict";
import { test } from "node:test";
import { ApiError, askStream, effectiveAgent, effectiveImpl, ingestUpdate, parseSSE, splitCitations, stepLabel, toChunk, uploadStream } from "./api.ts";

test("parseSSE splits complete events and keeps the partial tail", () => {
  const { events, rest } = parseSSE('event: token\ndata: {"text":"Hi"}\n\nevent: done\ndata: {"te');
  assert.deepEqual(events, [{ event: "token", data: { text: "Hi" } }]);
  assert.equal(rest, 'event: done\ndata: {"te');
});

test("toChunk maps an API hit to the UI chunk shape", () => {
  const c = toChunk({ id: 7, heading: "Deploy > Roll back", title: "Deploy", url: "https://k8s/d", content: "abc", score: 0.5 });
  assert.deepEqual(c, { id: "7", page: "https://k8s/d", url: "https://k8s/d", path: ["Deploy", "Roll back"], text: "abc", chars: 3 });
});

test("splitCitations turns known [n] into citation parts and keeps unknown ones as text", () => {
  const parts = splitCitations("Undo it [1]. See [9].", [{ n: 1, id: 7 }]);
  assert.deepEqual(parts, [
    { t: "t", v: "Undo it " },
    { t: "c", n: 1, id: "7" },
    { t: "t", v: ". See [9]." },
  ]);
});

const sseResponse = (body: string) =>
  new Response(new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode(body)); c.close(); } }), { status: 200 });

test("ingestUpdate maps the five server stages onto the four cards", () => {
  const u = (name: string, data: object) => ingestUpdate({ event: "stage", data: { name, ms: 5, ...data } });
  assert.deepEqual(u("load", { chars: 1200 }), { stage: "parse", done: false, ms: 5, key: "1,200 chars" });
  assert.deepEqual(u("clean", { title: "Notes" }), { stage: "parse", done: true, ms: 5, key: "Notes" });
  assert.deepEqual(u("chunk", { chunks: 4 }), { stage: "chunk", done: true, ms: 5, key: "4 chunks" });
  assert.deepEqual(u("embed", { dim: 384 }), { stage: "embed", done: true, ms: 5, key: "384-dim vectors" });
  assert.equal(u("store", {})?.stage, "index");
  assert.equal(ingestUpdate({ event: "done", data: {} }), null);
});

test("uploadStream sends a file as multipart and pasted text as JSON, and relays events", async (t) => {
  const calls: RequestInit[] = [];
  t.mock.method(globalThis, "fetch", async (_url: string, init: RequestInit) => {
    calls.push(init);
    return sseResponse('event: done\ndata: {"collection":"abc","docs":1,"chunks":2}\n\n');
  });
  const seen: string[] = [];
  await uploadStream(new File(["# Hi"], "hi.md"), (e) => seen.push(e.event));
  await uploadStream("pasted", (e) => seen.push(e.event));
  assert.ok(calls[0].body instanceof FormData);
  assert.equal(calls[1].body, '{"text":"pasted"}');
  assert.deepEqual(seen, ["done", "done"]);
});

test("uploadStream raises ApiError with the server's code on 413", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response('{"error":"too_large"}', { status: 413 }));
  await assert.rejects(uploadStream("x", () => {}), (e: ApiError) => e.status === 413 && e.code === "too_large");
});

test("askStream passes the collection through", async (t) => {
  let sent = "";
  t.mock.method(globalThis, "fetch", async (_url: string, init: RequestInit) => {
    sent = String(init.body);
    return sseResponse("");
  });
  await askStream({ question: "q", k: 5, rerank: true, max_one_per_page: true, collection: "a".repeat(32) }, () => {});
  assert.equal(JSON.parse(sent).collection, "a".repeat(32));
});

const info = (impls?: ("handwritten" | "langchain")[]) => ({ name: "k", docs: 1, chunks: 1, llm_model: "m", embedding_model: "e", impls });

test("effectiveImpl keeps the choice only when the server offers it", () => {
  assert.equal(effectiveImpl(info(["handwritten", "langchain"]), "langchain"), "langchain");
  assert.equal(effectiveImpl(info(["handwritten"]), "langchain"), "handwritten");
  assert.equal(effectiveImpl(info(), "langchain"), "handwritten"); // older server without `impls`
  assert.equal(effectiveImpl(null, "langchain"), "handwritten"); // collection not loaded yet
});

test("effectiveAgent needs both the setting and a server that offers it", () => {
  const withAgent = { ...info(), agent: true };
  assert.equal(effectiveAgent(withAgent, true), true);
  assert.equal(effectiveAgent(withAgent, false), false);
  assert.equal(effectiveAgent(info(), true), false); // server without agent mode
  assert.equal(effectiveAgent(null, true), false);
});

test("stepLabel describes a search, a repeat and a failure", () => {
  const base = { step: 1, query: "rollback", k: 5, count: 3, duplicate: false, error: null };
  assert.equal(stepLabel(base), "rollback · 3 passages");
  assert.equal(stepLabel({ ...base, count: 1 }), "rollback · 1 passage");
  assert.equal(stepLabel({ ...base, duplicate: true }), "rollback · already searched");
  assert.equal(stepLabel({ ...base, query: "", error: "Error: query is required." }), "Search failed: Error: query is required.");
});
