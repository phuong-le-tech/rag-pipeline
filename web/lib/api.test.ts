import assert from "node:assert/strict";
import { test } from "node:test";
import { parseSSE, splitCitations, toChunk } from "./api.ts";

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
