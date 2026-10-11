import assert from "node:assert/strict";
import { test } from "node:test";
import { K8S_COL } from "./corpus.ts";
import { compose, parseMd, retrieve } from "./mockEngine.ts";

const SET = { k: 5, rerank: true, one: false, inject: false };

test("parseMd splits on headings and keeps the heading path", () => {
  const { chunks, heads } = parseMd("a.md", "# Top\nintro\n\n## Sub\nbody text\n```\n# not a heading\n```");
  assert.equal(heads, 2);
  assert.deepEqual(chunks.map((c) => c.path), [["Top"], ["Top", "Sub"]]);
  assert.match(chunks[1].text, /# not a heading/);
});

test("in-scope question retrieves the rollback passage and cites it", () => {
  const r = retrieve(K8S_COL, SET, "How do I roll back a failed Deployment?");
  assert.equal(r.refused, false);
  assert.ok(r.ctx.some((c) => c.path.includes("Rolling back a Deployment")));
  const { cites } = compose(r.qt, r.ctx);
  assert.ok(cites.length >= 1);
});

test("out-of-scope question is refused", () => {
  assert.equal(retrieve(K8S_COL, SET, "Who won the 2018 World Cup?").refused, true);
});
