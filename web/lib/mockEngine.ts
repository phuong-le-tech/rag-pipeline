// In-browser stand-in for the Python pipeline (BM25, hashed 384-dim vectors, RRF, heuristic
// rerank, extractive answers). shortcut: no real models, swap for the FastAPI SSE client.
import { idOf } from "./corpus.ts";
import type { Chunk, Cite, Collection, Fused, Part, Reranked, Retrieval, Settings } from "./types.ts";

export const THRESH = 0.35;
const rnd = (a: number, b: number) => Math.round(a + Math.random() * (b - a));

/* ---------- markdown parse + chunk (real, used for uploads) ---------- */
export function parseMd(name: string, text: string): { chunks: Chunk[]; heads: number } {
  const lines = text.replace(/\r/g, "").split("\n");
  let stack: string[] = [];
  let cur: { path: string[]; buf: string[] } = { path: [], buf: [] };
  let inCode = false;
  let heads = 0;
  const out: { path: string[]; text: string }[] = [];
  const flush = () => {
    const t = cur.buf.join("\n").trim();
    if (!t) return;
    let acc = "";
    const emit = (s: string) => out.push({ path: cur.path.length ? cur.path.slice() : [name], text: s });
    t.split(/\n{2,}/).forEach((p) => {
      if (acc && (acc + p).length > 700) {
        emit(acc);
        acc = "";
      }
      acc += (acc ? "\n\n" : "") + p;
    });
    if (acc) emit(acc);
  };
  for (const ln of lines) {
    if (/^```/.test(ln)) inCode = !inCode;
    const m = !inCode && /^(#{1,6})\s+(.*)$/.exec(ln);
    if (m) {
      flush();
      heads++;
      const d = m[1].length;
      stack = stack.slice(0, d - 1);
      stack[d - 1] = m[2].trim();
      cur = { path: stack.filter(Boolean), buf: [] };
    } else cur.buf.push(ln);
  }
  flush();
  const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const chunks = out.map((c, i) => ({ id: idOf(i), page: name, url: name + "#" + slug(c.path[c.path.length - 1]), path: c.path, text: c.text, chars: c.text.length }));
  return { chunks, heads };
}

/* ---------- tokens, synonyms, vectors ---------- */
const STOP = new Set("a an the and or of to in on for with is are was were be been has have it its this that how do does did i my me we you your what why when which who can could should would between difference vs use using back about kubernetes k8s kubectl".split(" "));
const stem = (w: string) => (w.length < 4 ? w : w.replace(/(ing|ed|ly)$/, "").replace(/([^s])s$/, "$1").replace(/e$/, ""));
export const toks = (t: string) => (t.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => w.length > 1 && !STOP.has(w)).map(stem);
const SYNG = new Map<string, Set<string>>();
[["roll", "undo", "revert", "rollback", "revision", "previous"], ["fail", "crash", "stuck", "broken", "unstable", "deadline"], ["probe", "health", "healthy", "liveness", "readiness", "restart", "kubelet"], ["memory", "oom", "oomkilled", "limit", "kill", "terminate"], ["scale", "replica", "autoscale", "scaling", "hpa"], ["secret", "password", "token", "credential", "sensitive", "key"], ["expose", "reach", "access", "port", "network", "service", "traffic"], ["storage", "volume", "disk", "persist"], ["config", "configuration", "configmap", "environment", "env"], ["shutdown", "sigterm", "grace", "delete", "termination"]].forEach((g) => {
  const s = g.map(stem);
  s.forEach((t) => SYNG.set(t, new Set([...(SYNG.get(t) || []), ...s])));
});
const hash = (s: string) => {
  let x = 2166136261;
  for (let i = 0; i < s.length; i++) {
    x ^= s.charCodeAt(i);
    x = Math.imul(x, 16777619);
  }
  return x >>> 0;
};
function vec(ts: string[], ws?: number[]): Float32Array {
  const v = new Float32Array(384);
  ts.forEach((t, i) => {
    const w = ws ? ws[i] : 1;
    const s = "^" + t + "$";
    v[hash(t) % 384] += 2 * w;
    for (let j = 0; j + 3 <= s.length; j++) v[hash(s.slice(j, j + 3)) % 384] += w;
  });
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

/* ---------- per-collection index ---------- */
type Feat = { tf: Map<string, number>; len: number; set: Set<string>; ptoks: Set<string>; vec: Float32Array };
type Index = { N: number; df: Map<string, number>; avg: number; f: Map<string, Feat> };
const indexes = new WeakMap<Collection, Index>();
function ensureIdx(col: Collection): Index {
  const hit = indexes.get(col);
  if (hit) return hit;
  const docs = col.chunks.map((c) => toks(c.path.join(" ") + " " + c.path.join(" ") + " " + c.text));
  const df = new Map<string, number>();
  docs.forEach((d) => new Set(d).forEach((t) => df.set(t, (df.get(t) || 0) + 1)));
  const f = new Map<string, Feat>();
  col.chunks.forEach((c, i) => {
    const tf = new Map<string, number>();
    docs[i].forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
    f.set(c.id, { tf, len: docs[i].length, set: new Set(docs[i]), ptoks: new Set(toks(c.path.join(" "))), vec: vec(docs[i]) });
  });
  const ix = { N: docs.length, df, avg: docs.reduce((a, d) => a + d.length, 0) / (docs.length || 1) || 1, f };
  indexes.set(col, ix);
  return ix;
}
function bm25(qt: string[], c: Feat, ix: Index) {
  let s = 0;
  new Set(qt).forEach((t) => {
    const f = c.tf.get(t);
    if (!f) return;
    const n = ix.df.get(t) || 0;
    const idf = Math.log(1 + (ix.N - n + 0.5) / (n + 0.5));
    s += (idf * f * 2.2) / (f + 1.2 * (0.25 + (0.75 * c.len) / ix.avg));
  });
  return s;
}
const cos = (a: Float32Array, b: Float32Array) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
function relevance(qt: string[], c: Feat, dn: number) {
  const u = [...new Set(qt)];
  if (!u.length) return 0;
  const cov = u.filter((t) => c.set.has(t)).length / u.length;
  const pc = u.filter((t) => c.ptoks.has(t)).length / u.length;
  let bg = 0;
  let bn = 0;
  for (let i = 0; i < qt.length - 1; i++) {
    bn++;
    if (c.set.has(qt[i]) && c.set.has(qt[i + 1])) bg++;
  }
  return Math.min(1, 0.5 * cov + 0.2 * pc + 0.2 * Math.max(0, dn) * 1.4 + 0.1 * (bn ? bg / bn : 0));
}
export const label = (c: Chunk) => c.path.slice(-2).join(" › ");

/* ---------- retrieve: dense + keyword → RRF → rerank → context ---------- */
export function retrieve(col: Collection, set: Settings, q: string): Retrieval {
  const ix = ensureIdx(col);
  const k = set.k;
  const qt = toks(q);
  const F = (c: Chunk) => ix.f.get(c.id)!;
  const kw = col.chunks.map((c) => ({ c, s: bm25(qt, F(c), ix) })).filter((x) => x.s > 0).sort((a, b) => b.s - a.s).slice(0, k);
  const ex: string[] = [];
  const ew: number[] = [];
  qt.forEach((t) => {
    ex.push(t);
    ew.push(1);
    (SYNG.get(t) || []).forEach((s) => {
      if (s !== t) {
        ex.push(s);
        ew.push(0.5);
      }
    });
  });
  const qv = vec(ex, ew);
  const dnAll = col.chunks.map((c) => ({ c, s: cos(qv, F(c).vec) }));
  const dense = dnAll.slice().sort((a, b) => b.s - a.s).slice(0, k);
  const dsc = new Map(dnAll.map((x) => [x.c.id, x.s]));
  const fm = new Map<string, Fused>();
  dense.forEach((x, i) => fm.set(x.c.id, { c: x.c, d: i + 1, k: null, rrf: 1 / (60 + i + 1), rank: 0, s: 0, tag: "" }));
  kw.forEach((x, i) => {
    const e = fm.get(x.c.id) || { c: x.c, d: null, k: null, rrf: 0, rank: 0, s: 0, tag: "" };
    e.k = i + 1;
    e.rrf += 1 / (60 + i + 1);
    fm.set(x.c.id, e);
  });
  const fusion = [...fm.values()].sort((a, b) => b.rrf - a.rrf || (a.d || 99) - (b.d || 99));
  fusion.forEach((f, i) => {
    f.rank = i + 1;
    f.s = relevance(qt, F(f.c), dsc.get(f.c.id) || 0);
  });
  const best = fusion.reduce((m, f) => Math.max(m, f.s), 0);
  const order = set.rerank ? fusion.slice().sort((a, b) => b.s - a.s) : fusion.slice();
  const seen = new Set<string>();
  const ctx: Chunk[] = [];
  const rerank: Reranked[] = order.map((f, i) => {
    const r: Reranked = { c: f.c, s: f.s, from: f.rank, to: i + 1, tag: "" };
    if (set.one && seen.has(f.c.page)) r.tag = "drop";
    else if (ctx.length < 3 && f.s >= 0.2) {
      r.tag = "ctx";
      ctx.push(f.c);
      seen.add(f.c.page);
    }
    return r;
  });
  fusion.forEach((f) => {
    f.tag = rerank.find((x) => x.c.id === f.c.id)?.tag || "";
  });
  return { qt, dense, kw, fusion, rerank, ctx, best, refused: best < THRESH || !ctx.length, k, ms: { dn: rnd(34, 58), kw: rnd(9, 19), fu: rnd(2, 5), rr: rnd(120, 190), ci: rnd(1, 3) } };
}

/* ---------- compose: extractive answer with [n] citations ---------- */
export function compose(qt: string[], ctx: Chunk[]): { parts: Part[]; cites: Cite[] } {
  const qset = new Set(qt);
  const exp = new Set<string>();
  qt.forEach((t) => (SYNG.get(t) || []).forEach((x) => exp.add(x)));
  const parts: Part[] = [];
  const cites: Cite[] = [];
  let n = 0;
  ctx.forEach((c, ci) => {
    const sents = c.text.split(/(?<=[.!?])\s+(?=[A-Z`])/);
    const sc = sents.map((s, i) => {
      let v = 0;
      new Set(toks(s)).forEach((t) => {
        if (qset.has(t)) v += 1;
        else if (exp.has(t)) v += 0.5;
      });
      return { s, i, v };
    });
    const pick = sc.filter((x) => x.v >= (ci === 0 ? 0.5 : 1)).sort((a, b) => b.v - a.v).slice(0, ci === 0 ? 2 : 1).sort((a, b) => a.i - b.i);
    if (!pick.length) return;
    n++;
    cites.push({ n, id: c.id });
    parts.push({ t: "t", v: (parts.length ? " " : "") + pick.map((x) => x.s).join(" ") }, { t: "c", n, id: c.id });
  });
  if (!parts.length) {
    const c = ctx[0];
    parts.push({ t: "t", v: c.text.split(/(?<=[.!?])\s+/)[0] }, { t: "c", n: 1, id: c.id });
    cites.push({ n: 1, id: c.id });
  }
  return { parts, cites };
}
