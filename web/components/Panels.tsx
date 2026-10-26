"use client";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { stepLabel } from "@/lib/api.ts";
import type { Chunk, StageState } from "@/lib/types.ts";
import { Icon, Md } from "./Icons";
import { cx, fmtMs, type IngKey, type Msg, type QKey, type Store, STL } from "./store";

const label = (c: Chunk) => c.path.slice(-2).join(" › ");

export type Actions = {
  pick: (id: "k8s" | "up") => void;
  upload: (input: File | string) => Promise<boolean>;
  toggleChunks: () => void;
  ask: (text: string) => void;
  retry: () => void;
  pin: (el: HTMLElement) => void;
};
type P = { S: Store; A: Actions; h: (id: string) => string };

const Stt = ({ st }: { st: StageState }) => (<span className="stt"><i className="dot" />{STL[st]}</span>);

/* ---------- 01 Source ---------- */
const collSub = (c: { docs: number; total: number }) => `${c.docs.toLocaleString("en-US")} docs · ${c.total.toLocaleString("en-US")} chunks`;

export function SourcePanel({ S, A }: P) {
  const busy = S.phase === "ingesting" || S.busy;
  const fileIn = useRef<HTMLInputElement>(null);
  const [paste, setPaste] = useState(false);
  const take = (f?: File) => { if (f && !busy) void A.upload(f); };
  return (
    <section className="panel strip-src" id="p-src" aria-label="Source">
      <div className="src-row" id="src">
        <button className="btn drop" type="button" disabled={busy} onClick={() => fileIn.current?.click()}
          onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add("over"); }}
          onDragLeave={(e) => e.currentTarget.classList.remove("over")}
          onDrop={(e) => { e.preventDefault(); e.currentTarget.classList.remove("over"); take(e.dataTransfer.files[0]); }}>
          Drop file
        </button>
        <input ref={fileIn} type="file" accept=".md,.markdown,.txt,text/markdown,text/plain" hidden aria-label="Choose a Markdown or text file"
          onChange={(e) => { take(e.target.files?.[0]); e.target.value = ""; }} />
        <button className="btn" type="button" disabled={busy} aria-expanded={paste} aria-controls="paste-form" onClick={() => setPaste(!paste)}>Paste text</button>
        <span className="src-cap">.md or .txt, up to 1 MB. Kept for up to 24 hours, then deleted.</span>
        <div className="colls">
          <button className="collbtn" type="button" disabled={busy} onClick={() => A.pick("k8s")} aria-pressed={S.col?.id === "k8s"}>
            <span className="t">Kubernetes docs</span>
            <span className="s mono">{S.info && S.col?.id === "k8s" ? collSub(S.col) : S.apiDown ? "API unreachable · retry" : S.info ? "switch back" : "loading…"}</span>
          </button>
          {S.up && (
            <button className="collbtn" type="button" disabled={busy} onClick={() => A.pick("up")} aria-pressed={S.col?.id === S.up.col.id}>
              <span className="t od-truncate">{S.up.col.name}</span>
              <span className="s mono">{collSub(S.up.col)} · deleted after 24 h</span>
            </button>
          )}
        </div>
      </div>
      {paste && (
        <form className="paste" id="paste-form" onSubmit={(e) => {
          e.preventDefault();
          const t = String(new FormData(e.currentTarget).get("text") || "").trim();
          if (t && !busy) { const f = e.currentTarget; void A.upload(t).then((ok) => { if (ok) { f.reset(); setPaste(false); } }); }
        }}>
          <label className="lbl" htmlFor="paste-in">Paste text</label>
          <textarea id="paste-in" name="text" rows={4} spellCheck={false} disabled={busy} placeholder="# Notes&#10;Paste Markdown or plain text here" />
          <button className="btn" type="submit" disabled={busy}>Index this text</button>
        </form>
      )}
      {S.srcError && <div className="errbox src-err" role="alert"><div className="et"><Icon name="alert" /><span>{S.srcError}</span></div></div>}
    </section>
  );
}

/* ---------- 02 Pipeline ---------- */
const ING: [IngKey, string][] = [["parse", "Parse"], ["chunk", "Chunk"], ["embed", "Embed"], ["index", "Index"]];

function IngCard({ S, A, i, k, name }: P & { i: number; k: IngKey; name: string }) {
  const o = S.ing ? S.ing[k] : { st: "waiting" as StageState };
  const st = o.st, isChunk = k === "chunk";
  const keyTxt = st === "error" ? <div className="key bad">{o.key || "Failed"}</div>
    : st === "done" || (st === "running" && o.key) ? <div className="key">{o.key}</div>
    : <div className="key dim">{st === "running" ? "Working…" : "-"}</div>;
  const ms = st === "done" ? fmtMs(o.ms || 0) : st === "running" ? fmtMs(performance.now() - (o.t0 || 0)) : "-";
  const total = S.col?.total || 0;
  const inner = (
    <>
      <div className="st-top"><span className="num">0{i + 1}</span><h3 className="nm">{name}</h3></div>
      {keyTxt}
      <div className="od-row"><Stt st={st} /><span className="od-fill" /><span className="ms mono od-fixed">{ms}</span></div>
      {k === "embed" && st === "running" && o.frac != null && (
        <>
          <div className="prog"><i style={{ "--v": o.frac || 0 } as React.CSSProperties} /></div>
          <div className="cnt mono">{Math.round((o.frac || 0) * total).toLocaleString("en-US")} / {total.toLocaleString("en-US")}</div>
        </>
      )}
      {isChunk && st === "done" && (S.col?.chunks.length || 0) > 0 && <div className="reveal">{S.chunkOpen ? "Hide" : "Show"} chunks <Icon name="chev" /></div>}
    </>
  );
  const cls = cx("stage", S.fresh === k && "fresh");
  return isChunk ? (
    <button className={cls} type="button" data-st={st} onClick={A.toggleChunks} aria-expanded={S.chunkOpen && st === "done"} aria-controls="strip" disabled={st !== "done"}>{inner}</button>
  ) : (
    <div className={cls} data-st={st}>{inner}</div>
  );
}

function Explain() {
  return (
    <div className="explain">
      <h1>See every step between a file and an answer.</h1>
      <div className="zr"><span className="n">01</span><div><b>Source</b><span className="d">Drop a Markdown file, paste text, or use the Kubernetes docs.</span></div></div>
      <div className="zr"><span className="n">02</span><div><b>Pipeline</b><span className="d">Watch each stage run, with timings, scores and rank changes.</span></div></div>
      <div className="zr"><span className="n">03</span><div><b>Ask</b><span className="d">Every claim cites a passage. Hover a citation to trace it back.</span></div></div>
    </div>
  );
}

const SHOWN = 5; // rows per retrieval list; the rest collapse to "+N more"
const Bar = ({ v, i }: { v: number; i: number }) => (
  <div className="bar od-fixed"><i style={{ "--v": Math.max(0, Math.min(1, v)).toFixed(3), "--i": i } as React.CSSProperties} /></div>
);
const Skel = () => <>{Array.from({ length: 5 }, (_, i) => <li key={i} className="hit sk"><div className="skl" /></li>)}</>;

function QueryFlow({ S, h }: P) {
  const q = S.q!, R = q.R, st = q.st;
  const cls = (k: QKey) => cx("stage qs", !q.seen.has(k) && "enter", S.fresh === k && "fresh");
  const ctxIds = new Set(R.ctx.map((c) => c.id));
  const QH = ({ n, name, sub, k }: { n: string; name: string; sub: string; k: QKey }) => (
    <div className="od-row qh">
      <span className="num">{n}</span><h3 className="nm">{name}</h3>
      <span className="sub mono od-fill od-truncate">{sub}</span>
      <span className="od-fixed"><Stt st={st[k].st} /></span>
      <span className="ms mono od-fixed">{st[k].ms != null ? fmtMs(st[k].ms!) : ""}</span>
    </div>
  );
  const Row = ({ c, i, children }: { c: Chunk; i: number; children: ReactNode }) => (
    <li className={cx("hit", ctxIds.has(c.id) && "in-ctx", h(c.id))} data-chunk={c.id} style={{ "--i": i } as React.CSSProperties}>{children}</li>
  );
  const search = (k: QKey, n: string, name: string, sub: string, arr: { c: Chunk; s: number }[], f: (v: number) => string) => {
    const m = Math.max(...arr.map((x) => x.s), 1e-9);
    return (
      <section className={cls(k)} id={`st-${k}`} data-st={st[k].st} aria-label={name}>
        <QH n={n} name={name} sub={sub} k={k} />
        <ol className="hits">
          {st[k].st !== "done" ? <Skel /> : arr.length ? <>
            {arr.slice(0, SHOWN).map((x, i) => (
              <Row key={x.c.id} c={x.c} i={i}>
                <div className="od-row"><span className="rk mono">{i + 1}</span><span className="lb od-fill od-truncate" title={label(x.c)}>{label(x.c)}</span><Bar v={x.s / m} i={i} /><span className="sc mono od-fixed">{f(x.s)}</span></div>
              </Row>
            ))}
            {arr.length > SHOWN && <li className="more">+{arr.length - SHOWN} more</li>}
          </> : <li className="hit none">No match for these terms.</li>}
        </ol>
      </section>
    );
  };
  const fmx = Math.max(...R.fusion.map((f) => f.rrf), 1e-9);
  const m = S.msgs[S.msgs.length - 1];
  const cites = m && m.role === "a" ? m.cites : [];
  const chips = R.ctx.length > 0 && (
    <div className="od-cluster" style={{ marginTop: 8 }}>
      {R.ctx.map((c, i) => <span key={c.id} className={cx("pill mono", h(c.id))} data-chunk={c.id}><span className="od-truncate">P{i + 1} {c.path[c.path.length - 1]}</span></span>)}
    </div>
  );
  const go = st.generate.st;
  return (
    <div id="qry">
      <div className="sec-h"><h3 className="lbl">Query</h3></div>
      <p className="qq">“<q style={{ quotes: "none" }}>{q.text}</q>”</p>
      <div className="qflow">
        <h3 className="grp">Retrieve</h3>
        <div className="row2">
          {search("dense", "05", "Dense search", "pgvector · cosine · 384-dim", R.dense, (v) => v.toFixed(3))}
          {search("keyword", "06", "Keyword search", "Postgres FTS · ts_rank", R.kw, (v) => v.toFixed(2))}
        </div>
        <h3 className="grp">Fuse and rank</h3>
        <section className={cls("fusion")} id="st-fusion" data-st={st.fusion.st} aria-label="Fusion">
          <QH n="07" name="Fusion (RRF)" sub="reciprocal rank fusion · k=60" k="fusion" />
          {st.fusion.st === "done" ? (
            <>
              <ol className="hits">
                {R.fusion.slice(0, 6).map((f, i) => (
                  <Row key={f.c.id} c={f.c} i={i}>
                    <div className="od-row">
                      <span className="rk mono">{i + 1}</span><span className="lb od-fill od-truncate" title={label(f.c)}>{label(f.c)}</span>
                      <span className="fd mono od-fixed">D{f.d || "-"} · K{f.k || "-"}</span>
                      {!S.set.rerank && f.tag === "ctx" && <span className="tag ctx od-fixed">context</span>}
                      <Bar v={f.rrf / fmx} i={i} />
                      <span className="sc mono od-fixed">{f.rrf.toFixed(4)}</span>
                    </div>
                  </Row>
                ))}
              </ol>
              <p className="formula">score = Σ 1 / (60 + rank) · D = dense rank, K = keyword rank</p>
            </>
          ) : <ol className="hits"><Skel /></ol>}
        </section>
        <section className={cls("rerank")} id="st-rerank" data-st={st.rerank.st} aria-label="Rerank">
          <QH n="08" name="Rerank" sub="cross-encoder · ms-marco-MiniLM-L-6" k="rerank" />
          {st.rerank.st === "skipped" ? <p className="note">Rerank is off. Fusion order goes straight to the generator. Toggle it with <kbd>R</kbd>.</p>
            : st.rerank.st === "done" ? (
              <>
                <ol className="hits">
                  {R.rerank.slice(0, 6).map((r, i) => {
                    const d = r.from - r.to;
                    return (
                      <li key={r.c.id} className={cx("hit", r.tag === "ctx" && "in-ctx", h(r.c.id))} data-chunk={r.c.id} style={{ "--i": i } as React.CSSProperties}>
                        <div className="od-row">
                          <span className="rk mono">{i + 1}</span><span className="lb od-fill od-truncate" title={label(r.c)}>{label(r.c)}</span>
                          {r.tag === "ctx" ? <span className="tag ctx od-fixed">context</span> : r.tag === "drop" ? <span className="tag drop od-fixed">same page</span> : null}
                          <span className={cx("chg od-fixed", d > 0 && "up", d < 0 && "down")} aria-label={d > 0 ? `up ${d}` : d < 0 ? `down ${-d}` : "unchanged"}>{d > 0 ? "↑" + d : d < 0 ? "↓" + -d : "="}</span>
                          <Bar v={r.s} i={i} />
                          <span className="sc mono od-fixed">{r.s.toFixed(2)}</span>
                        </div>
                      </li>
                    );
                  })}
                </ol>
                <p className="formula">cross-encoder relevance (sigmoid of the logit) · higher is more relevant</p>
              </>
            ) : <ol className="hits"><Skel /></ol>}
        </section>
        <h3 className="grp">Generate</h3>
        <section className={cls("generate")} id="st-generate" data-st={go} aria-label="Generate">
          <QH n="09" name="Generate" sub={S.info?.llm_model || ""} k="generate" />
          {go === "waiting" ? <p className="note">Waits for the reranked context.</p>
            : go === "running" ? <><p className="note">Streaming from {S.info?.llm_model || "the model"} · <span className="mono">{q.tok}</span> tokens out</p>{chips}</>
            : go === "done" ? <><p className="note">Answer complete · <span className="mono">{q.tout}</span> tokens out</p>{chips}</>
            : go === "skipped" ? <p className="note">Not called: retrieval found no passage.</p>
            : <><p className="note">No answer this time. Retrieval results are kept.</p>{chips}</>}
        </section>
        <section className={cls("cite")} data-st={st.cite.st} aria-label="Cite">
          <QH n="" name="Cite" sub="only cited passages are listed as sources" k="cite" />
          {st.cite.st === "done" ? (
            <ol className="hits">
              {R.ctx.map((c, i) => {
                const ci = cites.find((x) => x.id === c.id);
                return (
                  <Row key={c.id} c={c} i={i}>
                    <div className="od-row"><span className="cite od-fixed">{ci ? ci.n : "-"}</span><span className="lb od-fill od-truncate" title={label(c)}>{label(c)}</span><span className={cx("tag od-fixed", ci && "on")}>{ci ? "cited" : "not cited"}</span></div>
                  </Row>
                );
              })}
            </ol>
          ) : st.cite.st === "skipped" ? <p className="note">Nothing to cite.</p> : <p className="note">Maps each [n] in the answer to its passage.</p>}
        </section>
      </div>
    </div>
  );
}

const fmtT = (x: number) => (x ? (x >= 1000 ? (x / 1000).toFixed(2) + " s" : Math.round(x) + " ms") : "-");
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// Stages the cited chunk passed through, with its rank at each: drives the rail while a chunk is hovered or pinned.
const traceOf = (S: Store): Partial<Record<string, string>> => {
  const id = S.hl, q = S.q;
  if (!id || !q) return {};
  const R = q.R, t: Partial<Record<string, string>> = {};
  const at = (a: { c: Chunk }[]) => a.findIndex((x) => x.c.id === id) + 1;
  const m = S.msgs[S.msgs.length - 1];
  const ci = m && m.role === "a" ? m.cites.find((x) => x.id === id) : undefined;
  const rk: [string, number][] = [["dense", at(R.dense)], ["keyword", at(R.kw)], ["fusion", at(R.fusion)], ["rerank", at(R.rerank)]];
  rk.forEach(([k, n]) => { if (n) t[k] = "#" + n; });
  if (ci) t.generate = `[${ci.n}]`;
  else if (R.ctx.some((c) => c.id === id)) t.generate = "ctx";
  return t;
};

const RAIL: [string, string, string][] = [
  ["ing", "1", "Parse"], ["ing", "2", "Chunk"], ["ing", "3", "Embed"], ["ing", "4", "Index"],
  ["dense", "5", "Dense"], ["keyword", "6", "Keyword"], ["fusion", "7", "Fusion"], ["rerank", "8", "Rerank"], ["generate", "9", "Generate"],
];
const DIVIDE_AFTER = new Set(["4", "6", "8"]);

function Rail({ S }: { S: Store }) {
  const t = traceOf(S);
  const [here, setHere] = useState("ing");
  useEffect(() => {
    const pipe = document.getElementById("pipe");
    if (!pipe) return;
    let raf = 0;
    const on = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        let cur = "ing";
        const y = pipe.scrollTop + 24;
        for (const [k] of RAIL) { const el = document.getElementById("st-" + k); if (el && el.offsetTop <= y) cur = k; }
        if (pipe.scrollTop + pipe.clientHeight >= pipe.scrollHeight - 2) cur = "generate";
        setHere(cur);
      });
    };
    pipe.addEventListener("scroll", on, { passive: true });
    on();
    return () => { pipe.removeEventListener("scroll", on); cancelAnimationFrame(raf); };
  }, [S.q]);
  const go = (k: string) => {
    const pipe = document.getElementById("pipe"), el = document.getElementById("st-" + k);
    if (pipe && el) pipe.scrollTo({ top: el.offsetTop - 8, behavior: reducedMotion() ? "auto" : "smooth" });
  };
  let lit = 0;
  return (
    <nav className="stg-rail" aria-label="Pipeline stages">
      {RAIL.map(([k, n, name]) => {
        const r = t[k];
        const i = r ? lit++ : 0;
        return (
          <span key={n} className="rail-seg">
            <button type="button" className={cx("it", r && "lit", here === k && (k !== "ing" || n === "1") && "here")} style={{ "--i": i } as React.CSSProperties}
              onClick={() => go(k)} aria-label={`Go to stage ${n}, ${name}${r ? `, cited chunk at ${r}` : ""}`}>
              <span className="n">{n}</span><span>{name}</span>{r && <span className="rk">{r}</span>}
            </button>
            {DIVIDE_AFTER.has(n) && <i className="dv" aria-hidden="true" />}
          </span>
        );
      })}
    </nav>
  );
}

function Hero({ S }: { S: Store }) {
  const q = S.q!, R = q.R, s = S.stats;
  const m = S.msgs[S.msgs.length - 1];
  const a = m && m.role === "a" ? m : null;
  const cited = a ? a.cites.length : 0;
  const seg: [string, number][] = s ? [["retrieve", s.retrieve], ["rerank", s.rerank], ["generate", s.gen]] : [];
  const slow = Math.max(...seg.map((x) => x[1]), 0);
  const sum = !s && a?.status !== "error" ? "Running the pipeline…"
    : a?.status === "error" ? "Retrieval finished. The model call failed, and the results are kept."
    : a?.status === "refusal" ? "No passage answered the question, so the model declined."
    : `${S.set.rerank ? `${R.rerank.length} of ${R.fusion.length} candidates were reranked` : `${R.fusion.length} candidates were fused`}, ${R.ctx.length} reached the model${cited ? ` and ${cited === R.ctx.length ? "all " : ""}${cited} ${cited === 1 ? "was" : "were"} cited` : ""}`;
  return (
    <div className="hero-band">
      <div className="hero-fig mono">{s ? fmtT(s.total) : "-"}<small>total latency</small></div>
      <div className="hero-r">
        <p className="hero-sum">{sum}</p>
        {s && (
          <>
            <div className="tbar" role="img" aria-label={seg.map(([k, v]) => `${k} ${fmtT(v)}`).join(", ")}>
              {seg.map(([k, v]) => <i key={k} className={v === slow ? "max" : ""} style={{ "--w": Math.max(v, 1) } as React.CSSProperties} />)}
            </div>
            <div className="tleg mono">{seg.map(([k, v]) => <span key={k}><i className={v === slow ? "max" : ""} />{k} {fmtT(v)}</span>)}</div>
          </>
        )}
      </div>
    </div>
  );
}

export function PipelinePanel(p: P) {
  const { S, A, h } = p;
  const c = S.col;
  return (
    <section className="panel pipeline" id="p-pipe" aria-label="Pipeline">
      {S.q ? <Hero S={S} /> : <div />}
      {S.q ? <Rail S={S} /> : <div />}
      <div className="pipe-b" id="pipe">
        <div>{S.phase === "empty" && !S.q && <Explain />}</div>
        <div>
          <div className="sec-h"><h3 className="grp">Ingest</h3><span className="cap">{c && S.phase === "ready" ? (c.id === "k8s" ? "Kubernetes docs · indexed ahead of time" : `${c.name} · deleted after 24 h`) : ""}</span></div>
          <div className="ing-grid" id="st-ing">{ING.map(([k, name], i) => <IngCard key={k} {...p} i={i} k={k} name={name} />)}</div>
          {S.chunkOpen && S.ing?.chunk.st === "done" && c && (
            <div className="strip-wrap" id="strip">
              <div className="od-rail strip" role="list" aria-label="Chunks">
                {c.chunks.map((k) => (
                  <article key={k.id} className={cx("chunk", h(k.id))} role="listitem" data-chunk={k.id}>
                    <div className="pth od-truncate">{k.path.join(" › ")}</div>
                    <p className="tx od-clamp-3">{k.text}</p>
                    <div className="ft mono"><span className="od-nowrap">{k.chars} chars</span><span>{k.id}</span></div>
                  </article>
                ))}
              </div>
              <p className="strip-cap">{c.id === "k8s" ? `Showing ${c.chunks.length} of ${c.total.toLocaleString("en-US")} chunks (retrievable slice).` : `Showing the first ${c.chunks.length} of ${c.total.toLocaleString("en-US")} chunks.`}</p>
            </div>
          )}
        </div>
        {S.q && <QueryFlow S={S} A={A} h={h} />}
      </div>
    </section>
  );
}

/* ---------- 03 Ask ---------- */
const waitText = (retryAt?: number) => {
  const s = retryAt ? Math.ceil((retryAt - Date.now()) / 1000) : 0;
  return s > 0 ? `Retry in ${s} s.` : "";
};

function Message({ m, S, A, h }: P & { m: Msg }) {
  if (m.role === "u") return <article className="msg"><div className="who">You</div><p className="qt">{m.text}</p></article>;
  const Hd = ({ r }: { r: ReactNode }) => <div className="who"><span>Answer</span><span className="grow" />{r}</div>;
  const took = (text: string) => <span className="mono">{m.agent && "Agent · "}{m.impl === "langchain" && "LangChain · "}{text}</span>;
  const steps = m.steps?.length ? (
    <ol className="steps" aria-label="Searches the agent ran">
      {m.steps.map((s) => <li key={s.step} className={cx("mono", (s.error || s.duplicate) && "dim")}>{stepLabel(s)}</li>)}
    </ol>
  ) : m.agent && m.status === "streaming" ? <p className="steps-wait">The agent is deciding what to search…</p> : null;
  const citeBtn = (id: string, n: number, key?: string | number) => (
    <button key={key} className={cx("cite", h(id))} type="button" data-chunk={id} data-n={n} aria-label={`Citation ${n}`} onClick={(e) => A.pin(e.currentTarget)}>{n}</button>
  );
  if (m.status === "refusal") {
    return (
      <article className="msg">
        <Hd r={took(fmtMs(m.ms || 0))} />
        {steps}
        <div className="refusal" role="status">
          <div className="rl"><Icon name="ban" />Refused</div>
          <p className="rt">I don&apos;t know based on the provided documentation.</p>
          <p className="rn">The model found no passage that answers this question, so it declined instead of guessing.</p>
        </div>
      </article>
    );
  }
  if (m.status === "error") {
    return (
      <article className="msg">
        <Hd r={null} />
        <div className="errbox" role="alert">
          <div className="et"><Icon name="alert" /><span>{m.message || "Something went wrong."}</span></div>
          {waitText(m.retryAt) ? <p className="en">{waitText(m.retryAt)}</p> : <button className="btn" type="button" onClick={A.retry}>Retry</button>}
        </div>
      </article>
    );
  }
  const done = m.status === "done";
  return (
    <article className="msg">
      <Hd r={took(done ? fmtMs(m.ms || 0) : "streaming")} />
      {steps}
      <p className="ans">
        {m.parts.map((p, i) => (p.t === "t" ? <Md key={i} text={p.v} /> : citeBtn(p.id, p.n, i)))}
        {!done && <span className="caret" />}
      </p>
      {done && m.cites.length > 0 && (
        <div className="srcs">
          <div className="lbl">Sources</div>
          {m.cites.map((c) => {
            const ch = S.chunks.get(c.id);
            if (!ch) return null;
            return (
              <button key={c.id} className={cx("src", h(c.id))} type="button" data-chunk={c.id} data-n={c.n} onClick={(e) => A.pin(e.currentTarget)}>
                <span className="cite">{c.n}</span>
                <span className="od-stack" style={{ "--od-gap": 0 } as React.CSSProperties}>
                  <span className="p od-truncate">{ch.path.join(" › ")}</span>
                  <span className="e mono od-truncate">{ch.page.replace(/^https?:\/\//, "")}</span>
                </span>
              </button>
            );
          })}
        </div>
      )}
    </article>
  );
}

export function AskPanel({ S, A, h, chatRef, inputRef }: P & { chatRef: React.RefObject<HTMLDivElement | null>; inputRef: React.RefObject<HTMLTextAreaElement | null> }) {
  const ok = S.phase === "ready" && !!S.col;
  let body: ReactNode;
  if (S.msgs.length) body = S.msgs.map((m, i) => <Message key={i} m={m} S={S} A={A} h={h} />);
  else if (ok) {
    const qs = S.col!.id !== "k8s" ? ["Summarize this document.", "What are the key points?", "Who won the 2018 World Cup?"] : ["How do I roll back a failed Deployment?", "What's the difference between liveness and readiness probes?", "Why was my container OOMKilled?", "Who won the 2018 World Cup?"];
    body = (
      <div className="empty-ask">
        <h3>Ask about {S.col!.name}</h3>
        <p>Answers are built only from retrieved passages. The last suggestion is out of scope on purpose.</p>
        <div className="suggs">{qs.map((x) => <button key={x} className="sugg" type="button" onClick={() => A.ask(x)}>{x}</button>)}</div>
      </div>
    );
  } else {
    const t = S.apiDown ? ["Service unavailable", "The API is not reachable right now. Try again later."] : ["Loading the index", "Connecting to the Kubernetes docs index…"];
    body = <div className="empty-ask"><h3>{t[0]}</h3><p>{t[1]}</p></div>;
  }
  const submit = () => {
    const ta = inputRef.current;
    const v = ta?.value.trim();
    if (!ta || !v) return;
    ta.value = "";
    A.ask(v);
  };
  return (
    <section className="panel ask" id="p-ask" aria-label="Ask">
      <div className="chat" id="chat" ref={chatRef} role="log" aria-live="polite" aria-label="Conversation">{body}</div>
      <form className="composer" autoComplete="off" onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <label htmlFor="ask-in">Question</label>
        <div className="cbox">
          <textarea id="ask-in" ref={inputRef} rows={2} spellCheck={false} disabled={!ok}
            placeholder={ok ? "Ask about " + S.col!.name : "Add a source to start asking"}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); } }} />
          <button className="btn primary" type="submit" data-tip="Send · ⏎" disabled={!ok || S.busy}>
            <span>{S.busy ? "Answering…" : "Ask"}</span><kbd aria-hidden="true">⏎</kbd>
          </button>
        </div>
        <p className="fine">Questions go to a third-party AI model: don&apos;t include personal data. <a href="/privacy">Privacy</a></p>
      </form>
    </section>
  );
}

/* ---------- run stats ---------- */
export function RunStats({ S }: { S: Store }) {
  const s = S.stats;
  const t = fmtT;
  const pstate = S.phase === "ingesting" ? ["busy", "Connecting"] : S.apiDown ? ["idle", "Offline"] : S.busy ? ["busy", "Answering"] : S.phase === "ready" ? ["ready", "Ready"] : ["idle", "Idle"];
  const F = ({ l, v, c }: { l: string; v: string; c?: string }) => <div className={cx("fs od-row", c)}><span className="fl">{l}</span><span className="fv mono" title={v}>{v}</span></div>;
  return (
    <footer className="runstats" aria-label="Run stats">
      <div className="fs od-row pstate" data-s={pstate[0]}><i /><span>{pstate[1]}</span></div>
      <F l="Total latency" v={s ? t(s.total) : "-"} />
      <F l="Retrieve" v={s ? t(s.retrieve) : "-"} />
      <F l="Rerank" v={s ? t(s.rerank) : "-"} />
      <F l="Generate" v={s ? t(s.gen) : "-"} />
      <F l="Tokens out" v={s ? String(s.tout) : "-"} />
      <F l="Model" v={S.info?.llm_model || "-"} c="shrink" />
      <F l="Embedding" v={S.info ? `${S.info.embedding_model} · 384-dim` : "-"} c="right shrink" />
      <div className="fs od-row credits">
        <span>Docs © <a href="https://github.com/kubernetes/website">The Kubernetes Authors</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></span>
        <a href="/legal">Legal</a>
        <a href="/privacy">Privacy</a>
      </div>
    </footer>
  );
}
