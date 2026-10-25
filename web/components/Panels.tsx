"use client";
import { type ReactNode, useRef } from "react";
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
  const take = (f?: File) => { if (f && !busy) void A.upload(f); };
  return (
    <section className="panel" id="p-src" aria-labelledby="h-src">
      <div className="ph"><span className="num">01</span><h2 id="h-src">Source</h2></div>
      <div className="pb src-b" id="src">
        <button className="drop" type="button" disabled={busy} onClick={() => fileIn.current?.click()}
          onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add("over"); }}
          onDragLeave={(e) => e.currentTarget.classList.remove("over")}
          onDrop={(e) => { e.preventDefault(); e.currentTarget.classList.remove("over"); take(e.dataTransfer.files[0]); }}>
          <span className="dz-t">Drop a Markdown file, or press to choose</span>
          <span className="dz-s">.md or .txt, up to 1 MB. Kept for up to 24 hours, then deleted.</span>
        </button>
        <input ref={fileIn} type="file" accept=".md,.markdown,.txt,text/markdown,text/plain" hidden aria-label="Choose a Markdown or text file"
          onChange={(e) => { take(e.target.files?.[0]); e.target.value = ""; }} />
        <form className="paste" onSubmit={(e) => {
          e.preventDefault();
          const t = String(new FormData(e.currentTarget).get("text") || "").trim();
          if (t && !busy) { const f = e.currentTarget; void A.upload(t).then((ok) => ok && f.reset()); }
        }}>
          <label className="lbl" htmlFor="paste-in">Or paste text</label>
          <textarea id="paste-in" name="text" rows={4} spellCheck={false} disabled={busy} placeholder="# Notes&#10;Paste Markdown or plain text here" />
          <button className="btn" type="submit" disabled={busy}>Index this text</button>
        </form>
        {S.srcError && <div className="errbox" role="alert"><div className="et"><Icon name="alert" /><span>{S.srcError}</span></div></div>}
        <div className="od-stack" style={{ "--od-gap": "12px" } as React.CSSProperties}>
          <div className="or">collection</div>
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

const Bar = ({ v, i }: { v: number; i: number }) => (
  <div className="bar"><i style={{ "--v": Math.max(0, Math.min(1, v)).toFixed(3), "--i": i } as React.CSSProperties} /></div>
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
      <section className={cls(k)} data-st={st[k].st} aria-label={name}>
        <QH n={n} name={name} sub={sub} k={k} />
        <ol className="hits">
          {st[k].st !== "done" ? <Skel /> : arr.length ? arr.map((x, i) => (
            <Row key={x.c.id} c={x.c} i={i}>
              <div className="od-row"><span className="rk mono">{i + 1}</span><span className="lb od-fill od-truncate">{label(x.c)}</span><span className="sc mono od-fixed">{f(x.s)}</span></div>
              <Bar v={x.s / m} i={i} />
            </Row>
          )) : <li className="hit none">No match for these terms.</li>}
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
        <div className="row2">
          {search("dense", "01", "Dense search", "pgvector · cosine · 384-dim", R.dense, (v) => v.toFixed(3))}
          {search("keyword", "02", "Keyword search", "Postgres FTS · ts_rank", R.kw, (v) => v.toFixed(2))}
        </div>
        <section className={cls("fusion")} data-st={st.fusion.st} aria-label="Fusion">
          <QH n="03" name="Fusion (RRF)" sub="reciprocal rank fusion · k=60" k="fusion" />
          {st.fusion.st === "done" ? (
            <>
              <ol className="hits">
                {R.fusion.slice(0, 6).map((f, i) => (
                  <Row key={f.c.id} c={f.c} i={i}>
                    <div className="od-row">
                      <span className="rk mono">{i + 1}</span><span className="lb od-fill od-truncate">{label(f.c)}</span>
                      <span className="fd mono od-fixed">D{f.d || "-"} · K{f.k || "-"}</span>
                      {!S.set.rerank && f.tag === "ctx" && <span className="tag ctx od-fixed">context</span>}
                      <span className="sc mono od-fixed">{f.rrf.toFixed(4)}</span>
                    </div>
                    <Bar v={f.rrf / fmx} i={i} />
                  </Row>
                ))}
              </ol>
              <p className="formula">score = Σ 1 / (60 + rank) · D = dense rank, K = keyword rank</p>
            </>
          ) : <ol className="hits"><Skel /></ol>}
        </section>
        <section className={cls("rerank")} data-st={st.rerank.st} aria-label="Rerank">
          <QH n="04" name="Rerank" sub="cross-encoder · ms-marco-MiniLM-L-6" k="rerank" />
          {st.rerank.st === "skipped" ? <p className="note">Rerank is off. Fusion order goes straight to the generator. Toggle it with <kbd>R</kbd>.</p>
            : st.rerank.st === "done" ? (
              <>
                <ol className="hits">
                  {R.rerank.slice(0, 6).map((r, i) => {
                    const d = r.from - r.to;
                    return (
                      <li key={r.c.id} className={cx("hit", r.tag === "ctx" && "in-ctx", h(r.c.id))} data-chunk={r.c.id} style={{ "--i": i } as React.CSSProperties}>
                        <div className="od-row">
                          <span className="rk mono">{i + 1}</span><span className="lb od-fill od-truncate">{label(r.c)}</span>
                          {r.tag === "ctx" ? <span className="tag ctx od-fixed">context</span> : r.tag === "drop" ? <span className="tag drop od-fixed">same page</span> : null}
                          <span className={cx("chg od-fixed", d > 0 && "up", d < 0 && "down")} aria-label={d > 0 ? `up ${d}` : d < 0 ? `down ${-d}` : "unchanged"}>{d > 0 ? "↑" + d : d < 0 ? "↓" + -d : "="}</span>
                          <span className="sc mono od-fixed">{r.s.toFixed(2)}</span>
                        </div>
                        <Bar v={r.s} i={i} />
                      </li>
                    );
                  })}
                </ol>
                <p className="formula">cross-encoder relevance (sigmoid of the logit) · higher is more relevant</p>
              </>
            ) : <ol className="hits"><Skel /></ol>}
        </section>
        <section className={cls("generate")} data-st={go} aria-label="Generate">
          <QH n="05" name="Generate" sub={S.info?.llm_model || ""} k="generate" />
          {go === "waiting" ? <p className="note">Waits for the reranked context.</p>
            : go === "running" ? <><p className="note">Streaming from {S.info?.llm_model || "the model"} · <span className="mono">{q.tok}</span> tokens out</p>{chips}</>
            : go === "done" ? <><p className="note">Answer complete · <span className="mono">{q.tout}</span> tokens out</p>{chips}</>
            : go === "skipped" ? <p className="note">Not called: retrieval found no passage.</p>
            : <><p className="note">No answer this time. Retrieval results are kept.</p>{chips}</>}
        </section>
        <section className={cls("cite")} data-st={st.cite.st} aria-label="Cite">
          <QH n="06" name="Cite" sub="only cited passages are listed as sources" k="cite" />
          {st.cite.st === "done" ? (
            <ol className="hits">
              {R.ctx.map((c, i) => {
                const ci = cites.find((x) => x.id === c.id);
                return (
                  <Row key={c.id} c={c} i={i}>
                    <div className="od-row"><span className="cite od-fixed">{ci ? ci.n : "-"}</span><span className="lb od-fill od-truncate">{label(c)}</span><span className={cx("tag od-fixed", ci && "on")}>{ci ? "cited" : "not cited"}</span></div>
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

export function PipelinePanel(p: P) {
  const { S, A, h } = p;
  const pstate = S.phase === "ingesting" ? ["busy", "Connecting"] : S.apiDown ? ["idle", "Offline"] : S.busy ? ["busy", "Answering"] : S.phase === "ready" ? ["ready", "Ready"] : ["idle", "Idle"];
  const c = S.col;
  return (
    <section className="panel hero" id="p-pipe" aria-labelledby="h-pipe">
      <div className="ph">
        <span className="num">02</span><h2 id="h-pipe">Pipeline</h2><span className="grow" />
        <span className="pstate" data-s={pstate[0]}><i /><span>{pstate[1]}</span></span>
      </div>
      <div className="pipe-b" id="pipe">
        <div>{S.phase === "empty" && !S.q && <Explain />}</div>
        <div>
          <div className="sec-h"><h3 className="lbl">Ingest</h3><span className="cap">{c && S.phase === "ready" ? (c.id === "k8s" ? "Kubernetes docs · indexed ahead of time" : `${c.name} · deleted after 24 h`) : ""}</span></div>
          <div className="ing-grid">{ING.map(([k, name], i) => <IngCard key={k} {...p} i={i} k={k} name={name} />)}</div>
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
  const citeBtn = (id: string, n: number, key?: string | number) => (
    <button key={key} className={cx("cite", h(id))} type="button" data-chunk={id} data-n={n} aria-label={`Citation ${n}`} onClick={(e) => A.pin(e.currentTarget)}>{n}</button>
  );
  if (m.status === "refusal") {
    return (
      <article className="msg">
        <Hd r={<span className="mono">{fmtMs(m.ms || 0)}</span>} />
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
      <Hd r={<span className="mono">{done ? fmtMs(m.ms || 0) : "streaming"}</span>} />
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
    <section className="panel ask" id="p-ask" aria-labelledby="h-ask">
      <div className="ph"><span className="num">03</span><h2 id="h-ask">Ask</h2></div>
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
  const t = (x: number) => (x ? (x >= 1000 ? (x / 1000).toFixed(2) + " s" : Math.round(x) + " ms") : "-");
  const F = ({ l, v, c }: { l: string; v: string; c?: string }) => <div className={cx("fs od-row", c)}><span className="fl">{l}</span><span className="fv mono">{v}</span></div>;
  return (
    <footer className="runstats" aria-label="Run stats">
      <F l="Total latency" v={s ? t(s.total) : "-"} />
      <F l="Retrieve" v={s ? t(s.retrieve) : "-"} />
      <F l="Rerank" v={s ? t(s.rerank) : "-"} />
      <F l="Generate" v={s ? t(s.gen) : "-"} />
      <F l="Tokens out" v={s ? String(s.tout) : "-"} />
      <F l="Model" v={S.info?.llm_model || "-"} />
      <F l="Embedding" v={S.info ? `${S.info.embedding_model} · 384-dim` : "-"} c="right" />
      <div className="fs od-row credits">
        <span>Docs © <a href="https://github.com/kubernetes/website">The Kubernetes Authors</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></span>
        <a href="/legal">Legal</a>
        <a href="/privacy">Privacy</a>
      </div>
    </footer>
  );
}
