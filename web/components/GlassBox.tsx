"use client";
import { useEffect, useReducer, useRef } from "react";
import { type ApiHit, ApiError, askStream, getCollection, type SSE, splitCitations, toChunk } from "@/lib/api.ts";
import type { Retrieval, Scored, StageState } from "@/lib/types.ts";
import { Icon } from "./Icons";
import { type Cmd, CitePopover, CollMenu, Palette, paletteItems, SettingsMenu } from "./Overlays";
import { type Actions, AskPanel, PipelinePanel, RunStats, SourcePanel } from "./Panels";
import { freshIng, type Msg, newStore, type QKey, type Store } from "./store";

type Ans = Extract<Msg, { role: "a" }>;

const UNAVAILABLE = "Service unavailable, try again later.";

export default function GlassBox() {
  const sRef = useRef<Store>(null as unknown as Store);
  if (!sRef.current) sRef.current = newStore();
  const S = sRef.current;
  const [, render] = useReducer((x: number) => x + 1, 0);
  const runId = useRef(0);
  const refs = { coll: useRef<HTMLButtonElement>(null), set: useRef<HTMLButtonElement>(null), theme: useRef<HTMLButtonElement>(null), chat: useRef<HTMLDivElement>(null), input: useRef<HTMLTextAreaElement>(null) };
  const popHide = useRef<ReturnType<typeof setTimeout>>(undefined);

  const RM = () => typeof window !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const announce = (t: string) => { S.sr = t; };

  /* ---------- overlays ---------- */
  const hidePop = (force: boolean) => {
    if (S.pop?.pinned && !force) return;
    S.pop = null;
    if (force) S.hl = null;
  };
  const closeAll = () => { S.pal.open = false; S.menu = null; hidePop(true); };
  const toggleMenu = (m: "coll" | "settings") => { const was = S.menu === m; closeAll(); S.menu = was ? null : m; render(); };
  const openPal = () => { closeAll(); S.pal = { open: true, query: "", sel: 0 }; render(); };
  const closePal = () => { if (S.pal.open) { S.pal.open = false; render(); } };
  const toggleTheme = () => {
    S.theme = S.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = S.theme;
    try { localStorage.setItem("gb-theme", S.theme); } catch {}
    render();
  };
  const toggleRerank = () => { S.set.rerank = !S.set.rerank; announce("Rerank " + (S.set.rerank ? "on" : "off")); render(); };

  /* ---------- collection: the Kubernetes index, built ahead of time ---------- */
  async function loadCollection() {
    S.phase = "ingesting";
    S.ing = freshIng();
    render();
    try {
      const info = await getCollection();
      S.info = info;
      S.apiDown = false;
      S.col = { id: "k8s", name: info.name, chunks: [], outline: [], total: info.chunks, docs: info.docs };
      S.ing = {
        parse: { st: "done", key: `${info.docs.toLocaleString("en-US")} docs` },
        chunk: { st: "done", key: `${info.chunks.toLocaleString("en-US")} chunks` },
        embed: { st: "done", key: "384-dim vectors" },
        index: { st: "done", key: "indexed in pgvector + full-text" },
      };
      S.phase = "ready";
      announce("Kubernetes docs loaded. You can ask a question.");
    } catch {
      S.apiDown = true;
      S.phase = "empty";
      S.ing = null;
    }
    render();
  }

  /* ---------- ask ---------- */
  const setQ = (name: QKey, st: StageState, ms?: number) => {
    S.q!.st[name] = { st, ms };
    S.fresh = name;
    render();
  };

  async function ask(text: string) {
    if (S.phase !== "ready" || S.busy || !text) return;
    S.busy = true;
    closeAll();
    const my = ++runId.current;
    const live = () => my === runId.current;
    S.msgs.push({ role: "u", text });
    const m: Ans = { role: "a", status: "streaming", parts: [], cites: [] };
    S.msgs.push(m);
    const R: Retrieval = { dense: [], kw: [], fusion: [], rerank: [], ctx: [], k: S.set.k };
    S.chunks = new Map();
    S.q = { text, R, seen: new Set(), tok: 0, st: { dense: { st: "running" }, keyword: { st: "running" }, fusion: { st: "waiting" }, rerank: { st: S.set.rerank ? "waiting" : "skipped" }, generate: { st: "waiting" }, cite: { st: "waiting" } } };
    S.stats = null;
    render();
    requestAnimationFrame(() => {
      const pipe = document.getElementById("pipe"), qry = document.getElementById("qry");
      if (pipe && qry) pipe.scrollTo({ top: qry.offsetTop - 16, behavior: RM() ? "auto" : "smooth" });
    });

    const ms: Record<string, number> = {};
    const remember = (hits: ApiHit[]): Scored[] => hits.map((h) => {
      const c = toChunk(h);
      S.chunks.set(c.id, c);
      return { c, s: h.score };
    });
    const pos = (arr: Scored[], id: string) => { const i = arr.findIndex((x) => x.c.id === id); return i < 0 ? null : i + 1; };

    const onEvent = ({ event, data }: SSE) => {
      if (!live()) return;
      const q = S.q!;
      if (event === "stage") {
        ms[data.name] = data.ms;
        const list = remember(data.hits);
        if (data.name === "dense") { R.dense = list; setQ("dense", "done", data.ms); }
        else if (data.name === "keyword") { R.kw = list; setQ("keyword", "done", data.ms); setQ("fusion", "running"); }
        else if (data.name === "fusion") {
          R.fusion = list.map((x, i) => ({ c: x.c, d: pos(R.dense, x.c.id), k: pos(R.kw, x.c.id), rrf: x.s, rank: i + 1, s: x.s, tag: "" }));
          setQ("fusion", "done", data.ms);
          if (S.set.rerank) setQ("rerank", "running");
        } else if (data.name === "rerank") {
          // Cross-encoder scores are logits; the sigmoid maps them to 0–1 bars.
          R.rerank = list.map((x, i) => ({ c: x.c, s: 1 / (1 + Math.exp(-x.s)), from: R.fusion.find((f) => f.c.id === x.c.id)?.rank ?? i + 1, to: i + 1, tag: "" }));
          setQ("rerank", "done", data.ms);
        }
      } else if (event === "context") {
        R.ctx = remember(data.hits).map((x) => x.c);
        const ctxIds = new Set(R.ctx.map((c) => c.id));
        const ctxPages = new Set(R.ctx.map((c) => c.page));
        R.rerank.forEach((r) => { r.tag = ctxIds.has(r.c.id) ? "ctx" : S.set.one && ctxPages.has(r.c.page) ? "drop" : ""; });
        R.fusion.forEach((f) => { f.tag = ctxIds.has(f.c.id) ? "ctx" : ""; });
        if (R.ctx.length) setQ("generate", "running");
      } else if (event === "token") {
        const last = m.parts[m.parts.length - 1];
        if (last && last.t === "t") last.v += data.text;
        else m.parts.push({ t: "t", v: data.text });
        q.tok++;
        render();
      } else if (event === "done") {
        m.parts = splitCitations(data.text, data.sources);
        m.cites = data.sources.map((s: { n: number; id: number }) => ({ n: s.n, id: String(s.id) }));
        m.status = data.refused ? "refusal" : "done";
        m.ms = data.ms;
        q.tout = q.tok;
        const retrieve = (ms.dense || 0) + (ms.keyword || 0) + (ms.fusion || 0);
        const rerank = ms.rerank || 0;
        S.stats = { total: data.ms, retrieve, rerank, gen: Math.max(0, data.ms - retrieve - rerank), tin: 0, tout: q.tok };
        setQ("generate", q.tok ? "done" : "skipped", q.tok ? S.stats.gen : undefined);
        setQ("cite", data.refused ? "skipped" : "done");
      } else if (event === "error") {
        m.status = "error";
        m.message = data.message;
        setQ("generate", "error");
      }
    };

    try {
      await askStream({ question: text, k: S.set.k, rerank: S.set.rerank, max_one_per_page: S.set.one }, onEvent);
      if (live() && m.status === "streaming") { m.status = "error"; m.message = UNAVAILABLE; }
    } catch (e) {
      if (!live()) return;
      m.status = "error";
      if (e instanceof ApiError && e.status === 429) {
        m.retryAt = Date.now() + (e.retryAfter || 60) * 1000;
        m.message = "Too many questions in a minute.";
      } else if (e instanceof ApiError && e.code === "busy") {
        m.message = "Two answers are already running. Retry in a few seconds.";
      } else if (e instanceof ApiError && e.status === 422) {
        m.message = "Questions must be 1 to 500 characters.";
      } else {
        m.message = UNAVAILABLE;
      }
      (Object.keys(S.q!.st) as QKey[]).forEach((k) => { if (S.q!.st[k].st === "running" || S.q!.st[k].st === "waiting") S.q!.st[k] = { st: "error" }; });
    } finally {
      if (live()) {
        S.busy = false;
        announce(m.status === "done" ? "Answer ready with citations." : m.status === "refusal" ? "The model declined: no passage answers this." : m.message || "");
        render();
      }
    }
  }

  const retry = () => {
    S.msgs.pop();
    const u = S.msgs.pop();
    S.busy = false; S.q = null; render();
    if (u && u.role === "u") ask(u.text);
  };

  /* ---------- citations ---------- */
  const showPop = (el: HTMLElement, pinned: boolean) => {
    clearTimeout(popHide.current);
    const id = el.dataset.chunk;
    if (!id || !S.chunks.has(id)) return;
    S.pop = { id, n: el.dataset.n || "", anchor: el, pinned };
    if (pinned) S.hl = id;
    render();
  };
  const hideSoon = () => { clearTimeout(popHide.current); popHide.current = setTimeout(() => { hidePop(false); render(); }, 160); };
  const pin = (el: HTMLElement) => {
    const id = el.dataset.chunk!;
    if (S.pop?.pinned && S.pop.id === id && S.pop.anchor === el) { hidePop(true); render(); return; }
    hidePop(true);
    showPop(el, true);
    document.querySelector(`#pipe .hit[data-chunk="${id}"]`)?.scrollIntoView({ block: "nearest", behavior: RM() ? "auto" : "smooth" });
  };

  const A: Actions = {
    ask, retry, pin,
    reload: loadCollection,
    toggleChunks: () => { S.chunkOpen = !S.chunkOpen; render(); },
  };

  const cmds: Cmd[] = [
    { t: "Toggle rerank", s: S.set.rerank ? "on" : "off", k: "R", run: toggleRerank },
    { t: "Toggle max 1 chunk per page", s: S.set.one ? "on" : "off", run: () => { S.set.one = !S.set.one; render(); } },
    { t: "Ask a question", k: "/", run: () => refs.input.current?.focus() },
    { t: "Toggle theme", k: "T", run: toggleTheme },
    { t: "Open settings", k: ",", run: () => toggleMenu("settings") },
  ];
  const runCmd = (c: Cmd) => { S.pal.open = false; render(); c.run(); };

  // Latest closures for the long-lived listeners below.
  const live = useRef({ S, cmds, runCmd, openPal, closePal, closeAll, toggleTheme, toggleRerank, showPop, hideSoon, render, refs, loadCollection });
  live.current = { S, cmds, runCmd, openPal, closePal, closeAll, toggleTheme, toggleRerank, showPop, hideSoon, render, refs, loadCollection };

  /* ---------- effects ---------- */
  useEffect(() => {
    const L = () => live.current;
    S.theme = document.documentElement.dataset.theme === "light" ? "light" : "dark";

    const onKey = (e: KeyboardEvent) => {
      const { S, cmds, runCmd, openPal, closePal, closeAll, toggleTheme, toggleRerank, render, refs } = L();
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); S.pal.open ? closePal() : openPal(); return; }
      if (e.key === "Escape") { const was = S.pal.open || S.menu || S.pop?.pinned; closeAll(); render(); if (was) refs.theme.current?.focus(); return; }
      if (S.pal.open) {
        const items = paletteItems(cmds, S.pal.query);
        if (e.key === "ArrowDown") { e.preventDefault(); S.pal.sel = Math.min(items.length - 1, S.pal.sel + 1); render(); }
        else if (e.key === "ArrowUp") { e.preventDefault(); S.pal.sel = Math.max(0, S.pal.sel - 1); render(); }
        else if (e.key === "Enter") { e.preventDefault(); const c = items[Math.min(S.pal.sel, items.length - 1)]; if (c) runCmd(c); }
        else if (e.key === "Tab") e.preventDefault();
        return;
      }
      const tg = e.target as HTMLElement;
      if (tg.matches?.("input,textarea,[contenteditable]") || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "r") toggleRerank();
      else if (k === "t") toggleTheme();
      else if (k === ",") { e.preventDefault(); closeAll(); S.menu = "settings"; render(); }
      else if (k === "/") { e.preventDefault(); refs.input.current?.focus(); }
    };
    // Hovering anything tied to a chunk highlights it everywhere; citations also preview.
    const onOver = (e: PointerEvent) => {
      const { S, showPop, render } = L();
      const t = e.target as HTMLElement;
      const d = t.closest<HTMLElement>("[data-chunk]");
      if (d && S.hl !== d.dataset.chunk) { S.hl = d.dataset.chunk!; render(); }
      const c = t.closest<HTMLElement>("button.cite[data-chunk]");
      if (c && !S.pop?.pinned) showPop(c, false);
    };
    const onOut = (e: PointerEvent) => {
      const { S, hideSoon, render } = L();
      const t = e.target as HTMLElement;
      const d = t.closest<HTMLElement>("[data-chunk]");
      if (d && !d.contains(e.relatedTarget as Node)) { S.hl = S.pop?.pinned ? S.pop.id : null; render(); }
      if (t.closest("button.cite[data-chunk]") && !S.pop?.pinned) hideSoon();
    };
    const onFocusIn = (e: FocusEvent) => {
      const { S, showPop } = L();
      const c = (e.target as HTMLElement).closest<HTMLElement>("button.cite[data-chunk]");
      if (c && !S.pop?.pinned) { S.hl = c.dataset.chunk!; showPop(c, false); }
    };
    const onFocusOut = (e: FocusEvent) => {
      const { S, hideSoon, render } = L();
      if ((e.target as HTMLElement).closest("button.cite[data-chunk]") && !S.pop?.pinned) { S.hl = null; render(); hideSoon(); }
    };
    const onClick = (e: MouseEvent) => {
      const { S, render } = L();
      const t = e.target as HTMLElement;
      if (t.closest("button,a,input,textarea,#pop,.menu,.scrim")) return;
      if (S.menu || S.pop?.pinned) { S.menu = null; if (S.pop?.pinned) { S.pop = null; S.hl = null; } render(); }
    };
    // Rate-limit countdown.
    const tick = setInterval(() => {
      const { S, render } = L();
      if (S.msgs.some((m) => m.role === "a" && m.status === "error" && (m.retryAt || 0) > Date.now() - 1000)) render();
    }, 1000);

    addEventListener("keydown", onKey);
    document.addEventListener("pointerover", onOver);
    document.addEventListener("pointerout", onOut);
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("focusout", onFocusOut);
    document.addEventListener("click", onClick);
    L().loadCollection();
    return () => {
      clearInterval(tick);
      removeEventListener("keydown", onKey);
      document.removeEventListener("pointerover", onOver);
      document.removeEventListener("pointerout", onOut);
      document.removeEventListener("focusin", onFocusIn);
      document.removeEventListener("focusout", onFocusOut);
      document.removeEventListener("click", onClick);
      runId.current++;
    };
  }, [S]);

  // Animation flags apply to one paint only; keep the chat pinned to the newest message.
  useEffect(() => {
    S.fresh = null;
    if (S.q) Object.keys(S.q.st).forEach((k) => S.q!.seen.add(k));
    const c = refs.chat.current;
    if (c && S.msgs.length) c.scrollTop = c.scrollHeight;
  });

  const h = (id: string) => (S.hl === id ? "hl" : "");

  return (
    <>
      <a className="skip" href="#ask-in">Skip to question</a>
      <div className="app od-screen">
        <header className="topbar">
          <div className="brand">
            <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><rect x="2.5" y="2.5" width="15" height="15" rx="4" stroke="currentColor" strokeWidth="1.5" /><rect x="7" y="7" width="6" height="6" rx="1.5" fill="var(--ac)" /></svg>
            <span>Measured RAG</span>
          </div>
          <span className="sep" aria-hidden="true" />
          <button className="collsel" ref={refs.coll} type="button" aria-haspopup="menu" aria-expanded={S.menu === "coll"} onClick={() => toggleMenu("coll")}>
            <span className="od-truncate">{S.col ? S.col.name : "No collection"}</span>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg>
          </button>
          <div className="tb-right">
            <button className="btn palbtn" type="button" onClick={openPal} data-tip="Command palette · ⌘K" aria-label="Open command palette">Commands <kbd>⌘K</kbd></button>
            <button className="icon-btn" ref={refs.set} type="button" onClick={() => toggleMenu("settings")} data-tip="Settings · ," aria-label="Settings" aria-haspopup="dialog" aria-expanded={S.menu === "settings"}><Icon name="sliders" /></button>
            <button className="icon-btn" ref={refs.theme} type="button" onClick={toggleTheme} data-tip="Toggle theme · T" aria-label={S.theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}><Icon name={S.theme === "dark" ? "sun" : "moon"} /></button>
          </div>
        </header>
        <main className="zones" id="main">
          <SourcePanel S={S} A={A} h={h} />
          <PipelinePanel S={S} A={A} h={h} />
          <AskPanel S={S} A={A} h={h} chatRef={refs.chat} inputRef={refs.input} />
        </main>
        <RunStats S={S} />
      </div>
      <CitePopover S={S} onEnter={() => clearTimeout(popHide.current)} onLeave={hideSoon} />
      <CollMenu S={S} anchor={refs.coll.current} onClose={() => { S.menu = null; render(); }} />
      <SettingsMenu S={S} anchor={refs.set.current} onChange={render} />
      <Palette S={S} cmds={cmds} onQuery={(q) => { S.pal.query = q; S.pal.sel = 0; render(); }} onRun={runCmd} onClose={closePal} />
      <div className="sr" aria-live="polite">{S.sr}</div>
    </>
  );
}
