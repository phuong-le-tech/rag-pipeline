"use client";
import { useEffect, useReducer, useRef } from "react";
import { K8S_COL, outlineFrom } from "@/lib/corpus.ts";
import { compose, parseMd, retrieve } from "@/lib/mockEngine.ts";
import type { Collection, StageState } from "@/lib/types.ts";
import { Icon } from "./Icons";
import { type Cmd, CitePopover, CollMenu, Palette, paletteItems, SettingsMenu } from "./Overlays";
import { type Actions, AskPanel, PipelinePanel, RunStats, SourcePanel } from "./Panels";
import { freshIng, type IngKey, type Msg, newStore, type QKey, type Store } from "./store";

type Ans = Extract<Msg, { role: "a" }>;

export default function GlassBox() {
  const sRef = useRef<Store>(null as unknown as Store);
  if (!sRef.current) sRef.current = newStore();
  const S = sRef.current;
  const [, render] = useReducer((x: number) => x + 1, 0);
  const ids = useRef({ ing: 0, run: 0 });
  const refs = { file: useRef<HTMLInputElement>(null), coll: useRef<HTMLButtonElement>(null), set: useRef<HTMLButtonElement>(null), theme: useRef<HTMLButtonElement>(null), chat: useRef<HTMLDivElement>(null), input: useRef<HTMLTextAreaElement>(null) };
  const popHide = useRef<ReturnType<typeof setTimeout>>(undefined);

  const RM = () => typeof window !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wait = (ms: number) => (S.fast ? Promise.resolve() : new Promise<void>((r) => setTimeout(r, RM() ? Math.min(ms, 80) : ms)));
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

  /* ---------- collections + ingest ---------- */
  const resetRun = () => { ids.current.run++; S.busy = false; S.q = null; S.msgs = []; S.stats = null; S.chunkOpen = false; };
  const setCollection = (col: Collection, hold = false) => { resetRun(); S.col = col; S.srcError = null; return ingest(col, hold); };
  const useK8s = () => { closeAll(); return setCollection(K8S_COL); };
  const useFile = () => { closeAll(); if (S.upl) return setCollection(S.upl); };
  const pickFile = () => { closeAll(); render(); refs.file.current?.click(); };
  const removeFile = () => {
    const was = S.col === S.upl;
    S.upl = null;
    if (was) { ids.current.ing++; resetRun(); S.col = null; S.phase = "empty"; S.ing = null; }
    S.srcError = null;
    render();
  };

  async function ingest(col: Collection, hold: boolean) {
    const my = ++ids.current.ing;
    S.phase = "ingesting";
    S.ing = freshIng();
    const k8 = col.id === "k8s", n = col.chunks.length, len = col.chunks.reduce((a, c) => a + c.chars, 0);
    const heads = k8 ? 1652 : col.heads || 0;
    const plan: Record<IngKey, { w: number; ms: number; key: string }> = k8 ? {
      parse: { w: 450, ms: 1184, key: "1,652 docs" }, chunk: { w: 550, ms: 2310, key: "17,730 chunks" },
      embed: { w: 1500, ms: 48902, key: "384-dim vectors" }, index: { w: 550, ms: 6418, key: "indexed in pgvector + full-text" },
    } : {
      parse: { w: 420, ms: Math.round(6 + len / 1800), key: heads + (heads === 1 ? " heading" : " headings") }, chunk: { w: 520, ms: 12 + n * 3, key: n + (n === 1 ? " chunk" : " chunks") },
      embed: { w: Math.min(1800, 700 + n * 25), ms: 90 + n * 38, key: "384-dim vectors" }, index: { w: 480, ms: 24 + n * 5, key: "indexed in pgvector + full-text" },
    };
    render();
    for (const key of ["parse", "chunk", "embed", "index"] as IngKey[]) {
      const p = plan[key];
      S.ing[key] = { st: "running", t0: performance.now(), frac: 0 };
      S.fresh = key;
      announce(`${key} running`);
      render();
      if (key === "embed") {
        await runEmbed(p.w, my, hold);
        if (my !== ids.current.ing || hold) return;
      } else {
        await wait(p.w);
        if (my !== ids.current.ing) return;
      }
      S.ing[key] = { st: "done", ms: p.ms, key: p.key };
      S.fresh = key;
      render();
    }
    S.phase = "ready";
    announce("Index ready. You can ask a question.");
    render();
  }

  function runEmbed(w: number, my: number, hold: boolean) {
    return new Promise<void>((res) => {
      const t0 = performance.now(), dur = S.fast ? 0 : RM() ? 80 : w;
      const step = () => {
        if (my !== ids.current.ing || !S.ing) return res();
        const f = dur ? Math.min(1, (performance.now() - t0) / dur) : 1;
        S.ing.embed.frac = hold ? f * 0.62 : f;
        render();
        if (f >= 1) return res();
        setTimeout(step, 60);
      };
      step();
    });
  }

  async function handleFile(f: File) {
    closeAll();
    S.srcError = null;
    const bad = !/\.(md|markdown)$/i.test(f.name) ? `${f.name} is not Markdown. Upload a .md or .markdown file.` : f.size > 2097152 ? `${f.name} is larger than 2 MB.` : f.size === 0 ? `${f.name} is empty.` : null;
    if (bad) {
      S.srcError = bad;
      if (!S.col) { S.phase = "empty"; S.ing = freshIng(); S.ing.parse = { st: "error", key: f.name.length > 22 ? "Not a Markdown file" : "Not Markdown" }; }
      announce(bad);
      render();
      return;
    }
    const my = ++ids.current.ing;
    resetRun();
    S.col = null;
    S.phase = "uploading";
    S.up = { name: f.name, size: f.size, p: 0 };
    S.ing = null;
    render();
    for (let i = 1; i <= 4; i++) { await wait(110); if (my !== ids.current.ing) return; S.up.p = i / 4; render(); }
    const text = await f.text();
    if (my !== ids.current.ing) return;
    const { chunks, heads } = parseMd(f.name, text);
    if (!chunks.length) { S.phase = "empty"; S.srcError = `${f.name} has no readable text.`; S.ing = freshIng(); S.ing.parse = { st: "error", key: "Nothing to parse" }; render(); return; }
    const col: Collection = { id: "upload", name: f.name, size: f.size, chunks, heads, outline: outlineFrom(chunks), total: chunks.length, docs: 1 };
    S.upl = col;
    S.col = col;
    ingest(col, false);
  }

  /* ---------- ask ---------- */
  const setQ = (name: QKey, st: StageState, ms?: number) => {
    S.q!.st[name] = { st, ms };
    S.fresh = name;
    render();
  };

  async function ask(text: string) {
    if (S.phase !== "ready" || S.busy || !text || !S.col) return;
    S.busy = true;
    closeAll();
    const my = ++ids.current.run, R = retrieve(S.col, S.set, text);
    S.msgs.push({ role: "u", text });
    const m: Ans = { role: "a", status: "streaming", parts: [], cites: [] };
    S.msgs.push(m);
    S.q = { text, R, seen: new Set(), tok: 0, st: { dense: { st: "running" }, keyword: { st: "running" }, fusion: { st: "waiting" }, rerank: { st: "waiting" }, generate: { st: "waiting" }, cite: { st: "waiting" } } };
    S.stats = null;
    render();
    requestAnimationFrame(() => {
      const pipe = document.getElementById("pipe"), qry = document.getElementById("qry");
      if (pipe && qry) pipe.scrollTo({ top: qry.offsetTop - 16, behavior: RM() || S.fast ? "auto" : "smooth" });
    });
    const live = () => my === ids.current.run;
    await wait(240); if (!live()) return; setQ("keyword", "done", R.ms.kw);
    await wait(140); if (!live()) return; setQ("dense", "done", R.ms.dn);
    setQ("fusion", "running"); await wait(220); if (!live()) return; setQ("fusion", "done", R.ms.fu);
    let rr = 0;
    if (S.set.rerank) { setQ("rerank", "running"); await wait(520); if (!live()) return; rr = R.ms.rr; setQ("rerank", "done", rr); }
    else setQ("rerank", "skipped");
    const retrieveMs = Math.max(R.ms.dn, R.ms.kw) + R.ms.fu;
    const fin = (gen: number, tin: number, tout: number) => {
      S.stats = { total: retrieveMs + rr + gen + (R.refused ? 0 : R.ms.ci), retrieve: retrieveMs, rerank: rr, gen, tin, tout };
      m.ms = S.stats.total;
    };
    if (R.refused) {
      setQ("generate", "skipped"); setQ("cite", "skipped");
      m.status = "refusal"; m.best = R.best; fin(0, 0, 0);
    } else if (S.set.inject) {
      S.set.inject = false;
      setQ("generate", "error");
      m.status = "error"; m.retryAt = Date.now() + 30000;
    } else {
      await wait(260); if (!live()) return;
      const { parts, cites } = compose(R.qt, R.ctx);
      const tin = 160 + Math.round(R.ctx.reduce((a, c) => a + c.chars, 0) / 4);
      setQ("generate", "running");
      let tok = 0;
      for (const p of parts) {
        if (p.t === "c") { m.parts.push(p); render(); await wait(70); if (!live()) return; continue; }
        const cur = { t: "t" as const, v: "" };
        m.parts.push(cur);
        for (const w of p.v.match(/\s+|`[^`]*`\S*|\S+/g) || []) {
          cur.v += w;
          if (/\S/.test(w)) { tok++; await wait(26); if (!live()) return; S.q.tok = tok; render(); }
        }
      }
      const words = parts.filter((p) => p.t === "t").reduce((a, p) => a + (p.t === "t" ? p.v.split(/\s+/).length : 0), 0);
      const tout = Math.round(words * 1.35), gen = 600 + words * 22;
      m.cites = cites; S.q.tin = tin; S.q.tout = tout;
      setQ("generate", "done", gen);
      setQ("cite", "running"); await wait(160); if (!live()) return;
      m.status = "done"; fin(gen, tin, tout);
      setQ("cite", "done", R.ms.ci);
    }
    S.busy = false;
    announce(m.status === "done" ? "Answer ready with citations." : m.status === "refusal" ? "Refused: no passage supported an answer." : "LLM rate limited.");
    render();
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
    if (!id || !S.col?.chunks.some((c) => c.id === id)) return;
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
    pickFile, useK8s, useFile, removeFile, ask, retry, pin,
    toggleChunks: () => { S.chunkOpen = !S.chunkOpen; render(); },
    outline: (id) => {
      if (S.ing?.chunk.st === "done") S.chunkOpen = true;
      S.hl = id;
      render();
      requestAnimationFrame(() => document.querySelector(`#strip .chunk[data-chunk="${id}"]`)?.scrollIntoView({ block: "nearest", inline: "center", behavior: RM() ? "auto" : "smooth" }));
    },
  };

  const cmds: Cmd[] = [
    { t: "Upload Markdown file", k: "U", run: () => refs.file.current?.click() },
    { t: "Switch collection: Kubernetes docs", k: "K", run: useK8s },
    ...(S.upl ? [{ t: "Switch collection: " + S.upl.name, run: useFile }] : []),
    { t: "Toggle rerank", s: S.set.rerank ? "on" : "off", k: "R", run: toggleRerank },
    { t: "Toggle max 1 chunk per page", s: S.set.one ? "on" : "off", run: () => { S.set.one = !S.set.one; render(); } },
    { t: "Ask a question", k: "/", run: () => refs.input.current?.focus() },
    { t: "Toggle theme", k: "T", run: toggleTheme },
    { t: "Open settings", k: ",", run: () => toggleMenu("settings") },
  ];
  const runCmd = (c: Cmd) => { S.pal.open = false; render(); c.run(); };

  // Latest closures for the long-lived listeners below.
  const live = useRef({ S, cmds, runCmd, openPal, closePal, closeAll, toggleTheme, toggleRerank, useK8s, handleFile, showPop, hideSoon, render, setCollection, ask, refs });
  live.current = { S, cmds, runCmd, openPal, closePal, closeAll, toggleTheme, toggleRerank, useK8s, handleFile, showPop, hideSoon, render, setCollection, ask, refs };

  /* ---------- effects ---------- */
  useEffect(() => {
    const L = () => live.current;
    S.theme = document.documentElement.dataset.theme === "light" ? "light" : "dark";

    const onKey = (e: KeyboardEvent) => {
      const { S, cmds, runCmd, openPal, closePal, closeAll, toggleTheme, toggleRerank, useK8s, render, refs } = L();
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
      if (k === "u") { e.preventDefault(); refs.file.current?.click(); }
      else if (k === "k") { e.preventDefault(); useK8s(); }
      else if (k === "r") toggleRerank();
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
    const onDragOver = (e: DragEvent) => { e.preventDefault(); const { S, render } = L(); if (!S.dragOver) { S.dragOver = true; render(); } };
    const onDragLeave = () => { const { S, render } = L(); S.dragOver = false; render(); };
    const onDrop = (e: DragEvent) => { e.preventDefault(); const { S, handleFile, render } = L(); S.dragOver = false; render(); const f = e.dataTransfer?.files[0]; if (f) handleFile(f); };
    // Boot: #empty (default), #ingesting, #answer
    const boot = () => {
      const { S, closeAll, setCollection, ask, render } = L();
      closeAll(); ids.current.ing++; ids.current.run++;
      Object.assign(S, { col: null, phase: "empty", ing: null, srcError: null, q: null, msgs: [], stats: null, busy: false, chunkOpen: false });
      const h = location.hash.slice(1);
      if (h === "ingesting") setCollection(K8S_COL, true);
      else if (h === "answer") { S.fast = true; setCollection(K8S_COL).then(() => ask("How do I roll back a failed Deployment?")).then(() => { S.fast = false; }); }
      else render();
    };
    // Live stage timers and rate-limit countdown.
    const tick = setInterval(() => {
      const { S, render } = L();
      const running = S.ing && Object.values(S.ing).some((o) => o.st === "running");
      const counting = S.msgs.some((m) => m.role === "a" && m.status === "error" && (m.retryAt || 0) > Date.now() - 1000);
      if (running || counting) render();
    }, 80);

    addEventListener("keydown", onKey);
    document.addEventListener("pointerover", onOver);
    document.addEventListener("pointerout", onOut);
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("focusout", onFocusOut);
    document.addEventListener("click", onClick);
    addEventListener("dragover", onDragOver);
    addEventListener("dragleave", onDragLeave);
    addEventListener("drop", onDrop);
    addEventListener("hashchange", boot);
    boot();
    return () => {
      clearInterval(tick);
      removeEventListener("keydown", onKey);
      document.removeEventListener("pointerover", onOver);
      document.removeEventListener("pointerout", onOut);
      document.removeEventListener("focusin", onFocusIn);
      document.removeEventListener("focusout", onFocusOut);
      document.removeEventListener("click", onClick);
      removeEventListener("dragover", onDragOver);
      removeEventListener("dragleave", onDragLeave);
      removeEventListener("drop", onDrop);
      removeEventListener("hashchange", boot);
      ids.current.ing++; ids.current.run++;
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
            <span>Glass-box RAG</span>
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
      <CollMenu S={S} anchor={refs.coll.current} onK8s={useK8s} onFile={useFile} onPick={pickFile} />
      <SettingsMenu S={S} anchor={refs.set.current} onChange={render} />
      <Palette S={S} cmds={cmds} onQuery={(q) => { S.pal.query = q; S.pal.sel = 0; render(); }} onRun={runCmd} onClose={closePal} />
      <input type="file" ref={refs.file} accept=".md,.markdown,text/markdown" hidden onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) handleFile(f); }} />
      <div className="sr" aria-live="polite">{S.sr}</div>
    </>
  );
}
