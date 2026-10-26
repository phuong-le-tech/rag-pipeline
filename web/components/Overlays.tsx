"use client";
import { useLayoutEffect, useRef } from "react";
import { Md } from "./Icons";
import { cx, type Store } from "./store";

// Fixed-position float under (or above) its anchor, clamped to the viewport.
function useFloat(anchor: HTMLElement | null, on: boolean, alignRight: boolean) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !anchor || !on) return;
    const place = () => {
      const r = anchor.getBoundingClientRect(), w = el.offsetWidth, h = el.offsetHeight;
      let x = alignRight ? r.right - w : r.left;
      x = Math.min(Math.max(8, x), innerWidth - w - 8);
      let y = r.bottom + 8;
      if (y + h > innerHeight - 8) y = Math.max(8, r.top - h - 8);
      el.style.transform = `translate(${x}px,${y}px)`;
    };
    place();
    addEventListener("resize", place);
    return () => removeEventListener("resize", place);
  });
  return ref;
}

export function CitePopover({ S, onEnter, onLeave }: { S: Store; onEnter: () => void; onLeave: () => void }) {
  const p = S.pop;
  const c = p ? S.chunks.get(p.id) : undefined;
  const ref = useFloat(p?.anchor || null, !!c, false);
  const r = c && S.q?.R.rerank.find((x) => x.c.id === c.id);
  const url = c && /^https?:/.test(c.url) ? c.url : null;
  return (
    <div className={cx("pop", c && "on")} ref={ref} role="dialog" aria-label="Cited passage" aria-hidden={!c} onPointerEnter={onEnter} onPointerLeave={onLeave}>
      {c && (
        <>
          <div className="od-row">
            <span className="cite od-fixed">{p!.n || "·"}</span>
            <span className="pp mono od-fill">{c.path.join(" › ")}</span>
            {r && <span className="mono tx3 od-fixed">{r.s.toFixed(2)}</span>}
          </div>
          <blockquote className="pq"><Md text={c.text} /></blockquote>
          <div className="od-row pf">
            {url ? <a className="mono od-fill od-truncate" href={url} target="_blank" rel="noopener noreferrer">{url.replace("https://", "")}</a>
              : <span className="mono tx3 od-fill od-truncate">{c.page}</span>}
            <span className="mono tx3 od-fixed">{c.id} · {c.chars} chars</span>
          </div>
        </>
      )}
    </div>
  );
}

type MenuProps = { S: Store; anchor: HTMLElement | null };

export function CollMenu({ S, anchor, onClose }: MenuProps & { onClose: () => void }) {
  const on = S.menu === "coll";
  const ref = useFloat(anchor, on, false);
  useLayoutEffect(() => { if (on) ref.current?.querySelector<HTMLElement>(".mi")?.focus(); }, [on, ref]);
  return (
    <div className={cx("menu", on && "on")} ref={ref} role="menu" aria-label="Collections" aria-hidden={!on}>
      {on && (
        <>
          <div className="mh lbl">Collections</div>
          <button className="mi" type="button" role="menuitemradio" aria-checked={!!S.col} onClick={onClose}>
            <span className="ck">{S.col ? "✓" : ""}</span><span className="od-fill">Kubernetes docs</span>
          </button>
          <div className="mh lbl">Your own files: coming soon</div>
        </>
      )}
    </div>
  );
}

export function SettingsMenu({ S, anchor, onChange }: MenuProps & { onChange: () => void }) {
  const on = S.menu === "settings";
  const ref = useFloat(anchor, on, true);
  useLayoutEffect(() => { if (on) ref.current?.querySelector<HTMLElement>(".stepper button")?.focus(); }, [on, ref]);
  const set = S.set;
  const step = (d: number) => { set.k = Math.max(3, Math.min(10, set.k + d)); onChange(); };
  const Sw = ({ k, label, hint }: { k: "rerank" | "one"; label: string; hint: string }) => (
    <div className="od-row setrow">
      <div className="od-field od-fill"><span className="sl">{label}</span><span className="sh">{hint}</span></div>
      <button className="switch od-fixed" type="button" role="switch" aria-checked={set[k]} aria-label={label} onClick={() => { set[k] = !set[k]; onChange(); }}><i /></button>
    </div>
  );
  return (
    <div className={cx("menu", on && "on")} id="settings" ref={ref} role="dialog" aria-label="Settings" aria-hidden={!on}>
      <div className="lbl" style={{ padding: "4px 0 8px" }}>Retrieval settings</div>
      <div className="od-row setrow">
        <div className="od-field od-fill"><span className="sl">Hits per search (k)</span><span className="sh">Top passages each retriever returns</span></div>
        <div className="stepper od-fixed">
          <button type="button" onClick={() => step(-1)} aria-label="Decrease k">−</button>
          <output className="mono">{set.k}</output>
          <button type="button" onClick={() => step(1)} aria-label="Increase k">+</button>
        </div>
      </div>
      <Sw k="rerank" label="Rerank" hint="Cross-encoder reorders fused passages · R" />
      <Sw k="one" label="Max 1 chunk per page" hint="Drop repeat chunks from the same page" />
      {S.info?.impls?.includes("langchain") && (
        <div className="od-row setrow">
          <div className="od-field od-fill"><span className="sl">LangChain</span><span className="sh">Answer with the LangChain pipeline instead of the hand-written one</span></div>
          <button className="switch od-fixed" type="button" role="switch" aria-checked={set.impl === "langchain"} aria-label="LangChain" onClick={() => { set.impl = set.impl === "langchain" ? "handwritten" : "langchain"; onChange(); }}><i /></button>
        </div>
      )}
    </div>
  );
}

export type Cmd = { t: string; s?: string; k?: string; run: () => void };

export function Palette({ S, cmds, onQuery, onRun, onClose }: { S: Store; cmds: Cmd[]; onQuery: (q: string) => void; onRun: (c: Cmd) => void; onClose: () => void }) {
  const { open, query, sel } = S.pal;
  const items = paletteItems(cmds, query);
  const i = Math.min(sel, Math.max(0, items.length - 1));
  const inRef = useRef<HTMLInputElement>(null);
  useLayoutEffect(() => { if (open) inRef.current?.focus(); }, [open]);
  return (
    <div className={cx("scrim", open && "on")} aria-hidden={!open} onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="pal" role="dialog" aria-modal="true" aria-label="Command palette">
        <div className="pal-in">
          <label htmlFor="pal-in">Command</label>
          <input id="pal-in" ref={inRef} role="combobox" aria-expanded="true" aria-controls="pal-list" aria-autocomplete="list" placeholder="Type to filter…" autoComplete="off"
            value={query} onChange={(e) => onQuery(e.target.value)} aria-activedescendant={items.length ? "pi" + i : ""} />
        </div>
        <ul className="pal-list" id="pal-list" role="listbox" aria-label="Commands">
          {items.length ? items.map((c, j) => (
            <li key={c.t} className="pal-it" role="option" id={"pi" + j} aria-selected={j === i} onClick={() => onRun(c)}>
              <span className="od-fill">{c.t}</span>{c.s && <span className="st mono">{c.s}</span>}{c.k && <kbd>{c.k}</kbd>}
            </li>
          )) : <li className="pal-it"><span className="st">No matching command</span></li>}
        </ul>
        <div className="pal-foot"><span><kbd>↑</kbd> <kbd>↓</kbd> navigate</span><span><kbd>⏎</kbd> run</span><span><kbd>esc</kbd> close</span></div>
      </div>
    </div>
  );
}

export const paletteItems = (cmds: Cmd[], q: string) => cmds.filter((c) => c.t.toLowerCase().includes(q.trim().toLowerCase()));
