import type { CollectionInfo } from "@/lib/api.ts";
import type { Chunk, Cite, Collection, Impl, Part, Retrieval, Settings, StageState, Step } from "@/lib/types.ts";

export type IngKey = "parse" | "chunk" | "embed" | "index";
export type QKey = "dense" | "keyword" | "fusion" | "rerank" | "generate" | "cite";
export type IngStage = { st: StageState; ms?: number; key?: string; t0?: number; frac?: number };
export type Ing = Record<IngKey, IngStage>;
export type QState = {
  text: string;
  R: Retrieval;
  st: Record<QKey, { st: StageState; ms?: number }>;
  seen: Set<string>;
  tok: number;
  tin?: number;
  tout?: number;
};
export type Msg =
  | { role: "u"; text: string }
  | { role: "a"; status: "streaming" | "done" | "refusal" | "error"; parts: Part[]; cites: Cite[]; ms?: number; message?: string; retryAt?: number; impl?: Impl; agent?: boolean; steps?: Step[]; searches?: number };
export type Stats = { total: number; retrieve: number; rerank: number; gen: number; tin: number; tout: number };
export type PopState = { id: string; n: string; anchor: HTMLElement; pinned: boolean } | null;

// Mutable view-model, mirrored from the prototype: actions mutate it, then call render().
export type Store = {
  col: Collection | null;
  up: { col: Collection; ing: Ing } | null; // the session's one upload, memory only
  info: CollectionInfo | null;
  apiDown: boolean;
  chunks: Map<string, Chunk>; // every passage seen in the current query, for citations
  phase: "empty" | "ingesting" | "ready";
  srcError: string | null;
  ing: Ing | null;
  chunkOpen: boolean;
  q: QState | null;
  msgs: Msg[];
  busy: boolean;
  fast: boolean;
  hl: string | null;
  fresh: string | null;
  stats: Stats | null;
  set: Settings;
  theme: "dark" | "light";
  menu: "coll" | "settings" | null;
  pal: { open: boolean; query: string; sel: number };
  pop: PopState;
  sr: string;
};

export const newStore = (): Store => ({
  col: null, up: null, info: null, apiDown: false, chunks: new Map(), phase: "empty", srcError: null, ing: null, chunkOpen: false,
  q: null, msgs: [], busy: false, fast: false, hl: null, fresh: null, stats: null,
  set: { k: 5, rerank: true, one: true, impl: "handwritten", agent: false }, // same defaults as `rag ask` and the API
  theme: "dark", menu: null, pal: { open: false, query: "", sel: 0 }, pop: null, sr: "",
});

export const freshIng = (): Ing => ({ parse: { st: "waiting" }, chunk: { st: "waiting" }, embed: { st: "waiting" }, index: { st: "waiting" } });

export const STL: Record<StageState, string> = { waiting: "Waiting", running: "Running", done: "Done", error: "Error", skipped: "Skipped" };
export const fmtMs = (n: number) => Math.round(n).toLocaleString("en-US") + " ms";
export const fmtSize = (b: number) => (b < 1024 ? b + " B" : b < 1048576 ? (b / 1024).toFixed(1) + " KB" : (b / 1048576).toFixed(1) + " MB");
export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");
