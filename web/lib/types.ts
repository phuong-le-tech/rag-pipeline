export type Chunk = {
  id: string;
  page: string;
  url: string;
  path: string[];
  text: string;
  chars: number;
};

export type OutlineItem = { d: number; label: string; id: string };

export type Collection = {
  id: string; // "k8s" or the 32-hex id of an upload
  name: string;
  chunks: Chunk[];
  outline: OutlineItem[];
  total: number;
  docs: number;
  heads?: number;
  size?: number;
};

export type Settings = { k: number; rerank: boolean; one: boolean };

export type Scored = { c: Chunk; s: number };
export type Fused = { c: Chunk; d: number | null; k: number | null; rrf: number; rank: number; s: number; tag: string };
export type Reranked = { c: Chunk; s: number; from: number; to: number; tag: "" | "ctx" | "drop" };

// Built incrementally from the API's stage events.
export type Retrieval = {
  dense: Scored[];
  kw: Scored[];
  fusion: Fused[];
  rerank: Reranked[];
  ctx: Chunk[];
  k: number;
};

export type Part = { t: "t"; v: string } | { t: "c"; n: number; id: string };
export type Cite = { n: number; id: string };

export type StageState = "waiting" | "running" | "done" | "error" | "skipped";
