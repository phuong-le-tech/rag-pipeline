// Client for the FastAPI backend (src/rag/api/app.py): JSON endpoints + SSE answer stream.
import type { Chunk, Part } from "./types.ts";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export type ApiHit = { id: number; heading: string; title: string; url: string; content: string; score: number };
export type CollectionInfo = { name: string; docs: number; chunks: number; llm_model: string; embedding_model: string };
export type AskBody = { question: string; k: number; rerank: boolean; max_one_per_page: boolean };
export type ApiSource = { n: number; id: number };
export type SSE = { event: string; data: any };

export class ApiError extends Error {
  status: number;
  code: string;
  retryAfter?: number;
  constructor(status: number, code: string, retryAfter?: number) {
    super(code);
    this.status = status;
    this.code = code;
    this.retryAfter = retryAfter;
  }
}

export function toChunk(h: ApiHit): Chunk {
  const path = h.heading ? h.heading.split(" > ") : [h.title];
  return { id: String(h.id), page: h.url, url: h.url, path, text: h.content, chars: h.content.length };
}

export function parseSSE(buffer: string): { events: SSE[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events = blocks.filter(Boolean).map((block) => {
    const fields: Record<string, string> = {};
    for (const line of block.split("\n")) {
      const i = line.indexOf(": ");
      if (i > 0) fields[line.slice(0, i)] = line.slice(i + 2);
    }
    return { event: fields.event, data: JSON.parse(fields.data) };
  });
  return { events, rest };
}

// [n] matching a returned source becomes a citation; any other [n] stays text (invalid citation).
export function splitCitations(text: string, sources: ApiSource[]): Part[] {
  const parts: Part[] = [];
  const pushText = (v: string) => {
    const last = parts[parts.length - 1];
    if (last && last.t === "t") last.v += v;
    else if (v) parts.push({ t: "t", v });
  };
  text.split(/\[(\d+)\]/).forEach((seg, i) => {
    if (i % 2 === 0) return pushText(seg);
    const src = sources.find((s) => s.n === Number(seg));
    if (src) parts.push({ t: "c", n: src.n, id: String(src.id) });
    else pushText(`[${seg}]`);
  });
  return parts;
}

export async function getCollection(): Promise<CollectionInfo> {
  const r = await fetch(`${API_URL}/api/collection`);
  if (!r.ok) throw new ApiError(r.status, "unavailable");
  return r.json();
}

export async function askStream(body: AskBody, onEvent: (e: SSE) => void, signal?: AbortSignal): Promise<void> {
  const r = await fetch(`${API_URL}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) {
    const j = await r.json().catch(() => ({}));
    throw new ApiError(r.status, j.error || "unavailable", j.retry_after);
  }
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    const { events, rest } = parseSSE(buf + value);
    buf = rest;
    events.forEach(onEvent);
  }
}
