import type { ReactNode } from "react";

// One stroke family, same paths as the design prototype.
const IC: Record<string, ReactNode> = {
  file: <path d="M7 3h7l5 5v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM14 3v5h5" />,
  x: <path d="M6 6l12 12M18 6 6 18" />,
  chev: <path d="m6 9 6 6 6-6" />,
  sun: (<><circle cx="12" cy="12" r="4" /><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6 7 7M17 17l1.4 1.4M5.6 18.4 7 17M17 7l1.4-1.4" /></>),
  moon: <path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z" />,
  alert: (<><path d="M12 4 3 20h18L12 4z" /><path d="M12 10v4M12 17v.01" /></>),
  ban: (<><circle cx="12" cy="12" r="8" /><path d="m6.3 6.3 11.4 11.4" /></>),
  sliders: <path d="M4 7h10M18 7h2M4 17h2M10 17h10M14 5v4M6 15v4" />,
};

export function Icon({ name }: { name: keyof typeof IC }) {
  return <svg viewBox="0 0 24 24" aria-hidden="true">{IC[name]}</svg>;
}

// Inline `code` spans, as the prototype's md() helper.
export function Md({ text }: { text: string }) {
  return <>{text.split(/`([^`]+)`/g).map((s, i) => (i % 2 ? <code key={i}>{s}</code> : s))}</>;
}
