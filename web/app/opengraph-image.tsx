import { ImageResponse } from "next/og";
import { DESCRIPTION } from "@/lib/site.ts";

export const alt = "Measured RAG";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

// Static share card, rendered at build time in the site's dark palette.
export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "center", padding: 80, background: "#0B0B0F", color: "#ECECF1" }}>
        <div style={{ display: "flex", gap: 12, marginBottom: 40 }}>
          {["#ECECF1", "#D97757", "#8A8A98"].map((c, i) => (
            <div key={c} style={{ width: [120, 80, 100][i], height: 14, borderRadius: 7, background: c }} />
          ))}
        </div>
        <div style={{ fontSize: 84, fontWeight: 700, letterSpacing: -2 }}>Measured RAG</div>
        <div style={{ fontSize: 34, color: "#A3A3B1", marginTop: 24, lineHeight: 1.35, maxWidth: 980 }}>{DESCRIPTION}</div>
      </div>
    ),
    size,
  );
}
