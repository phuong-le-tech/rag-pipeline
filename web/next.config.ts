import type { NextConfig } from "next";

const api = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
// The legal notice and privacy page must name a publisher and a contact in production (GDPR art. 13).
if (process.env.VERCEL_ENV === "production") {
  const missing = ["NEXT_PUBLIC_PUBLISHER_NAME", "NEXT_PUBLIC_CONTACT_EMAIL", "NEXT_PUBLIC_API_URL"].filter((k) => !process.env[k]);
  if (missing.length) throw new Error(`Missing production env: ${missing.join(", ")}`);
}
const dev = process.env.NODE_ENV !== "production";
const csp = [
  "default-src 'self'",
  // 'unsafe-inline': the theme script must run before first paint; dev tooling needs eval.
  `script-src 'self' 'unsafe-inline'${dev ? " 'unsafe-eval'" : ""}`,
  "style-src 'self' 'unsafe-inline'",
  "font-src 'self'",
  "img-src 'self' data:",
  `connect-src 'self' ${api}${dev ? " ws:" : ""}`,
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join("; ");

const config: NextConfig = {
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "Content-Security-Policy", value: csp },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
          { key: "Strict-Transport-Security", value: "max-age=31536000; includeSubDomains" },
        ],
      },
    ];
  },
};

export default config;
