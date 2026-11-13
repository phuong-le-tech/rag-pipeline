// Publisher identity for the legal pages, set in the hosting environment (kept out of git).
export const PUBLISHER = process.env.NEXT_PUBLIC_PUBLISHER_NAME || "";
export const CONTACT = process.env.NEXT_PUBLIC_CONTACT_EMAIL || "";
export const UPDATED = "2026-10-26";
// Absolute base for share links, robots and the sitemap. Vercel sets this at build time.
export const SITE_URL = process.env.VERCEL_PROJECT_PRODUCTION_URL
  ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
  : "http://localhost:3000";
export const DESCRIPTION =
  "Ask questions over the Kubernetes docs or your own documents, and watch every RAG stage: hybrid search, reranking and cited answers.";
