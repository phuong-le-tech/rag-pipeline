import type { MetadataRoute } from "next";
import { SITE_URL } from "@/lib/site.ts";

export default function sitemap(): MetadataRoute.Sitemap {
  return ["", "/privacy", "/legal"].map((path) => ({ url: `${SITE_URL}${path}` }));
}
